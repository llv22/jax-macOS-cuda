# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
from ..config import config
from .. import core
from ..core import Trace, Tracer
from ..abstract_arrays import ShapedArray, raise_to_shaped
from ..ad_util import add_jaxvals, add_jaxvals_p, zeros_like_jaxval, zeros_like_p
from .. import linear_util as lu
from ..util import (unzip2, partial, safe_map, wrap_name, split_list,
                    canonicalize_axis, moveaxis)
from . import xla
from . import partial_eval as pe

map = safe_map


def batch(fun: lu.WrappedFun, in_vals, in_dims, out_dim_dests, axis_name):
  # executes a batched version of `fun` following out_dim_dests
  batched_fun = batch_fun(fun, in_dims, out_dim_dests, axis_name=axis_name)
  return batched_fun.call_wrapped(*in_vals)

@lu.transformation_with_aux
def batch_subtrace(main, in_dims, *in_vals, **params):
  trace = main.with_cur_sublevel()
  in_tracers = [BatchTracer(trace, val, dim) if dim is not None else val
                for val, dim in zip(in_vals, in_dims)]
  outs = yield in_tracers, params
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_dims = unzip2((t.val, t.batch_dim) for t in out_tracers)
  yield out_vals, out_dims


def batch_fun(fun : lu.WrappedFun, in_dims, out_dim_dests, axis_name,
              sum_match=False):
  # transformation version of batch, which doesn't call the function
  fun, out_dims = batch_subtrace(fun)
  return _batch_fun(fun, axis_name, sum_match, in_dims, out_dims, out_dim_dests)

@lu.transformation
def _batch_fun(axis_name, sum_match, in_dims, out_dims_thunk, out_dim_dests,
               *in_vals, **params):
  in_dims = in_dims() if callable(in_dims) else in_dims
  in_dims = [
    canonicalize_axis(dim, np.ndim(val)) if isinstance(dim, int) else dim
    for val, dim in zip(in_vals, in_dims)]
  size, = {x.shape[d] for x, d in zip(in_vals, in_dims) if d is not not_mapped}
  with core.new_main(BatchTrace, axis_name=axis_name) as main:
    with core.extend_axis_env(axis_name, size, main):
      out_vals = yield (main, in_dims,) + in_vals, params
    del main
  out_dim_dests = out_dim_dests() if callable(out_dim_dests) else out_dim_dests
  out_dims = out_dims_thunk()
  for od, od_dest in zip(out_dims, out_dim_dests):
    if od is not None and not isinstance(od_dest, int) and not sum_match:
      msg = f"vmap has mapped output but out_axes is {od_dest}"
      raise ValueError(msg)
  out_vals = map(partial(matchaxis, size, sum_match=sum_match), out_dims, out_dim_dests, out_vals)
  yield out_vals

def batch_fun2(fun : lu.WrappedFun, in_dims):
  # like `batch_fun` but returns output batch dims (so no out_dim_dests)
  fun, out_dims = batch_subtrace(fun)
  return _batch_fun2(fun, in_dims), out_dims

@lu.transformation
def _batch_fun2(in_dims, *in_vals, **params):
  with core.new_main(BatchTrace, axis_name=None) as main:
    out_vals = yield (main, in_dims,) + in_vals, params
    del main
  yield out_vals


### tracer

# TODO(mattjj): use a special sentinel type rather than None
NotMapped = type(None)
not_mapped = None

class BatchTracer(Tracer):
  __slots__ = ['val', 'batch_dim']

  def __init__(self, trace, val, batch_dim: Optional[int]):
    assert core.skip_checks or type(batch_dim) in (int, NotMapped)  # type: ignore
    self._trace = trace
    self.val = val
    self.batch_dim = batch_dim

  @property
  def aval(self):
    aval = raise_to_shaped(core.get_aval(self.val))
    if self.batch_dim is not_mapped:
      return aval
    else:
      if aval is core.abstract_unit:
        return aval
      elif type(aval) is ShapedArray:
        assert 0 <= self.batch_dim < aval.ndim
        new_shape = tuple(np.delete(aval.shape, self.batch_dim))
        return ShapedArray(new_shape, aval.dtype)
      else:
        raise TypeError(aval)

  def full_lower(self):
    if self.batch_dim is not_mapped:
      return core.full_lower(self.val)
    else:
      return self

class BatchTrace(Trace):
  def __init__(self, *args, axis_name):
    super().__init__(*args)
    self.axis_name = axis_name

  def pure(self, val):
    return BatchTracer(self, val, not_mapped)

  def lift(self, val):
    return BatchTracer(self, val, not_mapped)

  def sublift(self, val):
    return BatchTracer(self, val.val, val.batch_dim)

  def process_primitive(self, primitive, tracers, params):
    vals_in, dims_in = unzip2((t.val, t.batch_dim) for t in tracers)
    if all(bdim is not_mapped for bdim in dims_in):
      return primitive.bind(*vals_in, **params)
    elif config.omnistaging_enabled and primitive in collective_rules:
      axis_names = params['axis_name']
      if not isinstance(axis_names, (tuple, list)):
        axis_names = (axis_names,)
      for i, axis_name in enumerate(axis_names):
        frame = core.axis_frame(axis_name)
        if frame.main_trace is not self.main:
          continue
        # We run the split_axis rule with tracers, which is supposed to never
        # mix this axis name with another one. We will handle any invocations
        # of collectives over the vmapped axis in a recursive call from a tracer.
        if len(axis_names) > 1:
          return split_axis(primitive, axis_name, tracers, params)
        vals_out, dims_out = collective_rules[primitive](vals_in, dims_in, frame.size, **params)
        results = map(partial(BatchTracer, self), vals_out, dims_out)
        return results if primitive.multiple_results else results[0]
    # TODO(mattjj,phawkins): if no rule implemented, could vmap-via-map here
    batched_primitive = get_primitive_batcher(primitive, self.axis_name)
    val_out, dim_out = batched_primitive(vals_in, dims_in, **params)
    if primitive.multiple_results:
      return map(partial(BatchTracer, self), val_out, dim_out)
    else:
      return BatchTracer(self, val_out, dim_out)

  def process_call(self, call_primitive, f: lu.WrappedFun, tracers, params):
    assert call_primitive.multiple_results
    params = dict(params, name=wrap_name(params.get('name', f.__name__), 'vmap'))
    vals, dims = unzip2((t.val, t.batch_dim) for t in tracers)
    if all(bdim is not_mapped for bdim in dims):
      return call_primitive.bind(f, *vals, **params)
    else:
      f, dims_out = batch_subtrace(f, self.main, dims)
      vals_out = call_primitive.bind(f, *vals, **params)
      return [BatchTracer(self, v, d) for v, d in zip(vals_out, dims_out())]

  def post_process_call(self, call_primitive, out_tracers, params):
    vals, dims = unzip2((t.val, t.batch_dim) for t in out_tracers)
    main = self.main
    def todo(vals):
      trace = main.with_cur_sublevel()
      return map(partial(BatchTracer, trace), vals, dims)
    return vals, todo

  def process_map(self, map_primitive, f: lu.WrappedFun, tracers, params):
    vals, dims = unzip2((t.val, t.batch_dim) for t in tracers)
    if all(dim is not_mapped for dim in dims):
      return map_primitive.bind(f, *vals, **params)
    else:
      assert len({x.shape[d] for x, d in zip(vals, dims) if d is not not_mapped}) == 1
      # The logic for the dimension math below is as follows:
      # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
      # ??? d / in_axis ??? None                                   ??? int       ???
      # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
      # ??? None        ??? No extra axis, so in_axis unaffected               ???
      # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
      # ??? int         ??? Not mapped, so batching dim unaffected ??? See below ???
      # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
      # When both d and in_axis are defined then:
      # - If `d <= in_axis`, we have to move the `in_axis` one dimension further;
      # - If `d >  in_axis`, we have to decrement `d` (as `in_axis` will get removed).
      def fmap(x, f): return None if x is None else f(x)
      new_in_axes = tuple(fmap(in_axis,
                               lambda ax: ax + (1 if d is not None and d <= in_axis else 0))
                          for d, in_axis in zip(dims, params['in_axes']))
      new_dims = tuple(fmap(d,
                            lambda dv: dv - (1 if in_axis is not None and in_axis < d else 0))
                       for d, in_axis in zip(dims, params['in_axes']))
      f, dims_out = batch_subtrace(f, self.main, new_dims)
      vals_out = map_primitive.bind(f, *vals, **dict(params, in_axes=new_in_axes))
      # We assume that out_axes are all 0, so we increment d
      dims_out = tuple(d + 1 if d is not not_mapped else d for d in dims_out())
      return [BatchTracer(self, v, d) for v, d in zip(vals_out, dims_out)]

  def post_process_map(self, call_primitive, out_tracers, params):
    vals, dims = unzip2((t.val, t.batch_dim) for t in out_tracers)
    main = self.main
    def todo(vals):
      trace = main.with_cur_sublevel()
      return [BatchTracer(trace, v, d + 1 if d is not not_mapped else d)
              for v, d in zip(vals, dims)]
    return vals, todo

  def process_custom_jvp_call(self, prim, fun, jvp, tracers):
    in_vals, in_dims = unzip2((t.val, t.batch_dim) for t in tracers)
    fun, out_dims1 = batch_subtrace(fun, self.main, in_dims)
    jvp, out_dims2 = batch_custom_jvp_subtrace(jvp, self.main, in_dims)
    out_vals = prim.bind(fun, jvp, *in_vals)
    fst, out_dims = lu.merge_linear_aux(out_dims1, out_dims2)
    if not fst:
      assert out_dims == out_dims[:len(out_dims) // 2] * 2
      out_dims = out_dims[:len(out_dims) // 2]
    return [BatchTracer(self, v, d) for v, d in zip(out_vals, out_dims)]

  def post_process_custom_jvp_call(self, out_tracers, params):
    vals, dims = unzip2((t.val, t.batch_dim) for t in out_tracers)
    main = self.main
    def todo(vals):
      trace = main.with_cur_sublevel()
      return map(partial(BatchTracer, trace), vals, dims)
    return vals, todo

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, *, out_trees):
    in_vals, in_dims = unzip2((t.val, t.batch_dim) for t in tracers)
    fun, out_dims1 = batch_subtrace(fun, self.main, in_dims)
    fwd, out_dims2 = batch_subtrace(fwd, self.main, in_dims)
    # TODO(mattjj,apaszke): support collectives in custom_vjp?
    bwd = batch_fun(bwd, out_dims2, in_dims,
                    axis_name='__unused_axis_name', sum_match=True)
    out_vals = prim.bind(fun, fwd, bwd, *in_vals, out_trees=out_trees)
    fst, out_dims = lu.merge_linear_aux(out_dims1, out_dims2)
    if not fst:
      out_dims = out_dims[-len(out_vals) % len(out_dims):]
    return [BatchTracer(self, v, d) for v, d in zip(out_vals, out_dims)]


### primitives

BatchingRule = Callable[..., Tuple[Any, Union[int, Tuple[int, ...]]]]
primitive_batchers : Dict[core.Primitive, BatchingRule] = {}
initial_style_batchers : Dict[core.Primitive, Any] = {}

def get_primitive_batcher(p, axis_name):
  if p in initial_style_batchers:
    return partial(initial_style_batchers[p], axis_name=axis_name)
  try:
    return primitive_batchers[p]
  except KeyError as err:
    msg = "Batching rule for '{}' not implemented"
    raise NotImplementedError(msg.format(p)) from err

def defvectorized(prim):
  primitive_batchers[prim] = partial(vectorized_batcher, prim)

def vectorized_batcher(prim, batched_args, batch_dims, **params):
  assert all(batch_dims[0] == bd for bd in batch_dims[1:]), batch_dims
  return prim.bind(*batched_args, **params), batch_dims[0]

def defbroadcasting(prim):
  primitive_batchers[prim] = partial(broadcast_batcher, prim)

def broadcast_batcher(prim, args, dims, **params):
  """Process a primitive with built-in broadcasting.

  Args:
    args: the possibly-batched arguments
    dims: list or tuple of the same length as `args`, where each
      entry indicates the batching state of the corresponding entry to `args`:
      either an int indicating the batch dimension, or else `not_mapped`
      indicating no batching.
  """
  shapes = {(x.shape, d) for x, d in zip(args, dims) if np.ndim(x)}
  if len(shapes) == 1:
    # if there's only agreeing batch dims and scalars, just call the primitive
    d = next(d for d in dims if d is not not_mapped)
    out = prim.bind(*args, **params)
    return (out, (d,) * len(out)) if prim.multiple_results else (out, d)
  else:
    size, = {shape[d] for shape, d in shapes if d is not not_mapped}
    args = [bdim_at_front(x, d, size) for x, d in zip(args, dims)]
    ndim = max(np.ndim(x) for x in args)  # special-case scalar broadcasting
    args = [_handle_scalar_broadcasting(ndim, x, d) for x, d in zip(args, dims)]
    out = prim.bind(*args, **params)
    return (out, (0,) * len(out)) if prim.multiple_results else (out, 0)

def _handle_scalar_broadcasting(nd, x, d):
  if d is not_mapped or nd == np.ndim(x):
    return x
  else:
    return x.reshape(x.shape + (1,) * (nd - np.ndim(x)))

def defreducer(prim):
  primitive_batchers[prim] = partial(reducer_batcher, prim)

def reducer_batcher(prim, batched_args, batch_dims, axes, **params):
  operand, = batched_args
  bdim, = batch_dims
  axes = tuple(np.where(np.less(axes, bdim), axes, np.add(axes, 1)))
  bdim_out = int(list(np.delete(np.arange(operand.ndim), axes)).index(bdim))
  if 'input_shape' in params:
    params = dict(params, input_shape=operand.shape)
  return prim.bind(operand, axes=axes, **params), bdim_out

# sets up primitive batchers for ad_util and xla primitives

def add_batched(batched_args, batch_dims):
  bdx, bdy = batch_dims
  x, y = batched_args
  if bdx == bdy or core.get_aval(x) == core.abstract_unit:
    return add_jaxvals(x, y), bdx
  elif bdx is not_mapped:
    x = broadcast(x, y.shape[bdy], bdy)
    return add_jaxvals(x, y), bdy
  elif bdy is not_mapped:
    y = broadcast(y, x.shape[bdx], bdx)
    return add_jaxvals(x, y), bdx
  else:
    x = moveaxis(x, bdx, bdy)
    return add_jaxvals(x, y), bdy
primitive_batchers[add_jaxvals_p] = add_batched

def zeros_like_batched(batched_args, batch_dims):
  val, = batched_args
  bdim, = batch_dims
  return zeros_like_jaxval(val), bdim
primitive_batchers[zeros_like_p] = zeros_like_batched

defvectorized(xla.device_put_p)

### util

def broadcast(x, sz, axis):
  if core.get_aval(x) is core.abstract_unit:
    return core.unit
  shape = list(np.shape(x))
  shape.insert(axis, sz)
  broadcast_dims = tuple(np.delete(np.arange(len(shape)), axis))
  return jax.lax.broadcast_in_dim(x, shape, broadcast_dims)

def matchaxis(sz, src, dst, x, sum_match=False):
  if core.get_aval(x) is core.abstract_unit:
    return core.unit
  if src == dst:
    return x
  elif type(src) == type(dst) == int:
    return moveaxis(x, src, dst)
  elif src is not_mapped and dst is not not_mapped:
    return broadcast(
      x, sz, canonicalize_axis(dst, np.ndim(x) + 1))
  elif dst is None and sum_match:
    return x.sum(src)
  else:
    raise ValueError((src, dst))

def bdim_at_front(x, bdim, size):
  if core.get_aval(x) is core.abstract_unit:
    return core.unit
  if bdim is not_mapped:
    return broadcast(x, size, 0)
  else:
    return moveaxis(x, bdim, 0)


def _promote_aval_rank(sz, aval):
  if aval is core.abstract_unit:
    return core.abstract_unit
  else:
    return ShapedArray((sz,) + aval.shape, aval.dtype)

def batch_jaxpr(closed_jaxpr, size, batched, instantiate, axis_name):
  f = lu.wrap_init(core.jaxpr_as_fun(closed_jaxpr))
  f, batched_out = batched_traceable(f, size, batched, instantiate, axis_name)
  avals_in = [_promote_aval_rank(size, a) if b else a
              for a, b in zip(closed_jaxpr.in_avals, batched)]
  jaxpr_out, _, consts = pe.trace_to_jaxpr_dynamic(f, avals_in)
  return core.ClosedJaxpr(jaxpr_out, consts), batched_out()

@lu.transformation_with_aux
def batched_traceable(size, batched, instantiate, axis_name, *vals):
  in_dims = [0 if b else None for b in batched]
  with core.new_main(BatchTrace, axis_name=axis_name) as main:
    with core.extend_axis_env(axis_name, size, main):
      trace = main.with_cur_sublevel()
      ans = yield map(partial(BatchTracer, trace), vals, in_dims), {}
      out_tracers = map(trace.full_raise, ans)
      out_vals, out_dims = unzip2((t.val, t.batch_dim) for t in out_tracers)
      del main, out_tracers
  if type(instantiate) is bool:
    instantiate = [instantiate] * len(out_vals)
  out_vals = [moveaxis(x, d, 0) if d is not not_mapped and d != 0
              else broadcast(x, size, 0) if d is not_mapped and inst else x
              for x, d, inst in zip(out_vals, out_dims, instantiate)]
  out_batched = [d is not not_mapped or inst
                 for d, inst in zip(out_dims, instantiate)]
  yield out_vals, out_batched


@lu.transformation_with_aux
def batch_custom_jvp_subtrace(main, in_dims, *in_vals):
  size, = {x.shape[d] for x, d in zip(in_vals, in_dims) if d is not not_mapped}
  trace = main.with_cur_sublevel()
  in_tracers = [BatchTracer(trace, val, dim) if dim is not None else val
                for val, dim in zip(in_vals, in_dims * 2)]
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_dims = unzip2((t.val, t.batch_dim) for t in out_tracers)
  out_primals, out_tangents = split_list(out_vals, [len(out_vals) // 2])
  out_primal_bds, out_tangent_bds = split_list(out_dims, [len(out_vals) // 2])
  out_dims = map(_merge_bdims, out_primal_bds, out_tangent_bds)
  out_primals  = map(partial(matchaxis, size),  out_primal_bds, out_dims,  out_primals)
  out_tangents = map(partial(matchaxis, size), out_tangent_bds, out_dims, out_tangents)
  yield out_primals + out_tangents, out_dims * 2

def _merge_bdims(x, y):
  if x == y:
    return x
  elif x is not_mapped:
    return y
  elif y is not_mapped:
    return x
  else:
    return x  # arbitrary


@config.register_omnistaging_disabler
def omnistaging_disabler() -> None:
  global batch_jaxpr

  def batch_jaxpr(jaxpr, size, batched, instantiate, axis_name):
    f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
    f, batched_out = batched_traceable(f, size, batched, instantiate, axis_name)
    avals_in = [_promote_aval_rank(size, a) if b else a
                for a, b in zip(jaxpr.in_avals, batched)]
    in_pvals = [pe.PartialVal.unknown(aval) for aval in avals_in]
    jaxpr_out, pvals_out, consts_out = pe.trace_to_jaxpr(f, in_pvals, instantiate=True)
    avals_out, _ = unzip2(pvals_out)
    return core.ClosedJaxpr(jaxpr_out, consts_out), batched_out()


# collective_rules can assume that the collective is only carried out throughout
# the vmapped axis (i.e. no tuples in axis_name).
collective_rules: Dict[core.Primitive, Callable] = {}
split_axis_rules: Dict[core.Primitive, Callable] = {}

def split_axis(primitive, split_name, args, params):
  return split_axis_rules[primitive](split_name, args, params)
