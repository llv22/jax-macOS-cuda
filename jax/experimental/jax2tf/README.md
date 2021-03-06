# JAX to TensorFlow converter (jax2tf)

This package provides an experimental JAX converter that can take a function
written in JAX, possibly including JAX transformations, and turn it into
a function that uses only TensorFlow operations. The converted function
can be used in a TensorFlow context and will behave as if it was written in TensorFlow.
In practice this means that you can take some code written in JAX and execute it using
TensorFlow eager mode, or stage it out as a TensorFlow graph, even save it
as a SavedModel for use with TensorFlow tools such as serving stack,
or TensorFlow Hub.

We describe below some general concepts and capabilities.
More involved examples, including using jax2tf with
Flax models and their use with TensorFlow Hub and Keras, are described in the
[examples directory](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/examples/README.md).

See also some internal ongoing design discussions at `go/jax2tf-doc`.

## Usage: converting basic functions.

As a rule of thumb, if you can `jax.jit` your function then you should be able
to use `jax2tf.convert`:

```python
import jax
from jax.experimental import jax2tf
from jax import numpy as jnp

import numpy as np
import tensorflow as tf

def f_jax(x):
  return jnp.sin(jnp.cos(x))

# jax2tf.convert is a higher order function that returns a wrapped function with
# the same signature as your input function but accepting TensorFlow tensors (or
# variables) as input.
f_tf = jax2tf.convert(f_jax)

# For example you execute f_tf eagerly with valid TensorFlow inputs:
f_tf(np.random(...))

# Additionally you can use tools like `tf.function` to improve the execution
# time of your function, or to stage it out to a SavedModel:
f_tf_graph = tf.function(f_tf, autograph=False)
```

The Autograph feature of `tf.function` cannot be expected to work on
functions converted from JAX as above, so it is recommended to
set `autograph=False` in order to avoid warnings or outright errors.

## Usage: saved model

Since jax2tf provides a regular TensorFlow function using it with SavedModel
is trivial:

```python
# You can save the model just like you would with any other TensorFlow function:
my_model = tf.Module()
# Save a function that can take scalar inputs.
my_model.f = tf.function(jax2tf.convert(f_jax), input_signature=[tf.TensorSpec([], tf.float32)])
tf.saved_model.save(my_model, '/some/directory')

# Restoring (note: the restored model does *not* require JAX to run, just XLA).
restored_model = tf.saved_model.load('/some/directory')
```

An important point is that in the above code snippet **everything is standard
TensorFlow code. In particular, the saving of the model is independent of JAX,
and one can therefore set metadata and assets as needed for their application,
as if the saved function had been written directly in TensorFlow**.

Just like for regular TensorFlow functions, it is possible to include in the
SavedModel multiple versions of a function for different input shapes, by
"warming up" the function on different input shapes:

```
my_model.f = tf.function(jax2tf.convert(f_jax), autograph=False)
my_model.f(tf.ones([1, 28, 28]))  # a batch size of 1
my_model.f(tf.ones([16, 28, 28]))  # a batch size of 16
tf.saved_model.save(my_model, '/some/directory')
```

For examples of how to save a Flax or Haiku model as a SavedModel see the 
[examples directory](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/examples/README.md).


## Differentiation

The converted code supports differentiation from TensorFlow.
The main challenge with TensorFlow-differentiation of the converted code is that some
of the JAX primitives or functions that were used in the original JAX code might
have had JAX custom gradients. One example is the ``jax.nn.relu``, which
at 0 has a JAX custom gradient of 0, but the primitive-by-primitive conversion
to TensorFlow is not mathematically differentiable at 0 and may generate another value. In this
particular case the TensorFlow differentiation of the raw conversion returns 1.
If we were to use ``tf.nn.relu``, then we would get a correct custom TensorFlow gradient of 0,
because ``tf.nn.relu`` has a TensorFlow custom gradient.

Other challenges for differentiation are that some of the TensorFlow ops used in translation
do not yet have differentiation rules defined.
For other ops, we would have to ensure that they match the JAX differentiation rules.

All of these problems are solved by having the jax2tf converter
annotate the converted
function with a ``tf.custom_gradient`` that, upon TensorFlow differentiation,
will lazily
call into JAX to compute the ``jax.vjp`` of the converted function, followed by
jax2tf conversion.
This ensures that JAX???s differentiation uses the JAX rules and custom gradients.
In particular, TensorFlow???s internal op differentiation rules will not be
used at all, and we need not worry about ops not having differentiation rules
that match JAX's.
The jax2tf converter has an option ``with_gradient=False`` to skip the
custom gradients and wrap instead the converted function with
``tf.raw_ops.PreventGradient`` to generated an error in case a gradient
computation is attempted.

Currently, there is a bug that prevents using custom gradients with SavedModel
(see [Caveats](#caveats) below).

## Running on GPU

To run jax2tf on GPU, both jaxlib and TensorFlow must be installed with support
for CUDA. One must be mindful to install a version of CUDA that is compatible
with both [jaxlib](https://github.com/google/jax/blob/master/README.md#pip-installation) and
[TensorFlow](https://www.tensorflow.org/install/source#tested_build_configurations).

As of today, the tests are run using `tf_nightly==2.4.0.dev20200916`.

## Caveats

### TensorFlow XLA ops

For most JAX primitives there is a natural TF op that fits the needed semantics.
There are a few (listed below) JAX primitives for which there is no
single TF op with matching semantics.
This is not so surprising, because JAX primitives have been designed
to be compiled to [HLO ops](https://www.tensorflow.org/xla/operation_semantics),
while the corresponding TF ops are sometimes higher-level.
For the cases when there is no matching canonical TF op,
we use a set of special TF ops that are thin wrappers over HLO ops
(a subset of those registered in
[tf2xla/ops/xla_ops.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/ops/xla_ops.cc)
and implemented in,
e.g.,
[tf2xla/kernels/xla_pad_op.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/kernels/xla_pad_op.cc).)
We refer to these ops here as the TFXLA ops.

There are several drawbacks of using TFXLA ops:

   * These ops will only be executable by a consumer that has XLA linked in.
   This should not be a problem for TPU execution, since that requires XLA anyway.
   But for other platforms (CPU, GPU, embedded) this can be a drawback in certain settings.
   * These ops are not yet recognized by tools that process
   tf.Graph, e.g., TensorFlow.js converter.

We use the following such TFXLA:

   * `XlaPad` (wraps XLA Pad operator). We use this instead of `tf.pad` in order to
     support `lax.pad` interior padding (dilation) or negative edge padding.
   * `XlaConv` (wraps XLA ConvGeneralDilated operator).
   * `XlaGather` (wraps XLA Gather operator). We could use `tf.gather` in some
     cases but not always. Also, `tf.gather` has a different semantics than `lax.gather`
     for index out of bounds.
   * `XlaScatter` (wraps XLA Scatter operator).
   * `XlaSelectAndScatter` (wraps XLA SelectAndScatter operator).
   * `XlaDynamicSlice` (wraps XLA DynamicSlice operator).
     We use this instead of `tf.slice` for reasons explained above for `XlaGather`.
   * `XlaDynamicUpdateSlice` (wraps XLA DynamicUpdateSlice operator).
   * `XlaReduceWindow` (wraps XLA ReduceWindow operator). These are used
     for `lax.reduce_window_sum_p`, `lax.reduce_window_min_p`,
     `lax.reduce_window_max_p`, and `lax.reduce_window_p`.
   * `XlaSort` (wraps XLA Sort operator).

### Incomplete data type coverage

A small number of JAX primitives are converted only
for certain data types, when the required TensorFlow ops are not implemented for some
data types on certain devices. There is an
[up-to-date list of unimplemented cases](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/g3doc/primitives_with_limited_support.md).

### Missing features

There is currently no support for replicated (e.g. `pmap`) or multi-device
(e.g. `sharded_jit`) functions. The collective operations are not yet handled.

### No SavedModel fine-tuning

Currently, TensorFlow SavedModel does not properly save the `tf.custom_gradient`.
It does save however some attributes that on model restore result in a warning
that the model might not be differentiable, and trigger an error if differentiation
is attempted. The plan is to fix this. Note that if no gradients are requested,
the PreventGradient ops will be saved along with the converted code and will
give a nice error if differentiation of the converted code is attempted.

### Different performance characteristics

The converted code may have slightly different performance characteristics than
the original JAX code.
If one were to write the same functionality in JAX idiomatic code vs.
native TensorFlow idiomatic code we could end up with very different compilation paths,
when TensorFlow is used without XLA. Take for example, the case of batch normalization.
In TensorFlow if one uses [tf.nn.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization),
a ???high-level??? TensorFlow op for batch
normalization is generated, and in the absence of XLA, on CPU or GPU,
a custom C++ ???high-level??? kernel implementing batch normalization is executed.
In JAX, there is no primitive for batch normalization, and instead the
operation is decomposed into low-level primitives (e.g., [flax.nn.BatchNorm](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.BatchNorm.html#flax.nn.BatchNorm),
or haiku.BatchNorm).
Once those primitives are converted to TensorFlow, and the resulting code is
run without XLA, the ensemble of the kernels executed will quite
possibly behave differently, performance-wise or even numerically,
than either the TensorFlow native or JAX native batch normalization.
A similar example is that of an LSTM cell.

Yet another example are the PRNG primitives. JAX programs use a [stateless
deterministic PRNG](https://github.com/google/jax/blob/master/design_notes/prng.md)
and it has an internal JAX primitive for it.
This primitive is at the moment converted to a soup of tf.bitwise operations,
which has a clear performance penalty. We plan to look into using the
HLO [RNGBitGenerator](https://www.tensorflow.org/xla/operation_semantics#rngbitgenerator)
(exposed as a TFXLA op), which does implement
the same basic Threefry algorithm as JAX???s PRNG, although that would
result in different results than JAX???s PRNG.

We do expect that the performance characteristics of converted code
should approximate those of JAX or TensorFlow native with XLA. This is because
during conversion we try to generate one TensorFlow op for one JAX primitive.
We expect that the lowering that XLA does is similar to that done by JAX
before conversion. (This is a hypothesis, we have not verified it extensively.)

