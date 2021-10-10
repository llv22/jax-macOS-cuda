# Compile Jax 0.2.21 on macOS with Cuda 10.1 + cudnn7.6.5 + nccl on mac 2.9.6

## Using original source v0.2.21

open -a /Applications/Visual\ Studio\ Code.app/  /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external
status: need nccl migrate to 2.9.6-1

### issue 1: /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/org_tensorflow/third_party/gpus/cuda_configure.bzl:1448:33

disable soname
 478 def _should_check_soname(version, static):
 479     return False
 480     #return version and not static

### issue 2:  -fno-canonical-system-headers

Line 93 vim /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad//execroot/__main__/external/local_config_cuda/crosstool/BUILD

### issue 3: Line 873 from <https://github.com/bazelbuild/bazel/blob/master/tools/cpp/unix_cc_toolchain_config.bzl>

search in visual code via keyword “rcsD” and change to “-o"
vim /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad//external/rules_cc/cc/private/toolchain/unix_cc_toolchain_config.bzl
 869     archiver_flags_feature = feature(
 870         name = "archiver_flags",
 871         flag_sets = [
 872             flag_set(
 873                 actions = [ACTION_NAMES.cpp_link_static_library],
 874                 flag_groups = [
 875                     flag_group(flags = ["-o"]),
 876                     flag_group(
 877                         flags = ["%{output_execpath}"],
 878                         expand_if_available = "output_execpath",
 879                     ),
 880                 ],
 881             ),

vim /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/bazel_tools/tools/cpp/unix_cc_toolchain_config.bzl
 940     archiver_flags_feature = feature(
 941         name = "archiver_flags",
 942         flag_sets = [
 943             flag_set(
 944                 actions = [ACTION_NAMES.cpp_link_static_library],
 945                 flag_groups = [
 946                     flag_group(flags = ["-o"]),
 947                     flag_group(
 948                         flags = ["%{output_execpath}"],
 949                         expand_if_available = "output_execpath",
 950                     ),
 951                 ],
 952             ),

vim /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/bazel_tools/tools/cpp/unix_cc_toolchain_config.bzl
 940     archiver_flags_feature = feature(
 941         name = "archiver_flags",
 942         flag_sets = [
 943             flag_set(
 944                 actions = [ACTION_NAMES.cpp_link_static_library],
 945                 flag_groups = [
 946                     flag_group(flags = ["-o"]),
 947                     flag_group(
 948                         flags = ["%{output_execpath}"],
 949                         expand_if_available = "output_execpath",
 950                     ),
 951                 ],
 952             ),

vim /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/org_tensorflow/third_party/gpus/crosstool/cc_toolchain_config.bzl.tpl
 452             feature(
 453                 name = "all_archive_flags",
 454                 enabled = True,
 455                 flag_sets = [
 456                     flag_set(
 457                         actions = all_archive_actions(),
 458                         flag_groups = [
 459                             flag_group(
 460                                 expand_if_available = "linker_param_file",
 461                                 flags = ["@%{linker_param_file}"],
 462                             ),
 463                             flag_group(flags = ["-o"]),
 464                             flag_group(
 465                                 flags = ["%{output_execpath}"],
 466                                 expand_if_available = "output_execpath",
 467                             ),

the following 5 files aren’t relevant with jax key compilation: 
vim /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/org_tensorflow/third_party/gpus/crosstool/hipcc_cc_toolchain_config.bzl.tpl [optional]
vim /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/llvm-raw/clang/lib/Driver/ToolChains/Gnu.cpp [optional]
skipped /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/bazel_toolchains/configs/experimental/debian8_clang/{version}/bazel_{bazel version}/ubsan/CROSSTOOL
skipped /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/bazel_toolchains/configs/experimental/ubuntu16_04_clang/{version}/bazel_{bazel version}/ubsan/CROSSTOOL
skipped /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/bazel_toolchains/configs/ubuntu16_04_clang/{version}/bazel_{bazel version}/cc/cc_toolchain_config.bzl

after change flags, you may meet with the following issue
Building XLA and installing it in the jaxlib source tree...
/usr/local/bin/bazel run --verbose_failures=true --config=avx_posix --config=mkl_open_source_only --config=cuda :build_wheel -- --output_path=/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/jax/dist --cpu=x86_64
FATAL: corrupt installation: file '/var/tmp/_bazel_llv23/install/d07238c5957d0addf241b6be07f3c14b/embedded_tools/tools/cpp/unix_cc_toolchain_config.bzl' is missing or modified.  Please remove '/var/tmp/_bazel_llv23/install/d07238c5957d0addf241b6be07f3c14b' and try again.

Building XLA and installing it in the jaxlib source tree...
/usr/local/bin/bazel run --verbose_failures=true --config=avx_posix --config=mkl_open_source_only --config=cuda :build_wheel -- --output_path=/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/jax/dist --cpu=x86_64
FATAL: corrupt installation: file '/var/tmp/_bazel_llv23/install/d07238c5957d0addf241b6be07f3c14b/embedded_tools/tools/cpp/unix_cc_toolchain_config.bzl' is missing or modified.  Please remove '/var/tmp/_bazel_llv23/install/d07238c5957d0addf241b6be07f3c14b' and try again.

### issue 4: /bin/bash -c 'source external/bazel_tools/tools/genrule/genrule-setup.sh; cp -rLf "/Developer/NVIDIA/CUDA-10.1/nvvm/libdevice/." "bazel-out/darwin-opt/bin/external/local_config_cuda/cuda/cuda/nvvm/libdevice/" ')

Execution platform: @local_execution_config_platform//:platform
cp: the -H, -L, and -P options may not be specified with the -r option.

### issue 5:  external/org_tensorflow/tensorflow/compiler/xla/util.cc:118:7: error: no matching function for call to ‘isnan'

vim  /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/org_tensorflow/tensorflow/compiler/xla/util.cc

external/org_tensorflow/tensorflow/compiler/xla/util.cc:118:7: error: no matching function for call to 'isnan'
  if (std::isnan(value) && kPayloadBits > 0) {
      ^~~~~~~~~~
external/org_tensorflow/tensorflow/compiler/xla/util.cc:129:3: note: in instantiation of function template specialization 'xla::RoundTripNanPayload<unsigned short, Eigen::bfloat16>' requested here
  RoundTripNanPayload<uint16_t>(value, &result);
  ^
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/math.h:505:1: note: candidate template ignored: requirement 'std::is_floating_point<bfloat16>::value' was not satisfied [with _A1 = Eigen::bfloat16]
isnan(_A1 __lcpp_x) _NOEXCEPT
^
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/math.h:513:1: note: candidate template ignored: requirement 'std::is_integral<bfloat16>::value' was not satisfied [with _A1 = Eigen::bfloat16]
isnan(_A1) _NOEXCEPT
^
external/org_tensorflow/tensorflow/compiler/xla/util.cc:118:7: error: no matching function for call to 'isnan'
  if (std::isnan(value) && kPayloadBits > 0) {
      ^~~~~~~~~~
external/org_tensorflow/tensorflow/compiler/xla/util.cc:135:3: note: in instantiation of function template specialization 'xla::RoundTripNanPayload<unsigned short, Eigen::half>' requested here
  RoundTripNanPayload<uint16_t>(value, &result);
  ^
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/math.h:505:1: note: candidate template ignored: requirement 'std::is_floating_point<half>::value' was not satisfied [with _A1 = Eigen::half]
isnan(_A1 __lcpp_x) _NOEXCEPT
^
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/math.h:513:1: note: candidate template ignored: requirement 'std::is_integral<half>::value' was not satisfied [with _A1 = Eigen::half]
isnan(_A1) _NOEXCEPT

127 string RoundTripFpToString(tensorflow::bfloat16 value) {
128   std::string result = absl::StrFormat("%.4g", static_cast<float>(value));
129   //see: convert Eigen::bfloat16 to bfloat16, Eigen to /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/eigen_archive/Eigen/src/Core/arch/Default/BFloat16.h
130   RoundTripNanPayload<uint16_t>(static_cast<unsigned short>(value), &result);
131   return result;
132 }
133
134 string RoundTripFpToString(Eigen::half value) {
135   std::string result = absl::StrFormat("%.5g", static_cast<float>(value));
136   //see: convert Eigen::half to half, Eigen to /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/eigen_archive/Eigen/src/Core/arch/Default/Half.h
137   RoundTripNanPayload<uint16_t>(static_cast<unsigned short>(value), &result);
138   return result;
139 }

### issue  6: missing CHECK_ERR macro

external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:194:3: error: use of undeclared identifier 'CHECK_ERR'
  CHECK_ERR(realpath(unresolved_path, exe_path) ? 1 : -1);
Line 194
 vim  /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_gpu_executor.cc

vim /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/org_tensorflow/tensorflow/stream_executor/platform/logging.h
 30 #if defined(__APPLE__)
 31 #define CHECK_ERR(indicator_result) CHECK_ERR_INDICATOR(indicator_result)
 32 #endif

vim /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/org_tensorflow/tensorflow/core/platform/default/logging.h
265 #if defined(__APPLE__)
266 #define CHECK_ERR_INDICATOR(condition)              \
267   if (TF_PREDICT_FALSE(!(condition == 1))) \
268   LOG(FATAL) << "Check failed: " #condition " "
269 #endif

### issue 7: upgrade nccl from 2.5.7-1/2.5.8 to 2.9.6-1, because ncclSend and ncclRecv haven’t been defined in previous version

external/org_tensorflow/tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.cc:106:34: error: use of undeclared identifier 'ncclSend'
        XLA_CUDA_RETURN_IF_ERROR(ncclSend(send_buffer + rank * chunk_bytes,
                                 ^
external/org_tensorflow/tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.cc:109:34: error: use of undeclared identifier 'ncclRecv'
        XLA_CUDA_RETURN_IF_ERROR(ncclRecv(recv_buffer + rank * chunk_bytes,
                                 ^
external/org_tensorflow/tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.cc:133:32: error: use of undeclared identifier 'ncclSend'
      XLA_CUDA_RETURN_IF_ERROR(ncclSend(send_buffer, element_count, dtype,
                               ^
external/org_tensorflow/tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.cc:135:32: error: use of undeclared identifier 'ncclRecv'
      XLA_CUDA_RETURN_IF_ERROR(ncclRecv(recv_buffer, element_count, dtype,

### issue 8: XLA to load nccl library without explicitly dependency

---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
<ipython-input-2-aab6390d0645> in <module>
----> 1 import jax
      2 n_devices = jax.local_device_count()

~/opt/miniconda3/lib/python3.8/site-packages/jax/__init__.py in <module>
     35 # We want the exported object to be the class, so we first import the module
     36 # to make sure a later import doesn't overwrite the class.
---> 37 from . import config as _config_module
     38 del _config_module
     39 

~/opt/miniconda3/lib/python3.8/site-packages/jax/config.py in <module>
     16 
     17 # flake8: noqa: F401
---> 18 from jax._src.config import config

~/opt/miniconda3/lib/python3.8/site-packages/jax/_src/config.py in <module>
     25 import warnings
     26 
---> 27 from jax._src import lib
     28 from jax._src.lib import jax_jit
     29 

~/opt/miniconda3/lib/python3.8/site-packages/jax/_src/lib/__init__.py in <module>
     71 cpu_feature_guard.check_cpu_features()
     72 
---> 73 from jaxlib import xla_client
     74 from jaxlib import lapack
     75 from jaxlib import pocketfft

~/opt/miniconda3/lib/python3.8/site-packages/jaxlib/xla_client.py in <module>
     29 from typing import List, Sequence, Tuple, Union
     30 
---> 31 from . import xla_extension as _xla
     32 
     33 from absl import logging

ImportError: dlopen(/Users/llv23/opt/miniconda3/lib/python3.8/site-packages/jaxlib/xla_extension.so, 2): Symbol not found: _ncclAllGather
  Referenced from: /Users/llv23/opt/miniconda3/lib/python3.8/site-packages/jaxlib/xla_extension.so
  Expected in: flat namespace
 in /Users/llv23/opt/miniconda3/lib/python3.8/site-packages/jaxlib/xla_extension.so

open -a /Applications/Visual\ Studio\ Code.app/ /private/var/tmp/_bazel_llv23/514259ef8f75fe7e040e6e3c97f3d5ad/external/org_tensorflow/tensorflow/compiler/xla/python/BUILD
pybind_extension(
    name = "xla_extension",
    srcs = [
        "xla.cc",
    ],
    module_name = "xla_extension",
    pytype_deps = [
        "//third_party/py/numpy",
    ],
    pytype_srcs = glob(["xla_extension/*.pyi"]),
    deps = [
        ":dlpack",
        ":jax_jit",
        ":pmap_lib",
        ":ops",
        ":profiler",
        ":py_client",
        ":pytree",
        ":python_ref_manager",
        ":traceback",
        ":outfeed_receiver_py",
        ":types",
        ":xla_compiler",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@pybind11",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/pjrt:cpu_device",
        "//tensorflow/compiler/xla/pjrt:interpreter_device",
        "//tensorflow/compiler/xla/pjrt:gpu_device",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "//tensorflow/compiler/xla/pjrt:tpu_client",
        "//tensorflow/compiler/xla/pjrt/distributed",
        "//tensorflow/compiler/xla/pjrt/distributed:client",
        "//tensorflow/compiler/xla/pjrt/distributed:service",
        "//tensorflow/core:lib",
        # Do NOT remove this dependency. The XLA Python extension must not
        # depend on any part of TensorFlow at runtime, **including**
        # libtensorflow_framework.so. The XLA module is deployed self-contained
        # without any TF dependencies as "jaxlib" on Pypi, and "jaxlib" does
        # not require Tensorflow.
        "//tensorflow/core:lib_internal_impl",  # buildcleaner: keep
        "//tensorflow/python:bfloat16_lib",
    ] + select({
        ":enable_gpu": ["//tensorflow/compiler/xla/service:gpu_plugin"],
        "//tensorflow:with_tpu_support": [
            "//tensorflow/core/tpu:tpu_executor_dlsym_initializer",
        ],
        "//conditions:default": [],
    }),
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    linkopts = [
        "-L/usr/local/nccl/lib",
        "-lnccl",
    ],

)

Final status: jax0.2.21 with cuda on macOS has been successfully compiled