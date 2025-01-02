import os
import tempfile
import numpy as np
import tvm
from tvm import IRModule, relax
from tvm.relax.frontend import nn

print("==========================================================================")

# 准备一个Relax Module
class RelaxModel(nn.Module):
    def __init__(self):
        super(RelaxModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


input_shape = (1, 784)
mod_original, params = RelaxModel().export_tvm({"forward": {"x": nn.spec.Tensor(input_shape, "float32")}})
mod_original.show()

print("==========================================================================")
# 这个是计算图层面的优化，具体看论文
# dispatch CUBLAS library
# Import cublas pattern
import tvm.relax.backend.contrib.cublas as _cublas

mod = mod_original

# TVM的operation 优化和mlir一样，定义pattern，match的pattern就用对应的
# CUBLAS或是CUDNN库来优化即可。
# Define a new pass for CUBLAS dispatch
@tvm.transform.module_pass(opt_level=0, name="CublasDispatch")
class CublasDispatch:
    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        # Check if CUBLAS is enabled
        if not tvm.get_global_func("relax.ext.cublas", True):
            raise Exception("CUBLAS is not enabled.")

        # Get interested patterns
        # 这里我们对于cublas的matmul_transposed_bias_relu模式感兴趣
        patterns = [relax.backend.get_pattern("cublas.matmul_transposed_bias_relu")]
        # Note in real-world cases, we usually get all patterns
        # patterns = relax.backend.get_patterns_with_prefix("cublas")

        # Fuse ops by patterns and then run codegen
        # 凡是匹配我们的pattern的，都会由最终的Cublas库来代替
        mod = relax.transform.FuseOpsByPattern(patterns, annotate_codegen=True)(mod)
        mod = relax.transform.RunCodegen()(mod)
        return mod


mod = CublasDispatch()(mod)
mod.show()

print("==========================================================================")
# 这个是TensorIR层面的auto tuning，具体参见TVM论文
mod = mod_original

# auto tuning
# 两步骤：
# 1. convert RELAX to Tensor IR
# 2. auto tuning
device = tvm.cuda(0)
target = tvm.target.Target.from_device(device)
if os.getenv("CI", "") != "true":
    trials = 100
    with target, tempfile.TemporaryDirectory() as tmp_dir:
        mod = tvm.ir.transform.Sequential(
            [
                relax.get_pipeline("zero"),
                relax.transform.MetaScheduleTuneTIR(work_dir=tmp_dir, max_trials_global=trials),
                relax.transform.MetaScheduleApplyDatabase(work_dir=tmp_dir),
            ]
        )(mod)

    mod.show()

print("==========================================================================")

mod = mod_original

# DLight Rules
# 对于特定的backend，提供前沿优化知识

from tvm import dlight as dl

# Apply DLight rules
with target:
    mod = tvm.ir.transform.Sequential(
        [
            relax.get_pipeline("zero"),
            dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            ),
        ]
    )(mod)

mod.show()

print("==========================================================================")

# deploy model on CUDA 
ex = relax.build(mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(ex, dev)
# Need to allocate data and params on GPU device
data = tvm.nd.array(np.random.rand(*input_shape).astype("float32"), dev)
gpu_params = [tvm.nd.array(np.random.rand(*p.shape).astype(p.dtype), dev) for _, p in params]
gpu_out = vm["forward"](data, *gpu_params).numpy()
print(gpu_out)

