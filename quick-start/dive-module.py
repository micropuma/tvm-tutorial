import numpy as np  
import tvm
from tvm import relax

# =================================== 从pytorch接入 =======================================
# 这个示例演示从pytorch接入TVM的流程
import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

# Create a dummy model
class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Give an example argument to torch.export
example_args = (torch.randn(1, 784, dtype=torch.float32),)

# Convert the model to IRModule
with torch.no_grad():
    exported_program = export(TorchModel().eval(), example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True
    )

mod_from_torch, params_from_torch = relax.frontend.detach_params(mod_from_torch)
# Print the IRModule
print("====================================================================")
mod_from_torch.show()

# =================================== 解析IRModule =======================================
mod = mod_from_torch
print(mod.get_global_vars())

# 检查使用名字索引global var和使用mod.get_global_vars得到的function是同一个
# index by global var name
print("====================================================================")
print(mod["main"])
# index by global var, and checking they are the same function
# 用(gv,)来解包
(gv,) = mod.get_global_vars()
assert mod[gv] == mod["main"]

# =================================== IRModule的transformation =======================================
# We first apply LegalizeOps transformation to the IRModule. 
# This transformation will convert the Relax module into a mixed stage, 
# with both Relax and TensorIR function within the same module. Meanwhile, 
# the Relax operators will be converted into call_tir.
# dump出来的ir得出，这个LegailizeOps重点作用是把Relax.relu类似的算子变成legalize的定义形式
mod = relax.transform.LegalizeOps()(mod)
print("====================================================================")
mod.show()
print(mod.get_global_vars())

print("====================================================================")
# zero pipeline是最简单的pipeline，包含如下四个pass：
# LegalizeOps: This transform converts the Relax operators into call_tir functions with the corresponding TensorIR Functions. After this transform, the IRModule will contain both Relax functions and TensorIR functions.
# AnnotateTIROpPattern: This transform annotates the pattern of the TensorIR functions, preparing them for subsequent operator fusion.
# FoldConstant: This pass performs constant folding, optimizing operations involving constants.
# FuseOps and FuseTIR: These two passes work together to fuse operators based on the patterns annotated in the previous step (AnnotateTIROpPattern). These passes transform both Relax functions and TensorIR functions.
mod = mod_from_torch 
mod = relax.get_pipeline("zero")(mod)
mod.show()

# =================================== Universally deploy =======================================
# 部署在cpu上
exec = relax.build(mod, target="llvm")
dev = tvm.cpu()
vm = relax.VirtualMachine(exec, dev)

raw_data = np.random.rand(1, 784).astype("float32")
data = tvm.nd.array(raw_data, dev)
cpu_out = vm["main"](data, *params_from_torch["main"]).numpy()
print(cpu_out)

# 尝试部署在gpu上，使用DLIGHT来部署。
from tvm import dlight as dl

# 用dlight来生成gpu代码
with tvm.target.Target("cuda"):
    gpu_mod = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.Fallback(),
    )(mod)

# 和cpu一样的部署方式，比对两者的结果
exec = relax.build(gpu_mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(exec, dev)
# Need to allocate data and params on GPU device
data = tvm.nd.array(raw_data, dev)
gpu_params = [tvm.nd.array(p, dev) for p in params_from_torch["main"]]
gpu_out = vm["main"](data, *gpu_params).numpy()
print(gpu_out)

# Check the correctness of the results
assert np.allclose(cpu_out, gpu_out, atol=1e-3)