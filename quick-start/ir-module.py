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
mod_from_torch.show()

# =================================== 利用Relax提供的api接入 =======================================
from tvm.relax.frontend import nn

class RelaxModel(nn.Module):
    def __init__(self):
        super(RelaxModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


mod_from_relax, params_from_relax = RelaxModel().export_tvm(
    {"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
mod_from_relax.show()

# =================================== 利用TVM script构建IRModule =======================================
from tvm.script import ir as I
from tvm.script import relax as R

@I.ir_module
class TVMScriptModule:
    @R.function
    def main(
        x: R.Tensor((1, 784), dtype="float32"),
        fc1_weight: R.Tensor((256, 784), dtype="float32"),
        fc1_bias: R.Tensor((256,), dtype="float32"),
        fc2_weight: R.Tensor((10, 256), dtype="float32"),
        fc2_bias: R.Tensor((10,), dtype="float32"),
    ) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims = R.permute_dims(fc1_weight, axes=None)
            matmul = R.matmul(x, permute_dims, out_dtype="void")
            add = R.add(matmul, fc1_bias)
            relu = R.nn.relu(add)
            permute_dims1 = R.permute_dims(fc2_weight, axes=None)
            matmul1 = R.matmul(relu, permute_dims1, out_dtype="void")
            add1 = R.add(matmul1, fc2_bias)
            gv = add1
            R.output(gv)
        return gv

mod_from_script = TVMScriptModule
mod_from_script.show()