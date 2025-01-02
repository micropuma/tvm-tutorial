import tvm
from tvm import relax
from tvm.relax.frontend import nn
import numpy as np

# 用 relax计算图表达式构建model
# 类似pytorch的写法
class MLPModule(nn.Module):
    def __init__(self):
        # 继承自nn.module，所以要先构造nn.module
        super(MLPModule, self).__init__()
        self.fc1 = nn.Linear(784,256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256,10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# export computation graph model
mod, param_spec = MLPModule().export_tvm(
    spec = {"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)

# 这个mod是IRModule
print("Model type:", type(mod))
mod.show()

# leverage zero optimization pipeline, instead of optimizing for any specific target.
mod = relax.get_pipeline("zero")(mod)
    
# 部署到硬件上
# 得到end-to-end目标是llvm
# 设定device是cpu设备
target = tvm.target.Target("llvm")
ex = relax.build(mod, target)
device = tvm.cpu()
vm = relax.VirtualMachine(ex, device)

data = np.random.rand(1, 784).astype("float32")
tvm_data = tvm.nd.array(data, device=device)
# 遍历param_spec里面的参数规格，生成随机param
params = [np.random.rand(*param.shape).astype("float32") for _, param in param_spec]
# 和输入的数据一样，用numpy库生成数据，需要用tvm.nd.array()转换数据为tvm格式
params = [tvm.nd.array(param, device=device) for param in params]
print(vm["forward"](tvm_data, *params).numpy())
