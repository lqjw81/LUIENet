# 创建一个简单CNN模型实例
from archs.FasterUIE import FIVE_APLUSNet
model = FIVE_APLUSNet()

# 计算模型的参数数量
total_params = sum(p.numel() for p in model.parameters())
print("总参数数量：", total_params)