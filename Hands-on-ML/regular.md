我来为你详细介绍深度学习中的这些常用操作，通过独立的代码片段来展示：

## 1. 定义优化器 (Optimizer)

```python
import torch.optim as optim

# SGD优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam优化器（最常用）
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW优化器（带权重衰减）
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## 2. 定义损失函数 (Loss Function)

```python
import torch.nn as nn

# 分类任务 - 交叉熵损失
criterion = nn.CrossEntropyLoss()

# 回归任务 - 均方误差
criterion = nn.MSELoss()

# 二分类 - 二元交叉熵
criterion = nn.BCEWithLogitsLoss()
```

## 3. 反向传播 (Backward)

```python
# 计算损失
loss = criterion(outputs, targets)

# 清零梯度（重要！）
optimizer.zero_grad()

# 反向传播计算梯度
loss.backward()
```

## 4. 参数更新 (Step)

```python
# 更新模型参数
optimizer.step()

# 完整的一步训练流程
optimizer.zero_grad()  # 清零梯度
loss = criterion(outputs, targets)  # 计算损失
loss.backward()  # 反向传播
optimizer.step()  # 更新参数
```

## 5. 训练模式 (Train Mode)

```python
# 设置模型为训练模式
model.train()

# 训练模式的特点：
# - 启用Dropout
# - 启用BatchNorm的参数更新
# - 梯度计算开启
```

## 6. 评估模式 (Eval Mode)

```python
# 设置模型为评估模式
model.eval()

# 通常配合no_grad使用
with torch.no_grad():
    outputs = model(inputs)
    
# 评估模式的特点：
# - 关闭Dropout
# - 使用BatchNorm的固定统计量
# - 通常关闭梯度计算以节省内存
```

## 7. 梯度清零操作

```python
# 方式1：使用优化器清零
optimizer.zero_grad()

# 方式2：手动清零模型参数梯度
for param in model.parameters():
    param.grad = None
```

## 8. 学习率调度器 (Scheduler)

```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# 固定步长降低学习率
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 根据指标自适应调整
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

# 使用方式
scheduler.step()  # 在每个epoch后调用
```

## 9. 设备管理 (Device)

```python
# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型移到指定设备
model = model.to(device)

# 将数据移到指定设备
inputs = inputs.to(device)
targets = targets.to(device)
```

## 10. 保存和加载模型

```python
# 保存模型状态字典
torch.save(model.state_dict(), 'model.pth')

# 加载模型状态字典
model.load_state_dict(torch.load('model.pth'))

# 保存完整的检查点
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')
```

## 11. 梯度裁剪 (Gradient Clipping)

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 通常在backward()和step()之间使用
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## 12. 冻结和解冻参数

```python
# 冻结模型参数（不更新梯度）
for param in model.parameters():
    param.requires_grad = False

# 解冻特定层
for param in model.classifier.parameters():
    param.requires_grad = True
```

## 13. 随机种子设置

```python
import random
import numpy as np

# 设置随机种子确保结果可复现
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

这些都是深度学习中最常用的基础操作，在实际项目中你会经常用到它们。每个操作都有其特定的用途和调用时机，掌握好这些基础操作对于深度学习实践非常重要。