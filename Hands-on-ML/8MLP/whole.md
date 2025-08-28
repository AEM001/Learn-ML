1. 常用激活函数映射（torch_activation_dict）

import torch
import torch.nn as nn
from torch.nn.init import normal_

torch_activation_dict = {
    'identity': lambda x: x,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'relu': torch.relu
}

用途：让激活函数的选择变成字符串参数，快速切换。
用法示例：

act_fn = torch_activation_dict['relu']


⸻

2. MLP 模板类（可直接复用）

class MLP_torch(nn.Module):
    def __init__(self, layer_sizes, use_bias=True,
                 activation='relu', out_activation='identity'):
        super().__init__()
        self.activation = torch_activation_dict[activation]
        self.out_activation = torch_activation_dict[out_activation]
        self.layers = nn.ModuleList()
        num_in = layer_sizes[0]
        for num_out in layer_sizes[1:]:
            layer = nn.Linear(num_in, num_out, bias=use_bias)
            normal_(layer.weight, std=1.0)       # 正态初始化
            layer.bias.data.fill_(0.0)           # 偏置初始化
            self.layers.append(layer)
            num_in = num_out

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.out_activation(self.layers[-1](x))
        return x

用途：快速搭建多层感知机，不必每次手写 forward。
用法示例：

mlp = MLP_torch(layer_sizes=[2, 4, 1], out_activation='sigmoid')


⸻

3. 通用训练循环模板（适用于小批量 SGD）

def train_model(model, x_train, y_train, x_test, y_test,
                num_epochs=1000, batch_size=128, lr=0.1, eps=1e-7):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses, test_losses, test_accs = [], [], []

    for epoch in range(num_epochs):
        st = 0
        batch_losses = []
        while st < len(x_train):
            ed = min(st + batch_size, len(x_train))
            x = torch.tensor(x_train[st: ed], dtype=torch.float32)
            y = torch.tensor(y_train[st: ed], dtype=torch.float32).reshape(-1, 1)

            y_pred = model(x)
            train_loss = torch.mean(-y * torch.log(y_pred + eps) -
                                    (1 - y) * torch.log(1 - y_pred + eps))

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            batch_losses.append(train_loss.item())
            st += batch_size

        losses.append(sum(batch_losses) / len(batch_losses))

        with torch.inference_mode():
            x = torch.tensor(x_test, dtype=torch.float32)
            y = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
            y_pred = model(x)
            test_loss = torch.mean(-y * torch.log(y_pred + eps) -
                                   (1 - y) * torch.log(1 - y_pred + eps))
            test_acc = (torch.round(y_pred) == y).float().mean()
            test_losses.append(test_loss.item())
            test_accs.append(test_acc.item())

    return losses, test_losses, test_accs

用途：输入任何 MLP 模型和数据，直接训练得到 losses、test_losses、test_accs。
用法示例：

losses, test_losses, test_accs = train_model(mlp, x_train, y_train, x_test, y_test)


⸻

4. 可视化模板

import matplotlib.pyplot as plt

def plot_training_results(losses, test_losses, test_accs):
    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plt.plot(losses, color='blue', label='train loss')
    plt.plot(test_losses, color='red', ls='--', label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Cross-Entropy Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(test_accs, color='red')
    plt.ylim(top=1.0)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.show()

用法示例：

plot_training_results(losses, test_losses, test_accs)


⸻

最终组合示例

# 1. 定义模型
mlp = MLP_torch(layer_sizes=[2, 4, 1], out_activation='sigmoid')

# 2. 训练
losses, test_losses, test_accs = train_model(mlp, x_train, y_train, x_test, y_test)

# 3. 可视化
plot_training_results(losses, test_losses, test_accs)

这样，你以后做分类任务时，基本只需要改 layer_sizes、数据集 和 超参数，就可以直接运行了。

⸻

我可以帮你额外做一个“极简速记版”，让你在脑子里记 10 行就能复现全流程，你要吗？这样以后敲出来比查笔记还快。