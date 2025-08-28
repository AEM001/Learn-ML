import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List


class MLP(nn.Module):
    """
    多层感知机 (MLP) 标准模板
    
    参数:
    - input_dim: 输入特征维度
    - hidden_dims: 隐藏层节点数列表，例如 [100, 50, 25]
    - output_dim: 输出维度
    - activation: 激活函数，'relu', 'sigmoid', 'tanh' 等
    - dropout_rate: Dropout 概率 (可选)
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 activation: str = 'relu', dropout_rate: float = 0.0):
        super(MLP, self).__init__()
        
        self.activation = activation.lower()
        self.dropout_rate = dropout_rate
        
        # 构建网络层
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # 添加激活函数和dropout
        if self.activation == 'relu':
            layers.append(nn.ReLU())
        elif self.activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif self.activation == 'tanh':
            layers.append(nn.Tanh())
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def create_data_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    """
    创建数据加载器
    
    参数:
    - X: 特征张量
    - y: 标签张量
    - batch_size: 批次大小
    - shuffle: 是否打乱数据
    """
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model: nn.Module, train_loader: DataLoader, criterion, optimizer, 
                num_epochs: int = 100, device: str = 'cpu'):
    """
    训练模型
    
    参数:
    - model: 模型
    - train_loader: 训练数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练轮数
    - device: 设备 ('cpu' 或 'cuda')
    
    返回:
    - 训练损失历史
    """
    model.to(device)
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_samples = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            total_samples += batch_X.size(0)
        
        avg_loss = epoch_loss / total_samples
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return train_losses


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu'):
    """
    评估模型
    
    参数:
    - model: 模型
    - test_loader: 测试数据加载器
    - device: 设备
    
    返回:
    - 预测结果和真实标签
    """
    model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)


# 使用示例
if __name__ == '__main__':
    # 示例配置
    config = {
        'input_dim': 8,           # 输入特征维度
        'hidden_dims': [100, 50, 25],  # 隐藏层节点数
        'output_dim': 1,         # 输出维度
        'activation': 'relu',    # 激活函数
        'dropout_rate': 0.0,     # Dropout 概率
        'batch_size': 64,        # 批次大小
        'learning_rate': 0.01,   # 学习率
        'num_epochs': 30,        # 训练轮数
    }
    
    # 创建模型
    model = MLP(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        output_dim=config['output_dim'],
        activation=config['activation'],
        dropout_rate=config['dropout_rate']
    )
    
    print(f"模型结构:\n{model}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 这里可以添加你的数据加载代码
    # 例如:
    # X_train = torch.randn(1000, 8)  # 1000个样本，8个特征
    # y_train = torch.randn(1000, 1)   # 回归任务
    # train_loader = create_data_loader(X_train, y_train, config['batch_size'])
    # 
    # X_test = torch.randn(200, 8)     # 200个测试样本
    # y_test = torch.randn(200, 1)
    # test_loader = create_data_loader(X_test, y_test, config['batch_size'], shuffle=False)
    # 
    # # 定义损失函数和优化器
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    # 
    # # 训练模型
    # train_losses = train_model(model, train_loader, criterion, optimizer, config['num_epochs'])
    # 
    # # 可视化训练损失
    # import matplotlib.pyplot as plt
    # plt.plot(train_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()
    # 
    # # 测试模型
    # predictions, true_labels = evaluate_model(model, test_loader)
    # from sklearn.metrics import mean_absolute_error
    # mae = mean_absolute_error(true_labels, predictions)
    # print(f'MAE: {mae:.4f}')
