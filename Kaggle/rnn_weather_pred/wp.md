import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('/Users/Mac/code/Practice/ML/Kaggle/rnn_weather_pred/seattle-weather.csv')

# 将 'date' 列转换为 datetime 类型，方便后续处理
df['date'] = pd.to_datetime(df['date'])

def month_to_season(month):
    """将月份转换为季节：1=冬, 2=春, 3=夏, 4=秋"""
    if month in [12, 1, 2]:
        return 1  # Winter
    elif month in [3, 4, 5]:
        return 2  # Spring
    elif month in [6, 7, 8]:
        return 3  # Summer
    else:  # 9, 10, 11
        return 4  # Autumn

# 生成编码后的季节变量
df['season'] = df['date'].dt.month.apply(month_to_season)

# 天气编码：保留数值编码，同时保存编码与标签的对应关系
df['weather_code'], weather_labels = pd.factorize(df['weather'])

# 在内存中删除不需要的原始列：date 和 weather
df.drop(columns=['date', 'weather'], inplace=True)

# 将 weather_code 调整到最后面，season 放前面
front_cols = ['season']
other_cols = [c for c in df.columns if c not in front_cols + ['weather_code']]
df = df[front_cols + other_cols + ['weather_code']]

# 输出检查
print("天气类型编码（索引=编码，对应标签如下）:")
print(list(enumerate(weather_labels)))
print(df.head())

# 将处理好的数据保存到新的CSV文件中（不保存索引和表头）
df.to_csv('/Users/Mac/code/Practice/ML/Kaggle/rnn_weather_pred/seattle-weather-processed.csv', index=False, header=False)
print("数据已保存到 seattle-weather-processed.csv")

import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('/Users/Mac/code/Practice/ML/Kaggle/rnn_weather_pred/seattle-weather.csv')

# 将 'date' 列转换为 datetime 类型，方便后续处理
df['date'] = pd.to_datetime(df['date'])

def month_to_season(month):
    """将月份转换为季节：1=冬, 2=春, 3=夏, 4=秋"""
    if month in [12, 1, 2]:
        return 1  # Winter
    elif month in [3, 4, 5]:
        return 2  # Spring
    elif month in [6, 7, 8]:
        return 3  # Summer
    else:  # 9, 10, 11
        return 4  # Autumn

# 生成编码后的季节变量
df['season'] = df['date'].dt.month.apply(month_to_season)

# 天气编码：保留数值编码，同时保存编码与标签的对应关系
df['weather_code'], weather_labels = pd.factorize(df['weather'])

# 在内存中删除不需要的原始列：date 和 weather
df.drop(columns=['date', 'weather'], inplace=True)

# 将 weather_code 调整到最后面，season 放前面
front_cols = ['season']
other_cols = [c for c in df.columns if c not in front_cols + ['weather_code']]
df = df[front_cols + other_cols + ['weather_code']]

# 输出检查
print("天气类型编码（索引=编码，对应标签如下）:")
print(list(enumerate(weather_labels)))
print(df.head())

# 将处理好的数据保存到新的CSV文件中（不保存索引和表头）
df.to_csv('/Users/Mac/code/Practice/ML/Kaggle/rnn_weather_pred/seattle-weather-processed.csv', index=False, header=False)
print("数据已保存到 seattle-weather-processed.csv")


import torch.nn as nn
input_size=5
output_size=1
learning_rate=1e-3
hidden_1 = 16
hidden_2 = 16
max_epoch=500
num_classes=5
# mlp = nn.Sequential(
#     nn.Linear(input_size, hidden_1),
#     nn.ReLU(),
#     nn.Linear(hidden_1, hidden_2),
#     nn.ReLU(),
#     nn.Linear(hidden_2, output_size)
# )
# mlp_optim = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
# for x,y in zip(x_train,y_train):

#定义为Encoder和decoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_1=32, x_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, x_dim),
            nn.ReLU(),  # 也可不加
        )
    def forward(self, x):
        return self.net(x)  # x_t

class Classifier(nn.Module):
    def __init__(self, x_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(x_dim, num_classes)
    def forward(self, x_t):
        return self.fc(x_t)  # logits

encoder = Encoder(input_size, hidden_1=32, x_dim=16)
clf = Classifier(16, num_classes)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(clf.parameters()), lr=learning_rate)
acc = []
criterion = nn.CrossEntropyLoss()  # Assuming classification task

for epoch in range(max_epoch):
    # 训练阶段
    encoder.train()
    clf.train()
    
    x_t = encoder(x_train)              # 编码得到潜在表示
    logits = clf(x_t)
    # 确保 y_train 是 1D 张量
    loss = criterion(logits, y_train.squeeze()) 
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 评估阶段
    with torch.inference_mode():
        encoder.eval()
        clf.eval()
        y_pred = clf(encoder(x_test))
        # 计算准确率并添加到acc列表
        _, predicted = torch.max(y_pred.data, 1)
        accuracy = (predicted == y_test.squeeze()).sum().item() / len(y_test)
        acc.append(accuracy)
    
    print(f'Epoch: {epoch+1}/{max_epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

---
……
Epoch: 499/500, Loss: 0.4565, Accuracy: 0.8635
Epoch: 500/500, Loss: 0.4563, Accuracy: 0.8635

---
# --- 生成全部数据的 latent x ---
encoder.eval()
with torch.inference_mode():
    # 用训练好的编码器处理全部数据（按时间顺序）
    all_data = np.vstack([x_train.numpy(), x_test.numpy()])  # 拼接训练和测试数据
    all_data_tensor = torch.from_numpy(all_data).float()
    x_all = encoder(all_data_tensor)  # shape: (N, 16)

print(f"生成的 x_all 形状: {x_all.shape}")

# --- 构造滑动窗口数据 ---
T, H = 30, 7  # 过去30天 -> 未来7天
N, d = x_all.shape

def make_seq_pairs(x_data, start_idx, end_idx, T, H):
    """构造滑动窗口：(过去T天的x) -> (未来H天的x)"""
    Xs, Ys = [], []
    for t in range(start_idx + T - 1, end_idx - H):
        X = x_data[t - T + 1 : t + 1]     # (T, d)
        Y = x_data[t + 1 : t + 1 + H]     # (H, d)
        Xs.append(X.unsqueeze(0))
        Ys.append(Y.unsqueeze(0))
    if not Xs: 
        return None, None
    return torch.cat(Xs, 0), torch.cat(Ys, 0)  # (num_samples, T, d), (num_samples, H, d)

# 按照原来的训练/测试划分构造序列
split = int(0.8 * len(all_data))
Xtr, Ytr = make_seq_pairs(x_all, 0, split, T, H)
Xte, Yte = make_seq_pairs(x_all, split, N, T, H)

print(f"GRU 训练数据: {Xtr.shape if Xtr is not None else 'None'}")
print(f"GRU 测试数据: {Xte.shape if Xte is not None else 'None'}")

---
from torch.utils.data import Dataset, DataLoader

# --- GRU 模型定义 ---
class GRUForecaster(nn.Module):
    def __init__(self, d=16, hidden=64, H=7):
        super().__init__()
        self.gru = nn.GRU(input_size=d, hidden_size=hidden, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden, H * d)
        self.H, self.d = H, d
    
    def forward(self, x):  # x: (batch, T, d)
        _, h_n = self.gru(x)         # h_n: (1, batch, hidden)
        h_last = h_n.squeeze(0)      # (batch, hidden)
        y = self.proj(h_last)        # (batch, H*d)
        return y.view(-1, self.H, self.d)  # (batch, H, d)

# --- 数据加载器 ---
class XYDataset(Dataset):
    def __init__(self, X, Y): 
        self.X, self.Y = X, Y
    def __len__(self): 
        return len(self.X)
    def __getitem__(self, i): 
        return self.X[i], self.Y[i]

train_loader = DataLoader(XYDataset(Xtr, Ytr), batch_size=32, shuffle=True)
test_loader = DataLoader(XYDataset(Xte, Yte), batch_size=64, shuffle=False)

# --- 训练 GRU ---
model_gru = GRUForecaster(d=16, hidden=64, H=H)
optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=1e-3)
criterion_gru = nn.MSELoss()

print("--- 开始训练 GRU 预测器 ---")
max_epoch_gru = 1000

for epoch in range(max_epoch_gru):
    model_gru.train()
    train_loss = 0
    
    for xb, yb in train_loader:
        y_hat = model_gru(xb)
        loss = criterion_gru(y_hat, yb)
        
        optimizer_gru.zero_grad()
        loss.backward()
        optimizer_gru.step()
        
        train_loss += loss.item() * len(xb)
    
    # 每5个epoch打印一次
    if (epoch + 1) % 25 == 0:
        avg_train_loss = train_loss / len(Xtr)
        print(f"Epoch [{epoch+1}/{max_epoch_gru}], Train MSE: {avg_train_loss:.6f}")

print("--- GRU 预测器训练完成 ---")

---
Epoch [600/1000], Train MSE: 10.865105
...
Epoch [950/1000], Train MSE: 8.219531
Epoch [975/1000], Train MSE: 21.375678
Epoch [1000/1000], Train MSE: 7.816187

---
# --- 最终评估：用预测的 x̂ 得到未来天气 ---
model_gru.eval()
clf.eval()

print("--- 开始最终评估 ---")

all_preds, all_true = [], []
test_mse = 0

with torch.no_grad():
    for xb, yb in test_loader:
        # GRU 预测未来的 x_hat
        x_hat = model_gru(xb)  # (batch, H, d)
        
        # 计算 MSE（在 latent space）
        test_mse += criterion_gru(x_hat, yb).item() * len(xb)
        
        # 将预测的 x_hat 转换为天气预测
        batch_size = x_hat.size(0)
        x_hat_flat = x_hat.view(batch_size * H, -1)  # (batch*H, d)
        weather_logits = clf(x_hat_flat)  # (batch*H, num_classes)
        weather_preds = weather_logits.argmax(dim=1)  # (batch*H,)
        
        # 对于真实标签，我们需要从测试集中获取对应的 weather_code
        # 这里简化处理：用真实的 yb 通过分类头得到"真实"标签
        yb_flat = yb.view(batch_size * H, -1)  # (batch*H, d)
        true_logits = clf(yb_flat)
        true_labels = true_logits.argmax(dim=1)  # (batch*H,)
        
        all_preds.append(weather_preds)
        all_true.append(true_labels)

# 计算指标
all_preds = torch.cat(all_preds)
all_true = torch.cat(all_true)
avg_test_mse = test_mse / len(Xte)
weather_accuracy = (all_preds == all_true).float().mean().item()

print(f"测试集 MSE (latent space): {avg_test_mse:.6f}")
print(f"未来 {H} 天天气预测平均准确率: {weather_accuracy:.4f}")

# 计算每个预测步的准确率
print("\n各预测步的准确率:")
for h in range(H):
    step_preds = all_preds[h::H]
    step_true = all_true[h::H]
    step_acc = (step_preds == step_true).float().mean().item()
    print(f"  未来第 {h+1} 天: {step_acc:.4f}")

---
# --- 最终评估：用预测的 x̂ 得到未来天气 ---
model_gru.eval()
clf.eval()

print("--- 开始最终评估 ---")

all_preds, all_true = [], []
test_mse = 0

with torch.no_grad():
    for xb, yb in test_loader:
        # GRU 预测未来的 x_hat
        x_hat = model_gru(xb)  # (batch, H, d)
        
        # 计算 MSE（在 latent space）
        test_mse += criterion_gru(x_hat, yb).item() * len(xb)
        
        # 将预测的 x_hat 转换为天气预测
        batch_size = x_hat.size(0)
        x_hat_flat = x_hat.view(batch_size * H, -1)  # (batch*H, d)
        weather_logits = clf(x_hat_flat)  # (batch*H, num_classes)
        weather_preds = weather_logits.argmax(dim=1)  # (batch*H,)
        
        # 对于真实标签，我们需要从测试集中获取对应的 weather_code
        # 这里简化处理：用真实的 yb 通过分类头得到"真实"标签
        yb_flat = yb.view(batch_size * H, -1)  # (batch*H, d)
        true_logits = clf(yb_flat)
        true_labels = true_logits.argmax(dim=1)  # (batch*H,)
        
        all_preds.append(weather_preds)
        all_true.append(true_labels)

# 计算指标
all_preds = torch.cat(all_preds)
all_true = torch.cat(all_true)
avg_test_mse = test_mse / len(Xte)
weather_accuracy = (all_preds == all_true).float().mean().item()

print(f"测试集 MSE (latent space): {avg_test_mse:.6f}")
print(f"未来 {H} 天天气预测平均准确率: {weather_accuracy:.4f}")

# 计算每个预测步的准确率
print("\n各预测步的准确率:")
for h in range(H):
    step_preds = all_preds[h::H]
    step_true = all_true[h::H]
    step_acc = (step_preds == step_true).float().mean().item()
    print(f"  未来第 {h+1} 天: {step_acc:.4f}")