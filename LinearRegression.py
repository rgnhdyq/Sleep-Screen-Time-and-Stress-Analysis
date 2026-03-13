import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

train = pd.read_csv('data/train_data.csv')
test = pd.read_csv('data/test_data.csv')

X_train = torch.tensor(train.drop(columns=['mental_fatigue_score', 'stress_level']).values, dtype=torch.float32)
X_test = torch.tensor(test.drop(columns=['mental_fatigue_score', 'stress_level']).values, dtype=torch.float32)

# 定义线性回归标签 y (连续值)
y_train = torch.tensor(train['mental_fatigue_score'].values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(test['mental_fatigue_score'].values, dtype=torch.float32).view(-1, 1)


# 定义模型
input_dim = X_train.shape[1]
net = nn.Sequential(
    nn.Linear(input_dim, 1)
)

criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(net.parameters(), lr=0.01)  # 随机梯度下降

epochs = 100
train_losses = []

print("开始训练")
for epoch in range(epochs):
    # 前向传播
    y_hat = net(X_train)
    loss = criterion(y_hat, y_train)

    # 反向传播与优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新权重

    train_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(net, 'models/linear_model.pth')


# 测试集验证

net.eval()
with torch.no_grad():
    y_test_pred = net(X_test)
    test_loss = criterion(y_test_pred, y_test)
    print(f'\n测试集 MSE Loss: {test_loss.item():.4f}')


# 结果展示

plt.plot(train_losses, label='Train Loss')
plt.title('Linear Regression Training Progress')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig('images/Linear Loss.png')
plt.show()
