import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

train = pd.read_csv('data/train_data.csv')
test = pd.read_csv('data/test_data.csv')

# 自行对压力等级分类
def map_stress_level(x):
    if x <= 3.33:
        return 0  # 低压力 (Low)
    elif x <= 6.66:
        return 1  # 中压力 (Medium)
    else:
        return 2  # 高压力 (High)

train['stress_level'] = train['stress_level'].apply(map_stress_level)
test['stress_level'] = test['stress_level'].apply(map_stress_level)

X_train = torch.tensor(train.drop(columns=['mental_fatigue_score', 'stress_level']).values, dtype=torch.float32)
X_test = torch.tensor(test.drop(columns=['mental_fatigue_score', 'stress_level']).values, dtype=torch.float32)

y_train = torch.tensor(train['stress_level'].values, dtype=torch.long)
y_test = torch.tensor(test['stress_level'].values, dtype=torch.long)


# 定义模型

input_dim = X_train.shape[1]
num_classes = 3  # 对应 0, 1, 2
net = nn.Sequential(
    nn.Linear(input_dim, num_classes)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)


epochs = 200
train_losses = []

print("开始训练")
for epoch in range(epochs):
    outputs = net(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(net, 'models/softmax_model.pth')


# 7. 测试集验证准确率 (Accuracy)
net.eval()
with torch.no_grad():
    outputs = net(X_test)
    _, predicted = torch.max(outputs, 1)

    correct = (predicted == y_test).sum().item()
    accuracy = correct / y_test.size(0)
    print(f'\n分类任务准确率: {accuracy * 100:.2f}%')

# 8. 可视化
plt.plot(train_losses)
plt.title('Softmax Regression Loss')
plt.savefig('images/Softmax Loss.png')
plt.show()