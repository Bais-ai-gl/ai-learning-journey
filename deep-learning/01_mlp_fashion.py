# Day 6：深度学习 · PyTorch MLP 训练 Fashion-MNIST
# 用法：py -m pip install torch torchvision   然后  py 01_mlp_fashion.py
# 目标：测试准确率 ≥ 0.88（首次运行会自动下载数据集约 30MB）

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH = 64
EPOCHS = 10        # 3 轮太少，模型还没收敛；提到 10 轮更容易超过 88%
LR = 1e-3

# 1. 数据：Fashion-MNIST（10 类衣服，28×28 灰度图）
# 关键改进：归一化到 [-1, 1]，让输入均值≈0、范围稳定，梯度更新更均匀、收敛更快
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),   # 灰度图只有 1 个通道：(均值, 标准差)
])
train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH)

# 2. 模型：3 层全连接（MLP）
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)   # 加宽：更多参数 → 拟合能力更强
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)     # 输出 10 类
        self.dropout = nn.Dropout(0.2)    # 训练时随机丢弃 20% 神经元，防止过拟合

    def forward(self, x):
        x = x.view(x.size(0), -1)         # 把 28×28 展平成 784
        x = F.relu(self.fc1(x))
        x = self.dropout(x)               # 每个全连接层后加 dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)                # 最后一层不做 softmax（CrossEntropyLoss 内部处理）

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
# 学习率调度：每 step_size 轮把学习率乘 gamma（1e-3→1e-4→1e-5…），后期步长变小、走得更稳
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 3. 评估函数
def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# 4. 训练循环（这是核心，务必逐行理解）
for epoch in range(1, EPOCHS + 1):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()          # 清空上一步梯度
        loss = loss_fn(model(x), y)    # 前向 + 算损失
        loss.backward()                # 反向传播求梯度
        optimizer.step()               # 更新参数
    acc = evaluate()
    print(f"Epoch {epoch}/{EPOCHS}  测试准确率 {acc:.3f}  当前学习率 {scheduler.get_last_lr()[0]:.1e}")
    scheduler.step()                  # 每个 epoch 结束后按调度器更新学习率

print("目标：≥ 0.88")
# 思考：optimizer.zero_grad() 为什么每次都要调用？去掉会怎样？
