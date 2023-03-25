import sqlite3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import get_autoencoder1, evaluate_and_get_normal_component, save_variable, load_variable
from AE2_extract import get_autoencoder2, extract_all_month
from SLF_train_test import CNN, DatasetForSLF


# 读取卷积神经网络模型
def get_cnn(path, device):
    _cnn = CNN().to(device)
    # 加载参数
    params = torch.load(path, map_location=torch.device(device))
    # 应用到网络结构中
    _cnn.load_state_dict(params)
    _cnn.eval()
    return _cnn


# stacking集成学习模型
class stacking_CNN(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        # N = (W − F + 2P )/S+1
        self.model_num = 4
        self.device = device
        self.cnn = [get_cnn(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\2.基于集成学习的空间负荷预测\CNN_20230212_141922.path", device),
                    get_cnn(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\2.基于集成学习的空间负荷预测\CNN_20230212_143623.path", device),
                    get_cnn(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\2.基于集成学习的空间负荷预测\CNN_20230212_175318.path", device),
                    get_cnn(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\2.基于集成学习的空间负荷预测\CNN_20230212_190104.path", device)]
        self.integrator = nn.Sequential(
            nn.Flatten(1, -1),
            # 4*12
            nn.Linear(48, 12),
        )

    def forward(self, _x):
        _y = torch.zeros(_x.shape[0], 1, 4, 12).to(self.device)
        for idx in range(self.model_num):
            _y[:, 0, idx, :] = self.cnn[idx](_x)
        _y = self.integrator(_y)
        return _y


# 训练循环函数
def train_loop(dataloader, model, _loss_fn, _optimizer):
    size = len(dataloader.dataset)
    _train_loss = 0
    model.train()
    for batch, (_x, _y) in enumerate(dataloader):
        _x = _x.to(device)
        _y = _y.to(device)
        pred = model(_x)
        loss = _loss_fn(pred, _y)
        _train_loss += loss.to("cpu")

        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.to("cpu").item(), batch * len(_x)
            print(f"loss: {loss:>8f}  [{current:>5d}/{size:>5d}]")
    _train_loss = _train_loss / (batch+1)
    return _train_loss


def test_loop(dataloader, model, _loss_fn):
    num_batches = len(dataloader)
    _test_loss, correct = 0, 0
    model.eval()

    with torch.no_grad():
        for _x, _y in dataloader:
            _x = _x.to(device)
            _y = _y.to(device)
            pred = model(_x)
            _test_loss += _loss_fn(pred, _y).to("cpu")

    _test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {_test_loss:>8f} \n")
    return _test_loss


if __name__ == '__main__':
    # 数据长度
    data_len = 70407
    # 数据库名
    db = r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db"
    # 训练参数设置
    batch_size = 2048
    learning_rate = 0.001
    # 设置训练代数
    epochs = 100
    # 构建torch格式的数据库
    dataset = DatasetForSLF(path=db, _data_len=data_len)
    dataset_train, dataset_test = random_split(dataset=dataset, lengths=[int(0.8*data_len), data_len-int(0.8*data_len)])
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size)

    device = "cuda"
    # 开始训练
    stacking_cnn = stacking_CNN(device).to(device)
    # 误差函数
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    # 优化方法
    optimizer = torch.optim.Adam(stacking_cnn.parameters(), lr=learning_rate)

    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss[t] = train_loop(dataloader_train, stacking_cnn, loss_fn, optimizer)
        test_loss[t] = test_loop(dataloader_test, stacking_cnn, loss_fn)
        if (t+1) % 10 == 0 and (t+1) >= 50:
            sns.lineplot(x=range(1, (t+1) + 1), y=train_loss[0:(t+1)])
            sns.lineplot(x=range(1, (t+1) + 1), y=test_loss[0:(t+1)])
            plt.show()
            choice = input("continue? Y/N")
            if choice == 'n' or choice == 'N':
                break
    print("Done!")
    # 输出损失曲线
    sns.lineplot(x=range(1, epochs+1), y=train_loss)
    sns.lineplot(x=range(1, epochs+1), y=test_loss)
    plt.show()
    # 保存模型参数
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(stacking_cnn.state_dict(), f"stacking_CNN_{date}.path")
