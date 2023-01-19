import sqlite3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


# 所有的数据集
class DatasetForAE(Dataset):
    def __init__(self, path, _data_len):
        self.data_len = _data_len
        self.conn = sqlite3.connect(path)
        self.cur = self.conn.cursor()
        self.cur.execute('''select * from "负荷数据表" where "年份" = 2016''')
        self.conn.commit()
        self.results = self.cur.fetchall()

    # 用来获取样本的总数目
    def __len__(self):
        return self.data_len

    # 通过idx来获取数据库的输入和输出
    def __getitem__(self, idx):
        # 以1000kW为基准值进行标幺化
        _input = torch.from_numpy(np.array(self.results[idx][33:33+365])) / 1000
        _input = _input.float()
        return _input, _input


# AE网络模型
class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(365, 128),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 365),
        )

    def forward(self, _x):
        _x = self.encoder(_x)
        _x = self.decoder(_x)
        return _x


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
    db = "负荷数据表.db"
    # 训练参数设置
    batch_size = 1024
    learning_rate = 0.0001
    # 设置训练代数
    epochs = 20
    # 构建torch格式的数据库
    dataset = DatasetForAE(path=db, _data_len=data_len)
    dataset_train, dataset_test = random_split(dataset=dataset, lengths=[int(0.8*data_len), data_len-int(0.8*data_len)])
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size)

    # 画出所有的训练集
    # cnt = 0
    # for x, y in dataloader_train:
    #     data = x.numpy() if cnt == 0 else np.concatenate((data, x.numpy()), axis=0)
    #     cnt += 1
    # for row in range(np.size(data, 0)):
    #     sns.lineplot(x=range(1, 365+1), y=data[row, :])
    # plt.show()

    device = "cuda"
    # 开始训练
    auto_encoder = AutoEncoder().to(device)
    # 误差函数
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    # 优化方法
    optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=learning_rate)

    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss[t] = train_loop(dataloader_train, auto_encoder, loss_fn, optimizer)
        test_loss[t] = test_loop(dataloader_test, auto_encoder, loss_fn)
    print("Done!")
    # 输出损失曲线
    sns.lineplot(x=range(1, epochs+1), y=train_loss)
    sns.lineplot(x=range(1, epochs+1), y=test_loss)
    plt.show()
    # 保存模型参数
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(auto_encoder.state_dict(), f"AutoEncoder_{date}.path")