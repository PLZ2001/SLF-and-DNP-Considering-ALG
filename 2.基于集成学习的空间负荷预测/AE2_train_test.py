import sqlite3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划1.异常增长诊断和概率模型")
from AE_evaluate import get_autoencoder1, evaluate, evaluate_and_get_normal_component, save_variable, load_variable


# 所有的数据集
class DatasetForAE(Dataset):
    def __init__(self, path, _data_len):
        self.data_len = _data_len

        self.conn = sqlite3.connect(path)
        self.cur = self.conn.cursor()
        self.cur.execute('''select * from "负荷数据表" where "年份" = 2016 OR "年份" = 2017''')
        self.conn.commit()
        self.results = self.cur.fetchall()
        # 填补缺失值
        self.missing_value_filling()
        # 获取社会指标
        self.social_index = self.get_social_index()
        # 获取电力指标
        self.electrical_index = self.get_electrical_index()
        # 指标合体
        self.index = np.concatenate((self.social_index, self.electrical_index), axis=1)
        # 保存指标
        save_variable(self.index, "index.dataset")
        self.index = load_variable("index.dataset")
        # 指标归一化
        self.index = normalization(self.index)

    # 用来获取样本的总数目
    def __len__(self):
        return self.data_len*12

    # 通过idx来获取数据库的输入和输出
    def __getitem__(self, idx):
        _input = torch.from_numpy(self.index[idx, :])
        _input = _input.float()
        return _input, _input

    def missing_value_filling(self):
        # 用平均值来填补缺失的社会指标
        # 存在缺失值的指标 26-32
        missing_indexes = [26, 27, 28, 29, 30, 31, 32]
        missing_values = {26: 0, 27: 0, 28: "-nan(ind)", 29: "-nan(ind)", 30: "-nan(ind)", 31: "-nan(ind)", 32: "-nan(ind)"}
        for missing_index in missing_indexes:
            # 计算平均值
            value_list = []
            for idx in range(self.data_len):
                if self.results[idx][missing_index] != missing_values[missing_index]:
                    value_list.append(self.results[idx][missing_index])
            average_value = np.average(np.array(value_list))
            # 填补缺失值
            for idx in range(self.data_len):
                if self.results[idx][missing_index] == missing_values[missing_index]:
                    temp = list(self.results[idx])
                    temp[missing_index] = average_value
                    self.results[idx] = tuple(temp)

    def get_social_index(self):
        # 20个社会指标 9-21 26-32
        social_index = np.zeros((self.data_len*12, 20))
        for idx in range(self.data_len):
            social_index[idx*12:(idx+1)*12, 0:13] = np.tile(self.results[idx][9:22], (12, 1))
            social_index[idx*12:(idx+1)*12, 13:20] = np.tile(self.results[idx][26:33], (12, 1))
        return social_index

    def get_electrical_index(self):
        # 4个电力指标
        electrical_index = np.zeros((self.data_len*12, 4))
        auto_encoder = get_autoencoder1(fr"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划1.异常增长诊断和概率模型\AutoEncoder_20230125_123858.path")
        for idx in range(self.data_len):
            if idx % 100 == 0:
                print(idx)
            if idx < 70407:
                load_profile_365 = np.array(self.results[idx][33:33+365])
            else:
                load_profile_365 = evaluate_and_get_normal_component(_auto_encoder=auto_encoder, _old_load_profile_365=np.array(self.results[idx - 70407][33:33+365]), _new_load_profile_365=np.array(self.results[idx][33:33+365]))
            electrical_index[idx*12:(idx+1)*12, 0] = monthly_maximum_load(_load_profile_365=load_profile_365)
            electrical_index[idx*12:(idx+1)*12, 1] = monthly_average_load(_load_profile_365=load_profile_365)
            electrical_index[idx*12:(idx+1)*12, 2] = monthly_minimum_load(_load_profile_365=load_profile_365)
            electrical_index[idx*12:(idx+1)*12, 3] = monthly_load_rate(_load_profile_365=load_profile_365)
        return electrical_index


# 月最大负荷
def monthly_maximum_load(_load_profile_365):
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    result = np.zeros(12)
    for idx, month in enumerate(months):
        start = 0
        for month_before in range(1, month):
            start += days[month_before]
        end = start + days[month]
        result[idx] = np.max(_load_profile_365[start:end])
    return result


# 月平均负荷
def monthly_average_load(_load_profile_365):
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    result = np.zeros(12)
    for idx, month in enumerate(months):
        start = 0
        for month_before in range(1, month):
            start += days[month_before]
        end = start + days[month]
        result[idx] = np.average(_load_profile_365[start:end])
    return result


# 月最小负荷
def monthly_minimum_load(_load_profile_365):
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    result = np.zeros(12)
    for idx, month in enumerate(months):
        start = 0
        for month_before in range(1, month):
            start += days[month_before]
        end = start + days[month]
        result[idx] = np.min(_load_profile_365[start:end])
    return result


# 月负荷率 = 月平均负荷/月最大负荷
def monthly_load_rate(_load_profile_365):
    result = monthly_average_load(_load_profile_365) / monthly_maximum_load(_load_profile_365)
    return result


# 季不均衡系数 = 月最大负荷的平均值 / 年最大负荷
def seasonal_unbalance_coefficient(_load_profile_365):
    result = np.average(monthly_maximum_load(_load_profile_365)) / np.max(monthly_maximum_load(_load_profile_365))
    return result


# 年负荷率 = 年平均负荷 / 年最大负荷
def annual_load_rate(_load_profile_365):
    result = np.average(_load_profile_365) / np.max(_load_profile_365)
    return result


# 对所有指标设定固定的基值，以进行归一化
def normalization(all_indexes):
    max_base = [10, 0.2, 800, 40, 20, 5, 2000, 60000000000, 2000000000, 1000000000000, 2, 100, 300, 2000, 8000, 100, 100, 100, 100, 100]
    max_base.extend([1000]*3)
    max_base.extend([1]*1)
    min_base = [0,  0,   0,   0,  0,  0, 0,    0,           0,          0,             0, 0,   0,   1900, 0,    0,   0,   0,   0,   0]
    min_base.extend([0]*4)

    n = np.size(all_indexes, 0)
    max_base = np.tile(np.reshape(np.array(max_base), (1, 24)), (n, 1))
    min_base = np.tile(np.reshape(np.array(min_base), (1, 24)), (n, 1))

    return (all_indexes - min_base) / (max_base - min_base)


# AE网络模型
class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(24, 16),
            nn.GELU(),
            nn.Linear(16, 12),
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 16),
            nn.GELU(),
            nn.Linear(16, 24),
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
    data_len = 70407*2
    # 数据库名
    db = r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划0.数据集清洗\负荷数据表.db"
    # 训练参数设置
    batch_size = 4096
    learning_rate = 0.005
    # 设置训练代数
    epochs = 100
    # 构建torch格式的数据库
    dataset = DatasetForAE(path=db, _data_len=data_len)
    dataset_train, dataset_test = random_split(dataset=dataset, lengths=[int(0.8*data_len*12), data_len*12-int(0.8*data_len*12)])
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size)

    # 画出所有的训练集
    cnt = 0
    for x, y in dataloader_train:
        data = x.numpy() if cnt == 0 else np.concatenate((data, x.numpy()), axis=0)
        cnt += 1
        print(f"加载数据集{np.size(data, 0)}/{data_len*12}")
    sns.set_context({'figure.figsize': [10, 5]})
    for row in range(min(np.size(data, 0), 10000)):
        sns.lineplot(x=range(1, 24+1), y=data[row, :])
        if row % 100 == 0:
            print(f"绘制进度{row}/{data_len*12}")
    plt.show()

    device = "cpu"
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
