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
from AE_evaluate import get_autoencoder1, evaluate_and_get_normal_component, save_variable, load_variable
from AE2_extract import get_autoencoder2, extract_all_month


def get_neighboring_load(_conn, _cur, condition, name):
    if condition == "0至100m以内":
        _cur.execute('''select "0至100m以内" from "负荷相邻数据表" where "负荷名称" = ?''', (name, ))
    elif condition == "100至200m以内":
        _cur.execute('''select "100至200m以内" from "负荷相邻数据表" where "负荷名称" = ?''', (name, ))
    elif condition == "200至300m以内":
        _cur.execute('''select "200至300m以内" from "负荷相邻数据表" where "负荷名称" = ?''', (name, ))
    _conn.commit()
    _results = _cur.fetchall()
    if _results[0][0] is None:
        _result = []
    else:
        _result = _results[0][0].split("|")
    return _result


# 所有的数据集
class DatasetForSLF(Dataset):
    def __init__(self, path, _data_len):
        self.data_len = _data_len
        self.radius = ["0至100m以内", "100至200m以内", "200至300m以内"]

        # self.conn = sqlite3.connect(path)
        # self.cur = self.conn.cursor()
        # self.cur.execute('''select * from "负荷数据表" where "年份" = 2016''')
        # self.conn.commit()
        # self.results = self.cur.fetchall()
        # # 建立名字和idx的映射表
        # self.name_map = {}
        # for idx in range(self.data_len):
        #     self.name_map[self.results[idx][1]] = idx
        # # 填补缺失值
        # self.missing_value_filling()
        # # 建立指标和idx的映射表
        # self.index_map = self.generate_index_map()
        # save_variable(self.index_map, "index_map.dataset")
        # self.index_map = load_variable("index_map.dataset")
        # # 获取所有三维特征数据空间
        # self.features = self.get_features()
        # save_variable(self.features, "features.dataset")
        # 获取所有的待预测12点负荷曲线
        # self.load_profile_12 = self.get_load_profile_12()
        # # save_variable(self.load_profile_12, "load_profile_12.dataset")
        # 读取

        self.features = load_variable("features.dataset")
        self.load_profile_12 = load_variable("load_profile_12.dataset")

    # 用来获取样本的总数目
    def __len__(self):
        return self.data_len

    # 通过idx来获取数据库的输入和输出
    def __getitem__(self, idx):
        _input = torch.from_numpy(self.features[idx, :, :, :])
        _input = _input.float().unsqueeze(0)  # 使输入从4,12,12变成1,4,12,12
        _output = torch.from_numpy(self.load_profile_12[idx, :])
        _output = _output.float()
        return _input, _output

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

    def get_features(self):
        # 70407个，4层，12行，12列
        features = np.zeros((self.data_len, 4, 12, 12))
        _conn = sqlite3.connect("负荷相邻数据表.db")
        _cur = _conn.cursor()
        # 遍历70407个
        for idx in range(self.data_len):
            # if idx % 1000 == 0:
            print(idx)
            # 遍历4个
            load_name = self.results[idx][1]
            features[idx, 0, :, :] = self.index_map[self.name_map[load_name], :, :]

            for cnt, neighboring_condition in enumerate(self.radius):
                    neighboring_load = get_neighboring_load(_conn, _cur, neighboring_condition, load_name)
                    if len(neighboring_load) > 0:
                        features[idx, cnt+1, :, :] = np.zeros((12, 12))
                        for load_name in neighboring_load:
                            features[idx, cnt+1, :, :] += self.index_map[self.name_map[load_name], :, :]
                        features[idx, cnt+1, :, :] /= len(neighboring_load)
                    else:
                        features[idx, cnt+1, :, :] = features[idx, 0, :, :]
        return features

    def generate_index_map(self):
        # 根据负荷名称，求出12个月的12个特征指标
        index_map = np.zeros((self.data_len, 12, 12))
        for idx in range(self.data_len):
            if idx % 100 == 0:
                print(f"正在生成指标映射表...{idx}")
            load_profile_365 = np.array(self.results[idx][33:33+365])
            social_index = np.array(self.results[idx][9:33])
            auto_encoder = get_autoencoder2("AutoEncoder_20230125_173655.path")
            index_map[idx, :, :] = extract_all_month(_auto_encoder=auto_encoder, _social_index=social_index, _load_profile_365=load_profile_365)
        return index_map

    def get_load_profile_12(self):
        _load_profile_12 = np.zeros((self.data_len, 12))
        months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        auto_encoder = get_autoencoder1(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划1.异常增长诊断和概率模型\AutoEncoder_20230125_123858.path")
        self.cur.execute('''select * from "负荷数据表" where "年份" = 2017 ''')
        self.conn.commit()
        results = self.cur.fetchall()
        for idx, result in enumerate(results):
            load_profile_365 = evaluate_and_get_normal_component(_auto_encoder=auto_encoder, _old_load_profile_365=np.array(self.results[idx][33:33+365]), _new_load_profile_365=np.array(result[33:33+365]))
            load_profile_365 = load_profile_365 / 1000
            for index, month in enumerate(months):
                start = 0
                for month_before in range(1, month):
                    start += days[month_before]
                end = start + days[month]
                _load_profile_12[idx, index] = np.max(load_profile_365[start:end])
        return _load_profile_12


# AE网络模型
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # N = (W − F + 2P )/S+1
        self.network = nn.Sequential(
            # 4,12,12 *1
            nn.Conv3d(in_channels=1, out_channels=256, kernel_size=(1, 6, 6), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.GELU(),
            # 4,7,7 *256
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.GELU(),
            # 4,5,5 *128
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.GELU(),
            # 4,5,5 *64
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.GELU(),
            # 4,5,5 *32
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.GELU(),
            # 4,5,5 *32
            nn.AdaptiveAvgPool3d((2, 3, 3)),
            # 2,3,3 *32
            nn.Flatten(1, -1),
            # 2*3*3*32
            nn.Linear(576, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 12),
        )

    def forward(self, _x):
        _x = self.network(_x)
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
    db = r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划0.数据集清洗\负荷数据表.db"
    # 训练参数设置
    batch_size = 4096
    learning_rate = 0.001
    # 设置训练代数
    epochs = 1000
    # 构建torch格式的数据库
    dataset = DatasetForSLF(path=db, _data_len=data_len)
    dataset_train, dataset_test = random_split(dataset=dataset, lengths=[int(0.8*data_len), data_len-int(0.8*data_len)])
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size)

    device = "cuda"
    # 开始训练
    cnn = CNN().to(device)
    # 误差函数
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    # 优化方法
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss[t] = train_loop(dataloader_train, cnn, loss_fn, optimizer)
        test_loss[t] = test_loop(dataloader_test, cnn, loss_fn)
    print("Done!")
    # 输出损失曲线
    sns.lineplot(x=range(1, epochs+1), y=train_loss)
    sns.lineplot(x=range(1, epochs+1), y=test_loss)
    plt.show()
    # 保存模型参数
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(cnn.state_dict(), f"CNN_{date}.path")
