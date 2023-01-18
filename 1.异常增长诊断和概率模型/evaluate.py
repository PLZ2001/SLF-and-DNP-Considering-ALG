import sqlite3
import numpy as np
import torch
import torch.nn as nn
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
import statsmodels.api as sm


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


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


# 读取自编码器模型
def get_autoencoder(path):
    device = "cpu"
    _auto_encoder = AutoEncoder().to(device)
    # 加载参数
    params = torch.load(path, map_location=torch.device(device))
    # 应用到网络结构中
    _auto_encoder.load_state_dict(params)
    _auto_encoder.eval()
    return _auto_encoder


# 异常增长分量诊断模型
def evaluate(_auto_encoder, _sample):
    device = "cpu"
    _x = torch.from_numpy(_sample).float()
    _x = _x.to(device)
    _pred = _auto_encoder(_x)
    _pred = _pred.to("cpu").detach().numpy()
    _mse = np.average(np.power(_sample-_pred, 2))
    _normal_component = _pred
    _abnormal_component = _sample - _normal_component
    rd = sm.tsa.seasonal_decompose(_abnormal_component, period=7)
    _abnormal_component_trend = np.array(rd.trend)
    _abnormal_component_trend[0:3] = _abnormal_component[0:3]
    _abnormal_component_trend[362:365] = _abnormal_component[362:365]
    _normal_component = _sample - _abnormal_component_trend
    return _normal_component, _abnormal_component_trend, _mse


if __name__ == '__main__':
    conn = sqlite3.connect('负荷数据表.db')
    cur = conn.cursor()

    auto_encoder = get_autoencoder("AutoEncoder_20230117_132712.path")

    data_len = 3040
    indexes = [1101, 102, 103]  # 夏峰型1101 无峰无谷型102 夏谷型103
    for idx in indexes:
        # 获取数据
        cur.execute('''select * from "负荷数据表" where "field1" = ? ''', (idx,))
        conn.commit()
        result = cur.fetchall()
        sample = np.array(result[0][34:34+365]) / 1000
        # 输入模型
        normal_component, abnormal_component, mse = evaluate(_auto_encoder=auto_encoder, _sample=sample)
        # 展示结果
        print(f"MSE:{mse:>8f}")
        print(f"LB检验结果:{lb_test(abnormal_component)}")
        sns.set_context({'figure.figsize': [15, 5]})
        sns.lineplot(x=range(1, 365+1), y=sample)
        sns.lineplot(x=range(1, 365+1), y=normal_component, linestyle='--')
        sns.lineplot(x=range(1, 365+1), y=abnormal_component, linestyle=':')
        plt.show()
