import sqlite3
import numpy as np
import torch
import torch.nn as nn
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
import statsmodels.api as sm
from AE_train_test import AutoEncoder


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


# 读取自编码器模型
def get_autoencoder1(path):
    device = "cpu"
    _auto_encoder = AutoEncoder().to(device)
    # 加载参数
    params = torch.load(path, map_location=torch.device(device))
    # 应用到网络结构中
    _auto_encoder.load_state_dict(params)
    _auto_encoder.eval()
    return _auto_encoder


# 异常增长分量诊断模型（输入是365点的年负荷曲线（而且是基值为1000kW的标幺值））
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
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    cur = conn.cursor()

    auto_encoder = get_autoencoder1("AutoEncoder_20230118_213844.path")

    data_len = 70407
    indexes = [101+70407, 112+70407]  # 101夏谷型 112夏峰型
    for idx in indexes:
        # 获取数据
        cur.execute('''select * from "负荷数据表" where "field1" = ? ''', (idx,))
        conn.commit()
        result = cur.fetchall()
        sample = np.array(result[0][33:33+365]) / 1000
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
