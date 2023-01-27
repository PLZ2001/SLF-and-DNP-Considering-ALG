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
def evaluate(_auto_encoder, _increment):
    device = "cpu"
    _x = torch.from_numpy(_increment).float()
    _x = _x.to(device)
    _pred = _auto_encoder(_x)
    _pred = _pred.to("cpu").detach().numpy()
    _mse = np.average(np.power(_increment-_pred, 2))
    _normal_increment = _pred
    _abnormal_increment = _increment - _normal_increment
    rd = sm.tsa.seasonal_decompose(_abnormal_increment, period=7)
    _abnormal_increment_trend = np.array(rd.trend)
    _abnormal_increment_trend[0:3] = _abnormal_increment[0:3]
    _abnormal_increment_trend[362:365] = _abnormal_increment[362:365]
    _normal_increment = _increment - _abnormal_increment_trend
    return _normal_increment, _abnormal_increment_trend, _mse


# 根据异常增长分量诊断模型获取正常增长的结果（输入是365点的年负荷曲线）
def evaluate_and_get_normal_component(_auto_encoder, _old_load_profile_365, _new_load_profile_365):
    _increment = (_new_load_profile_365 - _old_load_profile_365) / 1000
    # 输入模型
    normal_increment, abnormal_increment, mse = evaluate(_auto_encoder=_auto_encoder, _increment=_increment)
    return (_old_load_profile_365 + 1000 * normal_increment)


if __name__ == '__main__':
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    cur = conn.cursor()

    auto_encoder = get_autoencoder1("AutoEncoder_20230125_123858.path")

    data_len = 70407
    indexes = [101]
    for idx in indexes:
        # 获取数据
        cur.execute('''select * from "负荷数据表" where "field1" = ? ''', (idx,))
        conn.commit()
        result = cur.fetchall()

        cur.execute('''select * from "负荷数据表" where "field1" = ? ''', (idx+data_len,))
        conn.commit()
        result1 = cur.fetchall()
        increment = np.array(result1[0][33:33+365]) / 1000 - np.array(result[0][33:33+365]) / 1000
        # 输入模型
        normal_increment, abnormal_increment, mse = evaluate(_auto_encoder=auto_encoder, _increment=increment)
        # 展示结果
        print(f"MSE:{mse:>8f}")
        print(f"LB检验结果:{lb_test(abnormal_increment)}")
        sns.set_context({'figure.figsize': [15, 5]})
        sns.lineplot(x=range(1, 365+1), y=increment)
        sns.lineplot(x=range(1, 365+1), y=normal_increment, linestyle='--')
        sns.lineplot(x=range(1, 365+1), y=abnormal_increment, linestyle=':')
        plt.show()
        sns.lineplot(x=range(1, 365+1), y=np.array(result[0][33:33+365]) / 1000)
        sns.lineplot(x=range(1, 365+1), y=np.array(result1[0][33:33+365]) / 1000, linestyle='--')
        sns.lineplot(x=range(1, 365+1), y=np.array(result[0][33:33+365]) / 1000 + normal_increment, linestyle=':')
        plt.show()
