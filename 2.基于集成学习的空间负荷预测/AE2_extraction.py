import sqlite3
import numpy as np
import torch
import torch.nn as nn
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
import statsmodels.api as sm
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import get_autoencoder1, evaluate, save_variable, load_variable
from AE2_train_test import AutoEncoder, monthly_maximum_load, monthly_average_load, monthly_minimum_load, monthly_load_rate, seasonal_unbalance_coefficient, annual_load_rate, normalization


# 读取自编码器模型
def get_autoencoder2(path):
    device = "cpu"
    _auto_encoder = AutoEncoder().to(device)
    # 加载参数
    params = torch.load(path, map_location=torch.device(device))
    # 应用到网络结构中
    _auto_encoder.load_state_dict(params)
    _auto_encoder.eval()
    return _auto_encoder


# 特征指标提取模型（输入是24个社会指标原始值，和365点的年负荷曲线原始值）
def extraction(_auto_encoder, _social_index, _load_profile_365):
    # 社会指标
    __social_index = np.zeros((1, 20))
    __social_index[0, 0:13] = _social_index[0:13]
    __social_index[0, 13:20] = _social_index[17:24]

    # 电力指标
    electrical_index = np.zeros((1, 50))
    __auto_encoder = get_autoencoder1(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\AutoEncoder_20230118_213844.path")
    __load_profile_365 = np.array(_load_profile_365)
    __load_profile_365, abnormal_component, mse = evaluate(_auto_encoder=__auto_encoder, _sample=__load_profile_365 / 1000)
    __load_profile_365 *= 1000
    electrical_index[0, 0:0+12] = monthly_maximum_load(_load_profile_365=__load_profile_365)
    electrical_index[0, 12:12+12] = monthly_average_load(_load_profile_365=__load_profile_365)
    electrical_index[0, 24:24+12] = monthly_minimum_load(_load_profile_365=__load_profile_365)
    electrical_index[0, 36:36+12] = monthly_load_rate(_load_profile_365=__load_profile_365)
    electrical_index[0, 48] = seasonal_unbalance_coefficient(_load_profile_365=__load_profile_365)
    electrical_index[0, 49] = annual_load_rate(_load_profile_365=__load_profile_365)

    # 合体
    _index = np.concatenate((__social_index, electrical_index), axis=1)
    # 归一化
    _index = normalization(_index)

    device = "cpu"
    _x = torch.from_numpy(_index).float()
    _x = _x.to(device)
    # 输入模型
    _pred = _auto_encoder(_x)
    _extra = _auto_encoder.encoder(_x)
    _pred = _pred.to("cpu").detach().numpy()
    _extra = _extra.to("cpu").detach().numpy()
    _mse = np.average(np.power(_index-_pred, 2))

    return np.reshape(_index, 70), np.reshape(_pred, 70), np.reshape(_extra, 32), _mse


if __name__ == '__main__':
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    cur = conn.cursor()

    auto_encoder = get_autoencoder2("AutoEncoder_20230120_121634.path")

    data_len = 70407*2
    indexes = [1100, 1300]  # 1100 1300
    for idx in indexes:
        # 获取数据
        cur.execute('''select * from "负荷数据表" where "field1" = ? ''', (idx,))
        conn.commit()
        result = cur.fetchall()
        load_profile_365 = np.array(result[0][33:33+365])
        social_index = np.array(result[0][9:33])
        # 输入模型
        index, pred, extra, mse = extraction(_auto_encoder=auto_encoder, _social_index=social_index, _load_profile_365=load_profile_365)
        # 展示结果
        print(f"MSE:{mse:>8f}")
        sns.set_context({'figure.figsize': [15, 5]})
        sns.lineplot(x=range(1, 70+1), y=index)
        sns.lineplot(x=range(1, 70+1), y=pred, linestyle='--')
        plt.show()
        sns.lineplot(x=range(1, 32+1), y=extra, linestyle=':')
        plt.show()
