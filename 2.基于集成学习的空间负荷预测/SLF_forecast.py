import sqlite3
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import get_autoencoder1, evaluate, save_variable, load_variable
from AE2_extraction import get_autoencoder2, extraction_all_month
from SLF_train_test import CNN, get_neighboring_load

# 读取卷积神经网络模型
def get_cnn(path):
    device = "cpu"
    _cnn = CNN().to(device)
    # 加载参数
    params = torch.load(path, map_location=torch.device(device))
    # 应用到网络结构中
    _cnn.load_state_dict(params)
    _cnn.eval()
    return _cnn


# 空间负荷预测模型（输入是自身和周围的12个特征指标值）
def forecast(_cnn, _features):
    device = "cpu"
    _x = torch.from_numpy(_features).float().unsqueeze(0).unsqueeze(0)
    _x = _x.to(device)
    # 输入模型
    _pred = _cnn(_x)
    _pred = _pred.to("cpu").detach().numpy()

    return np.reshape(_pred, 12)


def get_features_by_name(load_name, year):
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    cur = conn.cursor()
    _conn = sqlite3.connect("负荷相邻数据表.db")
    _cur = _conn.cursor()

    features = np.zeros((4, 12, 12))
    auto_encoder = get_autoencoder2("AutoEncoder_20230121_163138.path")

    cur.execute('''select * from "负荷数据表" where "负荷名称" = ? AND "年份" = ? ''', (load_name, year, ))
    conn.commit()
    result = cur.fetchall()
    load_profile_365 = np.array(result[0][33:33 + 365])
    social_index = np.array(result[0][9:33])
    features[0, :, :] = extraction_all_month(_auto_encoder=auto_encoder, _social_index=social_index,
                                             _load_profile_365=load_profile_365)

    radius = ["0至100m以内", "100至200m以内", "200至300m以内"]
    for cnt, neighboring_condition in enumerate(radius):
        neighboring_load = get_neighboring_load(_conn, _cur, neighboring_condition, load_name)
        if len(neighboring_load) > 0:
            features[cnt + 1, :, :] = np.zeros((12, 12))
            for load_name in neighboring_load:
                cur.execute('''select * from "负荷数据表" where "负荷名称" = ? ''', (load_name,))
                conn.commit()
                result = cur.fetchall()
                load_profile_365 = np.array(result[0][33:33 + 365])
                social_index = np.array(result[0][9:33])
                features[cnt + 1, :, :] += extraction_all_month(_auto_encoder=auto_encoder, _social_index=social_index,
                                                                _load_profile_365=load_profile_365)
            features[cnt + 1, :, :] /= len(neighboring_load)
        else:
            features[cnt + 1, :, :] = features[0, :, :]

    return features


def get_load_profile_12_by_name(load_name, year, have_abnormal_component):
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    cur = conn.cursor()

    _load_profile_12 = np.zeros(12)
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    auto_encoder = get_autoencoder1(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\AutoEncoder_20230118_213844.path")

    cur.execute('''select * from "负荷数据表" where "负荷名称" = ? AND "年份" = ? ''', (load_name, year,))
    conn.commit()
    result = cur.fetchall()

    sample = np.array(result[0][33:33+365]) / 1000
    # 输入模型
    normal_component, abnormal_component, mse = evaluate(_auto_encoder=auto_encoder, _sample=sample)
    for index, month in enumerate(months):
        start = 0
        for month_before in range(1, month):
            start += days[month_before]
        end = start + days[month]
        if have_abnormal_component:
            _load_profile_12[index] = np.max(normal_component[start:end] + abnormal_component[start:end])
        else:
            _load_profile_12[index] = np.max(normal_component[start:end])
    return _load_profile_12


if __name__ == '__main__':
    cnn = get_cnn("CNN_20230125_102712.path")

    data_len = 70407
    indexes = [1007+data_len, 1005+data_len, 1003+data_len]
    for idx in indexes:
        # 获取数据
        conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
        cur = conn.cursor()
        cur.execute('''select * from "负荷数据表" where "field1" = ? ''', (idx,))
        conn.commit()
        result = cur.fetchall()

        features = get_features_by_name(result[0][1], 2017)

        # 输入模型
        pred = forecast(_cnn=cnn, _features=features)
        real_normal = get_load_profile_12_by_name(result[0][1], 2018, False)
        real_total = get_load_profile_12_by_name(result[0][1], 2018, True)
        real_normal_past = get_load_profile_12_by_name(result[0][1], 2017, False)
        mse = np.average(np.power(real_normal-pred, 2))
        # 展示结果
        print(f"MSE:{mse:>8f}")
        sns.set_context({'figure.figsize': [10, 5]})
        sns.lineplot(x=range(1, 12+1), y=real_total)
        sns.lineplot(x=range(1, 12+1), y=pred, linestyle='--')
        sns.lineplot(x=range(1, 12+1), y=real_normal, linestyle=':')
        sns.lineplot(x=range(1, 12+1), y=real_normal_past, linestyle='-.')
        plt.show()
