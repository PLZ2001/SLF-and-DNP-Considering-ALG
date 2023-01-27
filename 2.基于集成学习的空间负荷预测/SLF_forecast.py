import sqlite3
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import get_autoencoder1, evaluate_and_get_normal_component, save_variable, load_variable
from AE2_extract import get_autoencoder2, extract_all_month
from stacking_SLF_train_test import stacking_CNN
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


# 读取集成卷积神经网络模型
def get_stacking_cnn(path):
    device = "cpu"
    _stacking_cnn = stacking_CNN(device).to(device)
    # 加载参数
    params = torch.load(path, map_location=torch.device(device))
    # 应用到网络结构中
    _stacking_cnn.load_state_dict(params)
    _stacking_cnn.eval()
    return _stacking_cnn


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
    auto_encoder = get_autoencoder2("AutoEncoder_20230125_173655.path")
    _auto_encoder = get_autoencoder1(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\AutoEncoder_20230125_123858.path")

    if year == 2016:
        cur.execute('''select * from "负荷数据表" where "负荷名称" = ? AND "年份" = ? ''', (load_name, year, ))
        conn.commit()
        result = cur.fetchall()
        load_profile_365 = np.array(result[0][33:33 + 365])
        social_index = np.array(result[0][9:33])
        features[0, :, :] = extract_all_month(_auto_encoder=auto_encoder, _social_index=social_index,
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
                    features[cnt + 1, :, :] += extract_all_month(_auto_encoder=auto_encoder, _social_index=social_index,
                                                                 _load_profile_365=load_profile_365)
                features[cnt + 1, :, :] /= len(neighboring_load)
            else:
                features[cnt + 1, :, :] = features[0, :, :]
    elif year == 2017:
        cur.execute('''select * from "负荷数据表" where "负荷名称" = ? AND "年份" = ? ''', (load_name, year, ))
        conn.commit()
        result = cur.fetchall()
        cur.execute('''select * from "负荷数据表" where "负荷名称" = ? AND "年份" = ? ''', (load_name, year-1, ))
        conn.commit()
        result1 = cur.fetchall()
        load_profile_365 = evaluate_and_get_normal_component(_auto_encoder=_auto_encoder, _old_load_profile_365=np.array(result1[0][33:33+365]), _new_load_profile_365=np.array(result[0][33:33+365]))
        social_index = np.array(result[0][9:33])
        features[0, :, :] = extract_all_month(_auto_encoder=auto_encoder, _social_index=social_index,
                                              _load_profile_365=load_profile_365)

        radius = ["0至100m以内", "100至200m以内", "200至300m以内"]
        for cnt, neighboring_condition in enumerate(radius):
            neighboring_load = get_neighboring_load(_conn, _cur, neighboring_condition, load_name)
            if len(neighboring_load) > 0:
                features[cnt + 1, :, :] = np.zeros((12, 12))
                for load_name in neighboring_load:
                    cur.execute('''select * from "负荷数据表" where "负荷名称" = ? AND "年份" = ? ''', (load_name, year, ))
                    conn.commit()
                    result = cur.fetchall()
                    cur.execute('''select * from "负荷数据表" where "负荷名称" = ? AND "年份" = ? ''', (load_name, year-1, ))
                    conn.commit()
                    result1 = cur.fetchall()
                    load_profile_365 = evaluate_and_get_normal_component(_auto_encoder=_auto_encoder, _old_load_profile_365=np.array(result1[0][33:33+365]), _new_load_profile_365=np.array(result[0][33:33+365]))
                    social_index = np.array(result[0][9:33])
                    features[cnt + 1, :, :] += extract_all_month(_auto_encoder=auto_encoder, _social_index=social_index,
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
    auto_encoder = get_autoencoder1(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\AutoEncoder_20230125_123858.path")


    cur.execute('''select * from "负荷数据表" where "负荷名称" = ? AND "年份" = ? ''', (load_name, year,))
    conn.commit()
    result = cur.fetchall()
    if have_abnormal_component:
        load_profile_365 = np.array(result[0][33:33+365]) / 1000
    else:
        if year == 2016:
            load_profile_365 = np.array(result[0][33:33+365]) / 1000
        elif year == 2017:
            cur.execute('''select * from "负荷数据表" where "负荷名称" = ? AND "年份" = ? ''', (load_name, year-1,))
            conn.commit()
            result1 = cur.fetchall()
            load_profile_365 = evaluate_and_get_normal_component(_auto_encoder=auto_encoder, _old_load_profile_365=np.array(result1[0][33:33+365]), _new_load_profile_365=np.array(result[0][33:33+365]))
            load_profile_365 = load_profile_365 / 1000
        elif year == 2018:
            cur.execute('''select * from "负荷数据表" where "负荷名称" = ? AND "年份" = ? ''', (load_name, year-1,))
            conn.commit()
            result1 = cur.fetchall()
            cur.execute('''select * from "负荷数据表" where "负荷名称" = ? AND "年份" = ? ''', (load_name, year-2,))
            conn.commit()
            result2 = cur.fetchall()
            load_profile_365 = evaluate_and_get_normal_component(_auto_encoder=auto_encoder, _old_load_profile_365=evaluate_and_get_normal_component(_auto_encoder=auto_encoder, _old_load_profile_365=np.array(result2[0][33:33+365]), _new_load_profile_365=np.array(result1[0][33:33+365])), _new_load_profile_365=np.array(result[0][33:33+365]))
            load_profile_365 = load_profile_365 / 1000
    for index, month in enumerate(months):
        start = 0
        for month_before in range(1, month):
            start += days[month_before]
        end = start + days[month]
        _load_profile_12[index] = np.max(load_profile_365[start:end])

    return _load_profile_12


if __name__ == '__main__':
    # cnn = get_cnn("CNN_20230126_111239.path")
    cnn = get_stacking_cnn("stacking_CNN_20230127_185430.path")

    data_len = 70407
    indexes = [1007, 1005, 1003]
    for idx in indexes:
        # 获取数据
        conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
        cur = conn.cursor()
        cur.execute('''select * from "负荷数据表" where "field1" = ? ''', (idx,))
        conn.commit()
        result = cur.fetchall()

        features = get_features_by_name(result[0][1], 2016)

        # 输入模型
        pred = forecast(_cnn=cnn, _features=features)
        real_normal = get_load_profile_12_by_name(result[0][1], 2017, False)
        real_total = get_load_profile_12_by_name(result[0][1], 2017, True)
        # real_normal_past = get_load_profile_12_by_name(result[0][1], 2017, False)
        mse = np.average(np.power(real_normal-pred, 2))
        # 展示结果
        print(f"MSE:{mse:>8f}")
        sns.set_context({'figure.figsize': [10, 5]})
        sns.lineplot(x=range(1, 12+1), y=pred)
        # sns.lineplot(x=range(1, 12+1), y=real_total, linestyle='--')
        sns.lineplot(x=range(1, 12+1), y=real_normal, linestyle=':')
        # sns.lineplot(x=range(1, 12+1), y=real_normal_past, linestyle='-.')
        plt.show()
