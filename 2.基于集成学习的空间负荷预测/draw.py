import sqlite3
import numpy as np
import pandas as pd
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from SLF_forecast import get_stacking_cnn, get_features_by_name, get_load_profile_12_by_name, forecast
from KDE import find_abnormal_increment
from AE_evaluate import get_autoencoder1, evaluate_and_get_normal_component, save_variable, load_variable
from AE2_extract import get_autoencoder2, extract_all_month
from SLF_train_test import get_neighboring_load
import time
from multiprocessing import Process
from threading import Thread


def figure_data_1(figure_name):
    header = ["2016年实际负荷曲线(MW)", "2016年周围100m实际负荷曲线(MW)", "2017年实际负荷曲线(MW)", "2017年实际正常增长负荷曲线(MW)", "2017年预测正常增长负荷曲线(MW)", "2017年周围100m实际正常增长负荷曲线(MW)",  "2018年实际负荷曲线(MW)", "2018年实际正常增长负荷曲线(MW)", "2018年预测正常增长负荷曲线(MW)"]
    p = [0.9, 0.6, 0.3, 0.1]
    for _p in p:
        header.append(f"2017年预测正常增长负荷曲线(MW){_p*100}%概率区间下界")
        header.append(f"2017年预测正常增长负荷曲线(MW){_p*100}%概率区间上界")
    for _p in p:
        header.append(f"2018年预测正常增长负荷曲线(MW){_p*100}%概率区间下界")
        header.append(f"2018年预测正常增长负荷曲线(MW){_p*100}%概率区间上界")
    writer = pd.ExcelWriter(figure_name)

    sample_matrix = load_variable(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\sample_matrix.kde")
    abnormal = {}
    for _p in p:
        abnormal[_p] = find_abnormal_increment(_probability=_p, _sample_matrix=sample_matrix)

    cnn = get_stacking_cnn("stacking_CNN_20230212_191903.path")

    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    cur = conn.cursor()

    data_len = 70407
    indexes = [113, 1008, 1003]
    for idx in indexes:
        pd_final_table = pd.DataFrame(index=range(400), columns=header)

        # 获取数据
        cur.execute('''select * from "负荷数据表" where "field1" = ? ''', (idx,))
        conn.commit()
        result = cur.fetchall()

        pd_final_table.loc[0:11, "2016年实际负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2016, True, 0)
        pd_final_table.loc[0:11, "2016年周围100m实际负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2016, True, 1)
        # pd_final_table.loc[0:11, "2016年周围200m实际负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2016, True, 2)
        # pd_final_table.loc[0:11, "2016年周围300m实际负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2016, True, 3)
        features = get_features_by_name(result[0][1], 2016)
        # 输入模型
        pred = forecast(_cnn=cnn, _features=features[0:2, :, :])
        pd_final_table.loc[0:11, "2017年实际负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2017, True, 0)
        pd_final_table.loc[0:11, "2017年实际正常增长负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2017, False, 0)
        pd_final_table.loc[0:11, "2017年预测正常增长负荷曲线(MW)"] = pred
        for _p in p:
            pd_final_table.loc[0:11, f"2017年预测正常增长负荷曲线(MW){_p*100}%概率区间下界"] = pred + abnormal[_p][0]*get_load_profile_12_by_name(result[0][1], 2016, False, 0)
            pd_final_table.loc[0:11, f"2017年预测正常增长负荷曲线(MW){_p*100}%概率区间上界"] = pred + abnormal[_p][1]*get_load_profile_12_by_name(result[0][1], 2016, False, 0)


        pd_final_table.loc[0:11, "2017年周围100m实际正常增长负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2017, False, 1)
        # pd_final_table.loc[0:11, "2017年周围200m实际正常增长负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2017, False, 2)
        # pd_final_table.loc[0:11, "2017年周围300m实际正常增长负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2017, False, 3)
        _features = get_features_by_name(result[0][1], 2017)
        # 输入模型
        _pred = forecast(_cnn=cnn, _features=_features[0:2, :, :])
        pd_final_table.loc[0:11, "2018年实际负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2018, True, 0)
        pd_final_table.loc[0:11, "2018年实际正常增长负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2018, False, 0)
        pd_final_table.loc[0:11, "2018年预测正常增长负荷曲线(MW)"] = _pred
        for _p in p:
            pd_final_table.loc[0:11, f"2018年预测正常增长负荷曲线(MW){_p*100}%概率区间下界"] = _pred + abnormal[_p][0]*get_load_profile_12_by_name(result[0][1], 2017, False, 0)
            pd_final_table.loc[0:11, f"2018年预测正常增长负荷曲线(MW){_p*100}%概率区间上界"] = _pred + abnormal[_p][1]*get_load_profile_12_by_name(result[0][1], 2017, False, 0)

        pd_final_table.to_excel(writer, sheet_name=f"{idx}号用户")
    writer.save()
    writer.close()


def figure_data_2(figure_name, start_idx, end_idx):
    # header = ["负荷名称", "维度Lng", "经度Lat", "2016年最大负荷(MW)", "2017年最大负荷(MW)", "2018年最大负荷(MW)", "2018年最大正常负荷预测值(MW)"]
    header = ["负荷名称", "维度Lng", "经度Lat"]
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    years = [2016, 2017, 2018]
    for year in years:
        for month in months:
            header.append(f"{year}年负荷曲线{month}月(MW)")
    for month in months:
        header.append(f"2018年正常负荷预测曲线{month}月(MW)")
    p = [0.9, 0.6, 0.3, 0.1]
    for _p in p:
        for month in months:
            header.append(f"2018年{_p*100}%概率异常增长负荷预测曲线上界{month}月(MW)")
        for month in months:
            header.append(f"2018年{_p*100}%概率异常增长负荷预测曲线下界{month}月(MW)")

    sample_matrix = load_variable(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\sample_matrix.kde")
    abnormal = {}
    for _p in p:
        abnormal[_p] = find_abnormal_increment(_probability=_p, _sample_matrix=sample_matrix)

    cnn = get_stacking_cnn("stacking_CNN_20230212_191903.path")

    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    cur = conn.cursor()

    _conn = sqlite3.connect("负荷相邻数据表.db")
    _cur = _conn.cursor()

    features = np.zeros((4, 12, 24))
    auto_encoder = get_autoencoder2("AutoEncoder_20230125_173655.path")
    _auto_encoder = get_autoencoder1(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\AutoEncoder_20230125_123858.path")


    data_len = 70407
    pd_final_table = pd.DataFrame(index=range(start_idx, end_idx+1), columns=header)

    results = {}
    for year in years:
        # 获取数据
        cur.execute('''select * from "负荷数据表" where "年份" = ? ''', (year, ))
        conn.commit()
        results[year] = cur.fetchall()

        # 用平均值来填补缺失的社会指标
        # 存在缺失值的指标 26-32
        missing_indexes = [26, 27, 28, 29, 30, 31, 32]
        missing_values = {26: 0, 27: 0, 28: "-nan(ind)", 29: "-nan(ind)", 30: "-nan(ind)", 31: "-nan(ind)", 32: "-nan(ind)"}
        for missing_index in missing_indexes:
            # 计算平均值
            value_list = []
            for idx in range(data_len):
                if results[year][idx][missing_index] != missing_values[missing_index]:
                    value_list.append(results[year][idx][missing_index])
            average_value = np.average(np.array(value_list))
            # 填补缺失值
            for idx in range(data_len):
                if results[year][idx][missing_index] == missing_values[missing_index]:
                    temp = list(results[year][idx])
                    temp[missing_index] = average_value
                    results[year][idx] = tuple(temp)

    # 建立名字和idx的映射表
    name_map = {}
    for idx in range(data_len):
        name_map[results[2016][idx][1]] = idx

    start_time = time.time()
    for idx in range(start_idx, end_idx+1):
        print(idx)
        # 获取数据

        name = results[2016][idx][1]

        content = [name, results[2016][idx][3], results[2016][idx][4]]
        for year in years:
            _load_profile_12 = np.zeros(12)
            load_profile_365 = np.array(results[year][idx][33:33+365]) / 1000
            for index, month in enumerate(months):
                start = 0
                for month_before in range(1, month):
                    start += days[month_before]
                end = start + days[month]
                _load_profile_12[index] = np.max(load_profile_365[start:end])
            content.extend(list(_load_profile_12))


        load_name = name
        load_profile_365 = evaluate_and_get_normal_component(_auto_encoder=_auto_encoder, _old_load_profile_365=np.array(results[2016][idx][33:33+365]), _new_load_profile_365=np.array(results[2017][idx][33:33+365]))
        social_index = np.array(results[2017][idx][9:33])
        features[0, :, :] = extract_all_month(_auto_encoder=auto_encoder, _social_index=social_index,
                                              _load_profile_365=load_profile_365)
        radius = ["0至100m以内"]
        for cnt, neighboring_condition in enumerate(radius):
            neighboring_load = get_neighboring_load(_conn, _cur, neighboring_condition, load_name)
            if len(neighboring_load) > 0:
                features[cnt + 1, :, :] = np.zeros((12, 24))
                for load_name in neighboring_load:
                    _load_profile_365 = evaluate_and_get_normal_component(_auto_encoder=_auto_encoder, _old_load_profile_365=np.array(results[2016][name_map[load_name]][33:33+365]), _new_load_profile_365=np.array(results[2017][name_map[load_name]][33:33+365]))
                    social_index = np.array(results[2017][name_map[load_name]][9:33])
                    features[cnt + 1, :, :] += extract_all_month(_auto_encoder=auto_encoder, _social_index=social_index,
                                                                 _load_profile_365=_load_profile_365)
                features[cnt + 1, :, :] /= len(neighboring_load)
            else:
                features[cnt + 1, :, :] = features[0, :, :]

        # 输入模型
        pred = forecast(_cnn=cnn, _features=features[0:2, :, :])
        content.extend(list(pred))

        _load_profile_12 = np.zeros(12)
        load_profile_365 = load_profile_365 / 1000
        for index, month in enumerate(months):
            start = 0
            for month_before in range(1, month):
                start += days[month_before]
            end = start + days[month]
            _load_profile_12[index] = np.max(load_profile_365[start:end])

        for _p in p:
            content.extend(list(pred + abnormal[_p][1] * _load_profile_12))
            content.extend(list(pred + abnormal[_p][0] * _load_profile_12))
        pd_final_table.loc[idx] = content

        end_time = time.time()
        print(f"预计剩余时间：{(end_time-start_time)/(idx-start_idx+1)*(end_idx-start_idx+1) - (end_time-start_time)}s")
    pd_final_table.to_excel(figure_name)


if __name__ == '__main__':
    # figure_data_1("基于集成学习的空间负荷预测结果.xlsx")

    task1 = Thread(target=figure_data_2, args=("全域的基于集成学习的空间负荷预测结果(0-17601).xlsx", 0, 17601, ))
    task2 = Thread(target=figure_data_2, args=("全域的基于集成学习的空间负荷预测结果(17602-35202).xlsx", 17602, 35202, ))
    task3 = Thread(target=figure_data_2, args=("全域的基于集成学习的空间负荷预测结果(35203-52803).xlsx", 35203, 52803, ))
    task4 = Thread(target=figure_data_2, args=("全域的基于集成学习的空间负荷预测结果(52804-70406).xlsx", 52804, 70406, ))
    task1.start()
    task2.start()
    task3.start()
    task4.start()
    task1.join()
    task2.join()
    task3.join()
    task4.join()

