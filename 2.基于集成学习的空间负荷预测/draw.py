import sqlite3
import numpy as np
import pandas as pd
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from SLF_forecast import get_stacking_cnn, get_features_by_name, get_load_profile_12_by_name, forecast


def figure_data_1(figure_name):
    # header = ["2016年实际负荷曲线(MW)", "2016年周围100m实际负荷曲线(MW)", "2016年周围200m实际负荷曲线(MW)", "2016年周围300m实际负荷曲线(MW)", "2017年实际负荷曲线(MW)", "2017年实际正常增长负荷曲线(MW)", "2017年预测正常增长负荷曲线(MW)", "2017年周围100m实际正常增长负荷曲线(MW)", "2017年周围200m实际正常增长负荷曲线(MW)", "2017年周围300m实际正常增长负荷曲线(MW)", "2018年实际正常增长负荷曲线(MW)", "2018年预测正常增长负荷曲线(MW)"]
    header = ["2016年实际负荷曲线(MW)", "2016年周围100m实际负荷曲线(MW)", "2017年实际负荷曲线(MW)", "2017年实际正常增长负荷曲线(MW)", "2017年预测正常增长负荷曲线(MW)", "2017年周围100m实际正常增长负荷曲线(MW)",  "2018年实际负荷曲线(MW)", "2018年实际正常增长负荷曲线(MW)", "2018年预测正常增长负荷曲线(MW)"]
    writer = pd.ExcelWriter(figure_name)

    cnn = get_stacking_cnn("stacking_CNN_20230212_191903.path")

    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    cur = conn.cursor()

    data_len = 70407
    indexes = [1008, 1003]
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

        pd_final_table.loc[0:11, "2017年周围100m实际正常增长负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2017, False, 1)
        # pd_final_table.loc[0:11, "2017年周围200m实际正常增长负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2017, False, 2)
        # pd_final_table.loc[0:11, "2017年周围300m实际正常增长负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2017, False, 3)
        _features = get_features_by_name(result[0][1], 2017)
        # 输入模型
        _pred = forecast(_cnn=cnn, _features=_features[0:2, :, :])
        pd_final_table.loc[0:11, "2018年实际负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2018, True, 0)
        pd_final_table.loc[0:11, "2018年实际正常增长负荷曲线(MW)"] = get_load_profile_12_by_name(result[0][1], 2018, False, 0)
        pd_final_table.loc[0:11, "2018年预测正常增长负荷曲线(MW)"] = _pred

        pd_final_table.to_excel(writer, sheet_name=f"{idx}号用户")
    writer.save()
    writer.close()


if __name__ == '__main__':
    figure_data_1("基于集成学习的空间负荷预测结果.xlsx")

