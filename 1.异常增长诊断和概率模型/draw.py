import sqlite3
import numpy as np
import pandas as pd
from AE_train_test import DatasetForAE
from AE_evaluate import get_autoencoder1, evaluate, load_variable
from KDE import generate_pdf, find_abnormal_increment


def figure_data_1(figure_name):
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    header = []
    for month in months:
        for day in range(days[month]):
            header.append(f"{month}月{day+1}日各用户峰荷增长量(MW)")
    pd_final_table = pd.DataFrame(columns=header)

    # 数据长度
    data_len = 70407
    # 数据库名
    db = r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db"
    # 构建torch格式的数据库
    dataset = DatasetForAE(path=db, _data_len=data_len)
    increment = {}
    for month in months:
        increment[month] = {}
        for day in range(days[month]):
            increment[month][day] = np.array([])
    for idx in range(dataset.__len__()):
        print(f"进度：{idx}/{dataset.__len__()}")
        data = list(dataset.__getitem__(idx)[0].numpy())
        pd_final_table.loc[len(pd_final_table)] = data
    pd_final_table.to_excel(figure_name)


def figure_data_2(figure_name):
    header = ["增长前负荷曲线(MW)", "增长后负荷曲线(MW)", "增长分量(MW)", "正常增长分量(MW)", "异常增长分量(MW)", "去除异常增长分量后的增长后负荷曲线(MW)"]
    writer = pd.ExcelWriter(figure_name)

    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    cur = conn.cursor()

    auto_encoder = get_autoencoder1("AutoEncoder_20230125_123858.path")

    data_len = 70407
    indexes = [101, 113]
    for idx in indexes:
        pd_final_table = pd.DataFrame(index=range(400), columns=header)

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

        pd_final_table.loc[0:len(increment)-1, "增长分量(MW)"] = increment
        pd_final_table.loc[0:len(increment)-1, "正常增长分量(MW)"] = normal_increment
        pd_final_table.loc[0:len(increment)-1, "异常增长分量(MW)"] = abnormal_increment
        pd_final_table.loc[0:len(increment)-1, "增长前负荷曲线(MW)"] = np.array(result[0][33:33+365]) / 1000
        pd_final_table.loc[0:len(increment)-1, "增长后负荷曲线(MW)"] = np.array(result1[0][33:33+365]) / 1000
        pd_final_table.loc[0:len(increment)-1, "去除异常增长分量后的增长后负荷曲线(MW)"] = np.array(result[0][33:33+365]) / 1000 + normal_increment

        pd_final_table.to_excel(writer, sheet_name=f"{idx}号用户")
    writer.save()
    writer.close()


def figure_data_3(figure_name):
    writer = pd.ExcelWriter(figure_name)

    header = ["x"]
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for month in months:
        header.append(f"{month}月概率密度函数f(x)")
    pd_final_table = pd.DataFrame(index=range(40000), columns=header)

    sample_matrix = load_variable("sample_matrix.kde")
    x, pdf = generate_pdf(_sample_matrix=sample_matrix)
    pd_final_table.loc[0:len(x)-1, "x"] = x
    for idx in range(12):
        pd_final_table.loc[0:len(x)-1, f"{idx+1}月概率密度函数f(x)"] = pdf[idx, :]
    pd_final_table.to_excel(writer, sheet_name="异常增长概率密度函数")

    header = ["月份"]
    p = [0.9, 0.6, 0.3, 0.1]
    for _p in p:
        header.append(f"{_p*100}%概率区间下界")
        header.append(f"{_p*100}%概率区间上界")
    pd_final_table = pd.DataFrame(index=range(12), columns=header)

    abnormal = {}
    for _p in p:
        abnormal[_p] = find_abnormal_increment(_probability=_p, _sample_matrix=sample_matrix)
    pd_final_table.loc[0:len(months)-1, "月份"] = months
    for _p in p:
        pd_final_table.loc[0:len(months)-1, f"{_p*100}%概率区间下界"] = abnormal[_p][0]
        pd_final_table.loc[0:len(months)-1, f"{_p*100}%概率区间上界"] = abnormal[_p][1]
    pd_final_table.to_excel(writer, sheet_name="异常增长概率区间")

    writer.save()
    writer.close()




if __name__ == '__main__':
    # figure_data_1("16-17年各用户各日期峰荷增长数据.xlsx")
    # figure_data_2("16-17年各用户各日期峰荷异常增长情况.xlsx")
    figure_data_3("16-17年异常增长概率模型.xlsx")
