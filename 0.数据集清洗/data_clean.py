import pandas as pd
import os
import json
import re
import numpy as np

# 根据SMART-DS数据集，汇总出负荷数据表.csv
# 表头：负荷名称 年份 维度 经度 区域 子区域 用户类型 馈线名称 馈线年统计量（24个） 年负荷曲线（365点）
if __name__ == '__main__':
    ROOT_PATH = r"D:\OneDrive\桌面\毕设\代码\SMART-DS数据集\SMART-DS\v1.0"
    feeder_data_labels = ["Maximum Node Distance From Substation (miles)",
                          "Maximum Line Length From Transformer to Load (miles)",
                          "Number of Transformers",
                          "Total Transformer Capacity (MVA)",
                          "Total Peak Planning Load (MW)",
                          "Average Number of Loads per Transformer",
                          "Total Number of Customers",
                          "Number of Customers per Square Mile of Feeder Convex Hull",
                          "Total Peak Planning Load (MW) per Square Mile of Feeder Convex Hull",
                          "Total Transformer Capacity (MVA) per Square Mile of Feeder Convex Hull",
                          "Average Node Degree",
                          "Average Shortest Path Length",
                          "Diameter (Maximum Eccentricity)",
                          "Number of PVs",
                          "Total PV Capacity (MW)",
                          "Number of Batteries",
                          "Total Capacity of Batteries (MW)",
                          "Average Year of Building Construction",
                          "Average Land Value (USD per Square Foot)",
                          "Percentage of Rural Customers",
                          "Percentage of Urban Customers",
                          "Percentage of Residential Customers",
                          "Percentage of Commercial Customers",
                          "Percentage of Industrial Customers"]
    years = ["2016", "2017", "2018"]
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    areas = ["GSO"]
    sub_areas = ["industrial", "rural", "urban-suburban"]

    header = ["负荷名称", "年份", "维度Lng", "经度Lat", "区域", "子区域", "用户类型", "馈线名称"]
    for label in feeder_data_labels:
        header.append(f"{label}")
    for idx in range(1, 365+1):
        header.append(f"第{idx}天")

    final_table = pd.DataFrame(columns=header)


    for year in years:
        for area in areas:
            for sub_area in sub_areas:
                print(f"正在读取{year}年{area}地区{sub_area}子地区的负荷...")
                # 该文件可获取每个负荷点的名称、经纬度、年最大功率
                base_path = ROOT_PATH + fr"\{year}\{area}\{sub_area}\scenarios\base_timeseries\geojson"
                file_names = os.listdir(base_path)
                for file_name in file_names:
                    fp = open(base_path+fr"\{file_name}", 'r', encoding='utf8')
                    feeder_json = json.load(fp)
                    # 馈线名称
                    feeder_name = re.findall("^(.+)\.json$", file_name)[0]
                    for point in feeder_json['features']:
                        if point['properties']['type'] == 'Load':
                            # 负荷名称
                            name = re.findall("^load_(.+)$", point['properties']['name'])[0]
                            if name.find("mv") != -1:
                                continue
                            # 维度
                            lng = point['geometry']['coordinates'][0]
                            # 经度
                            lat = point['geometry']['coordinates'][1]
                            maxP_kw = point['properties']['max_kw']
                            # 该文件可获取每个负荷点的用户类型
                            path = ROOT_PATH + fr"\{year}\{area}\{sub_area}\scenarios\base_timeseries\cyme\loads.txt"
                            fp = open(path+r"\loads.txt", 'r', encoding='utf8')
                            txt = fp.readlines()
                            for txt_line in txt:
                                name_found = re.findall("^.+,.*?_load_(.*?),[a-z]{3}_.*?,.+$", txt_line)
                                if name_found:
                                    if name == name_found[0]:
                                        # 用户类型
                                        customer_type = re.findall("^.+,.*?_load_.*?,([a-z]{3}_.*?),.+$", txt_line)[0]
                                        break
                            # 该文件可获取每个馈线的所有年统计量
                            path = ROOT_PATH + fr"\{year}\{area}\{sub_area}\scenarios\base_timeseries\metrics.csv"
                            feeder_staticstics = pd.read_csv(path+r"\metrics.csv")
                            idx = feeder_staticstics[(feeder_staticstics["Feeder Name"] == feeder_name)].index.tolist()[0]
                            feeder_data = []
                            for label in feeder_data_labels:
                                # 馈线指标
                                feeder_data.append(feeder_staticstics.loc[idx, label])
                            # 该文件可获取每种用户类型的典型负荷曲线
                            path = ROOT_PATH + fr"\{year}\{area}\{sub_area}\load_data"
                            customer_profile = pd.read_parquet(path + fr"\{customer_type}.parquet")
                            # 负荷曲线365*96点
                            load_profile_365_96 = np.array(customer_profile.loc[:, "total_site_electricity_kw"].values.tolist())
                            load_profile_365_96 = load_profile_365_96 / np.max(load_profile_365_96) * maxP_kw
                            # 负荷曲线365点
                            load_profile_365 = np.zeros(365)
                            cnt = 0
                            for month in months:
                                for day in range(days[month]):
                                    load_profile_365[cnt] = np.max(load_profile_365_96[cnt*96:(cnt+1)*96])
                                    cnt += 1

                            # 表头：负荷名称 年份 维度 经度 区域 子区域 用户类型 馈线名称 馈线年统计量（24个） 年负荷曲线（365点）
                            content = [name, year, lng, lat, area, sub_area, customer_type, feeder_name]
                            for idx in range(len(feeder_data_labels)):
                                content.append(feeder_data[idx])
                            for idx in range(365):
                                content.append(load_profile_365[idx])
                            final_table.loc[len(final_table)] = content
    final_table.to_csv("负荷数据表.csv")