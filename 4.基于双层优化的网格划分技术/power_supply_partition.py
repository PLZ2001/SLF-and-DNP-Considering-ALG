# 划分原则：根据供电区域（只考虑最高等级和其他等级）、行政划分、供电所范围进行供电分区划分，一个供电分区只能属于1种供电区域、行政划分、供电所

import numpy as np
import sqlite3
from GIS_object import get_block_index
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import time
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import save_variable, load_variable


def print_administrative_division_map(GIS_object):
    x = GIS_object.administrative_division_map[:, 0]
    y = GIS_object.administrative_division_map[:, 1]
    administrative_division = GIS_object.administrative_division_map[:, 2]
    plt.scatter(x, y, s=1, c=administrative_division, cmap="viridis")
    plt.colorbar()
    plt.show()


def print_TS_area_map(GIS_object):
    x = GIS_object.TS_area_map[:, 0]
    y = GIS_object.TS_area_map[:, 1]
    TS_area = GIS_object.TS_area_map[:, 2]
    plt.scatter(x, y, s=1, c=TS_area, cmap="viridis")
    plt.colorbar()
    plt.show()


def print_power_supply_partition_map(GIS_object):
    x = GIS_object.power_supply_partition_map[:, 0]
    y = GIS_object.power_supply_partition_map[:, 1]
    ower_supply_partition = GIS_object.power_supply_partition_map[:, 2]
    plt.scatter(x, y, s=1, c=ower_supply_partition, cmap="viridis")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    GIS_object = load_variable("power_supply_area_optimization_20230322_182429.gisobj")

    highest_area = 'A'

    # 给行政划分网格赋值
    print("正在生成行政划分网格。。。")
    GIS_object.administrative_divisions_unit = ["industrial", "rural", "urban-suburban"]
    GIS_object.administrative_divisions = []
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    results = cur.fetchall()
    # 先统计一共有几种行政划分
    for (idx,result) in enumerate(results):
        administrative_division = []
        for (idx_administrative_division_unit, administrative_division_unit) in enumerate(GIS_object.administrative_divisions_unit):
            if result[3] == administrative_division_unit:
                administrative_division.append(administrative_division_unit)
        administrative_division = "+".join(administrative_division)
        if administrative_division not in GIS_object.administrative_divisions:
            GIS_object.administrative_divisions.append(administrative_division)
    # 行政划分
    GIS_object.administrative_division_map = np.zeros((len(results), 4))
    for (idx,result) in enumerate(results):
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(result[4], result[5])
        GIS_object.administrative_division_map[idx, 0] = x
        GIS_object.administrative_division_map[idx, 1] = y
        GIS_object.administrative_division_map[idx, 3] = idx
        administrative_division = []
        for (idx_administrative_division_unit, administrative_division_unit) in enumerate(GIS_object.administrative_divisions_unit):
            if result[3] == administrative_division_unit:
                administrative_division.append(administrative_division_unit)
        administrative_division = "+".join(administrative_division)
        GIS_object.administrative_division_map[idx, 2] = GIS_object.administrative_divisions.index(administrative_division)
    print_administrative_division_map(GIS_object)

    # 给供电所范围网格赋值
    print("正在生成供电所范围网格。。。")
    path_relationship = load_variable("path_relationship.np")
    GIS_object.TS_areas_unit = ["ITS0", "RTS0", "UTS0", "UTS1", "UTS2", "UTS3", "UTS4"]
    GIS_object.TS_areas = []
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    results = cur.fetchall()
    # 先统计一共有几种供电所范围
    for (idx, result) in enumerate(results):
        TS_area = []
        for (idx_TS_area_unit, TS_area_unit) in enumerate(GIS_object.TS_areas_unit):
            if path_relationship[idx, idx_TS_area_unit] == 1:
                TS_area.append(TS_area_unit)
        TS_area = "+".join(TS_area)
        if TS_area not in GIS_object.TS_areas:
            GIS_object.TS_areas.append(TS_area)
    # 供电所范围
    GIS_object.TS_area_map = np.zeros((len(results), 4))
    for (idx, result) in enumerate(results):
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(result[4], result[5])
        GIS_object.TS_area_map[idx, 0] = x
        GIS_object.TS_area_map[idx, 1] = y
        GIS_object.TS_area_map[idx, 3] = idx
        TS_area = []
        for (idx_TS_area_unit, TS_area_unit) in enumerate(GIS_object.TS_areas_unit):
            if path_relationship[idx, idx_TS_area_unit] == 1:
                TS_area.append(TS_area_unit)
        TS_area = "+".join(TS_area)
        GIS_object.TS_area_map[idx, 2] = GIS_object.TS_areas.index(TS_area)
    print_TS_area_map(GIS_object)

    # 给供电分区网格赋值
    print("正在生成供电分区网格。。。")
    GIS_object.power_supply_partitions = []
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    results = cur.fetchall()
    # 先统计一共有几种供电分区
    for (idx, result) in enumerate(results):
        # power_supply_partition = f"{'重点' if GIS_object.power_supply_areas[int(GIS_object.power_supply_area_map[idx, 2])] == highest_area else '非重点'}-{GIS_object.administrative_divisions[int(GIS_object.administrative_division_map[idx, 2])]}-{GIS_object.TS_areas[int(GIS_object.TS_area_map[idx, 2])]}"
        power_supply_partition = f"{GIS_object.administrative_divisions[int(GIS_object.administrative_division_map[idx, 2])]}-{GIS_object.TS_areas[int(GIS_object.TS_area_map[idx, 2])]}"
        if power_supply_partition not in GIS_object.power_supply_partitions:
            GIS_object.power_supply_partitions.append(power_supply_partition)
    # 供电分区
    GIS_object.power_supply_partition_map = np.zeros((len(results), 4))
    for (idx, result) in enumerate(results):
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(result[4], result[5])
        GIS_object.power_supply_partition_map[idx, 0] = x
        GIS_object.power_supply_partition_map[idx, 1] = y
        GIS_object.power_supply_partition_map[idx, 3] = idx
        # power_supply_partition = f"{'重点' if GIS_object.power_supply_areas[int(GIS_object.power_supply_area_map[idx, 2])] == highest_area else '非重点'}-{GIS_object.administrative_divisions[int(GIS_object.administrative_division_map[idx, 2])]}-{GIS_object.TS_areas[int(GIS_object.TS_area_map[idx, 2])]}"
        power_supply_partition = f"{GIS_object.administrative_divisions[int(GIS_object.administrative_division_map[idx, 2])]}-{GIS_object.TS_areas[int(GIS_object.TS_area_map[idx, 2])]}"
        GIS_object.power_supply_partition_map[idx, 2] = GIS_object.power_supply_partitions.index(power_supply_partition)
    print_power_supply_partition_map(GIS_object)

    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_variable(GIS_object, f"power_supply_partition_optimization_{date}.gisobj")

    # # 计算每个供电分区的最大负荷（不能超过1000MW）
    # partition_load_profile = np.zeros((len(GIS_object.power_supply_partitions), 12))
    # for x in range(GIS_object.horizontal_block_num):
    #     for y in range(GIS_object.vertical_block_num):
    #         partition_load_profile[int(GIS_object.power_supply_partition_map[x, y]), :] += (GIS_object.lv_load_profile[x, y, :] + GIS_object.mv_load_profile[x, y, :])
    # for idx, power_supply_partition in enumerate(GIS_object.power_supply_partitions):
    #     result = "满足导则要求" if np.max(partition_load_profile[idx, :]) <= 1000 else "不满足导则要求"
    #     print(f"{idx}：分区{power_supply_partition}的最大负荷为 {np.max(partition_load_profile[idx, :])} MW，{result}")


