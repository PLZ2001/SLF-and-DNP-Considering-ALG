# 划分原则：相同馈线的负荷点作为一个供电单元

import numpy as np
import time
from GIS_object import get_block_index
import sqlite3
import matplotlib.pyplot as plt
import datetime
import networkx as nx
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import save_variable, load_variable


def print_power_supply_unit_map(GIS_object):
    cnt = 0
    x = []
    y = []
    power_supply_unit = []
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        if idx_power_supply_partition >= len(GIS_object.power_supply_meshes):
            break
        for (idx_power_supply_mesh, power_supply_mesh) in enumerate(GIS_object.power_supply_meshes[idx_power_supply_partition]):
            if idx_power_supply_mesh >= len(GIS_object.power_supply_units[idx_power_supply_partition]):
                break
            if idx_power_supply_partition == 0 and idx_power_supply_mesh == 0:
                x = GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 0]
                y = GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 1]
                power_supply_unit = GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 2] + cnt
            else:
                x = np.concatenate((x, GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 0]))
                y = np.concatenate((y, GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 1]))
                power_supply_unit = np.concatenate((power_supply_unit, GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 2] + cnt))
            cnt += len(GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh])
    plt.scatter(x, y, s=1, c=power_supply_unit, cmap="viridis")
    plt.colorbar()
    plt.show()

def L1_distance(matrix):
    return np.sum(np.fabs(matrix))


if __name__ == '__main__':
    GIS_object = load_variable("power_supply_mesh_optimization_20230322_194545.gisobj")

    # 读取用户数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    users = cur.fetchall()

    GIS_object.power_supply_units = []
    GIS_object.power_supply_unit_map = []
    # 对每个供电网格分别进行讨论
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        GIS_object.power_supply_units.append([])
        GIS_object.power_supply_unit_map.append([])
        for (idx_power_supply_mesh, power_supply_mesh) in enumerate(GIS_object.power_supply_meshes[idx_power_supply_partition]):
            # 获取属于这个网格的用户id
            user_ids_of_partition = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 3].tolist()
            user_ids = []
            for i in range(len(user_ids_of_partition)):
                if GIS_object.power_supply_mesh_map[idx_power_supply_partition][i, 2] == idx_power_supply_mesh:
                    user_ids.append(int(user_ids_of_partition[i]))

            # 给供电单元赋值
            print("正在生成供电单元。。。")
            GIS_object.power_supply_units[idx_power_supply_partition].append([])
            # 先统计一共有几种供电单元
            for user_id in user_ids:
                feeder = users[user_id][9]
                area = GIS_object.power_supply_areas[int(GIS_object.power_supply_area_map[user_id, 2])]
                unit = feeder + f"-{area}"
                if unit not in GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh]:
                    GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh].append(unit)
            # 供电单元
            GIS_object.power_supply_unit_map[idx_power_supply_partition].append(np.zeros((len(user_ids), 4)))
            for user_id in user_ids:
                x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(users[user_id][4], users[user_id][5])
                GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][user_ids.index(user_id), 0] = x
                GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][user_ids.index(user_id), 1] = y
                GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][user_ids.index(user_id), 3] = user_id

                feeder = users[user_id][9]
                area = GIS_object.power_supply_areas[int(GIS_object.power_supply_area_map[user_id, 2])]
                unit = feeder + f"-{area}"
                GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][user_ids.index(user_id), 2] = GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh].index(unit)
    print_power_supply_unit_map(GIS_object)


    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_variable(GIS_object, f"power_supply_unit_optimization_{date}.gisobj")
