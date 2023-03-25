# 划分原则：求取负荷点的供电变电站，相同供电变电站的区域作为一个供电网格

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


def print_power_supply_mesh_map(GIS_object):
    cnt = 0
    x = []
    y = []
    power_supply_mesh = []
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        if idx_power_supply_partition >= len(GIS_object.power_supply_meshes):
            break
        if idx_power_supply_partition == 0:
            x = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 0]
            y = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 1]
            power_supply_mesh = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 2] + cnt
        else:
            x = np.concatenate((x, GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 0]))
            y = np.concatenate((y, GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 1]))
            power_supply_mesh = np.concatenate((power_supply_mesh, GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 2] + cnt))
        cnt += len(GIS_object.power_supply_meshes[idx_power_supply_partition])
    plt.scatter(x, y, s=1, c=power_supply_mesh, cmap="viridis")
    plt.colorbar()
    plt.show()

def L1_distance(matrix):
    return np.sum(np.fabs(matrix))


if __name__ == '__main__':
    GIS_object = load_variable("power_supply_partition_optimization_20230322_194303.gisobj")

    # 读取用户数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    users = cur.fetchall()

    # 读取所有高压变电站数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "高压变电站数据"''')
    conn.commit()
    HV_stations = cur.fetchall()

    # 生成距离矩阵
    # distance_matrix = np.zeros((len(users), len(HV_stations)))
    # for (user_id, user) in enumerate(users):
    #     print(user_id)
    #     user_x, user_y = GIS_object.CRS_transformer.gisxy2mercator.transform(user[4], user[5])
    #     for (HV_station_id, HV_station) in enumerate(HV_stations):
    #         HV_station_x, HV_station_y = GIS_object.CRS_transformer.gisxy2mercator.transform(HV_station[4], HV_station[5])
    #         distance_matrix[user_id, HV_station_id] = L1_distance(np.array([user_x - HV_station_x, user_y - HV_station_y]))
    # save_variable(distance_matrix, "distance_matrix.np")
    distance_matrix = load_variable("distance_matrix.np")

    # 获取每个用户和高压变电站的连接关系
    # 先删除供电所节点
    # graph = GIS_object.graph.copy()
    # for TS in GIS_object.TS_areas:
    #     if TS != "None":
    #         node = GIS_object.node_name_list.index(TS)
    #         graph.remove_node(node)
    # start_time = time.time()
    # HV_station_of_user = np.zeros((len(users), len(HV_stations)))
    # for (user_id, user) in enumerate(users):
    #     print(f"{user_id} 剩余时间{(time.time() - start_time) / (user_id +1) * (len(users) - user_id - 1)}s")
    #     start_node = GIS_object.node_name_list.index(user[1])
    #     for (HV_station_id, HV_station) in enumerate(HV_stations):
    #         if user[3] == "industrial" and HV_station_id >= 6:
    #             continue
    #         if user[3] == "urban-suburban" and HV_station_id <= 9:
    #             continue
    #         if user[3] == "rural" and (HV_station_id <= 5 or HV_station_id >= 10):
    #             continue
    #         end_node = GIS_object.node_name_list.index(HV_station[1])
    #         if nx.has_path(graph, start_node, end_node):
    #             HV_station_of_user[user_id, HV_station_id] = 1
    # save_variable(HV_station_of_user, "HV_station_of_user.np")
    HV_station_of_user = load_variable("HV_station_of_user.np")

    GIS_object.power_supply_meshes = []
    GIS_object.power_supply_mesh_map = []
    # 对每个供电分区分别进行讨论
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        # 获取属于这个分区的用户id
        user_ids = []
        for (user_id, user) in enumerate(users):
            if GIS_object.power_supply_partition_map[user_id, 2] == idx_power_supply_partition:
                user_ids.append(user_id)

        # 获取属于这个分区的高压变电站id
        HV_station_ids = []
        for (HV_station_id, HV_station) in enumerate(HV_stations):
            nearest_user_id = 0
            nearest_distance = 999999999
            for (user_id, user) in enumerate(users):
                if distance_matrix[user_id, HV_station_id] < nearest_distance:
                    nearest_distance = distance_matrix[user_id, HV_station_id]
                    nearest_user_id = user_id
            if GIS_object.power_supply_partition_map[nearest_user_id, 2] == idx_power_supply_partition:
                HV_station_ids.append(HV_station_id)

        # 给供电网格赋值
        print("正在生成供电网格。。。")
        GIS_object.power_supply_meshes.append([])
        # 先统计一共有几种供电网格
        for user_id in user_ids:
            HV_station = []
            for HV_station_id in HV_station_ids:
                if HV_station_of_user[user_id, HV_station_id] == 1:
                    HV_station.append(HV_stations[HV_station_id][1])
            area = GIS_object.power_supply_areas[int(GIS_object.power_supply_area_map[user_id, 2])]
            # mesh = "-".join(HV_station) + f"-{area}"
            mesh = "-".join(HV_station)
            if mesh not in GIS_object.power_supply_meshes[idx_power_supply_partition]:
                GIS_object.power_supply_meshes[idx_power_supply_partition].append(mesh)
        # 供电网格
        GIS_object.power_supply_mesh_map.append(np.zeros((len(user_ids), 4)))
        for user_id in user_ids:
            x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(users[user_id][4], users[user_id][5])
            GIS_object.power_supply_mesh_map[idx_power_supply_partition][user_ids.index(user_id), 0] = x
            GIS_object.power_supply_mesh_map[idx_power_supply_partition][user_ids.index(user_id), 1] = y
            GIS_object.power_supply_mesh_map[idx_power_supply_partition][user_ids.index(user_id), 3] = user_id

            HV_station = []
            for HV_station_id in HV_station_ids:
                if HV_station_of_user[user_id, HV_station_id] == 1:
                    HV_station.append(HV_stations[HV_station_id][1])
            area = GIS_object.power_supply_areas[int(GIS_object.power_supply_area_map[user_id, 2])]
            # mesh = "-".join(HV_station) + f"-{area}"
            mesh = "-".join(HV_station)
            GIS_object.power_supply_mesh_map[idx_power_supply_partition][user_ids.index(user_id), 2] = GIS_object.power_supply_meshes[idx_power_supply_partition].index(mesh)
    print_power_supply_mesh_map(GIS_object)


    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_variable(GIS_object, f"power_supply_mesh_optimization_{date}.gisobj")
