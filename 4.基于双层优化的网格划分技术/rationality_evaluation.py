# 配电网网格划分合理性评估体系
# A：供电分区
#   A1：供电分区管理责任独立性（=Σwi*fuzzy(xi)，其中xi=供电分区i的管理者个数，fuzzy满足xi==1时最佳，wi权重由供电区域等级决定）
#   A2：供电分区行政规划独立性（=Σwi*fuzzy(xi)，其中xi=供电分区i的行政规划个数，fuzzy满足xi==1时最佳，wi权重由供电区域等级决定）
#   A3：供电分区负荷峰值（=Σwi*fuzzy(xi)，其中xi=供电分区i的负荷峰值MW，fuzzy满足xi<=1000时最佳，wi权重由供电区域等级决定）
#   get_weight_for_index_by_AHP(np.array([[1, 2, 1], [1/2, 1, 1], [1, 1, 1]]))
# B：供电网格
#   B1：供电网格行政规划独立性（=Σwi*fuzzy(xi)，其中xi=供电网格i的行政规划个数，fuzzy满足xi==1时最佳，wi权重由供电区域等级决定）
#   B2：供电网格供电范围独立性（=Σwi*fuzzy(xi)，其中xi=该分区与供电网格i存在供电范围重叠的网格个数/该分区所有供电网格数目，fuzzy满足xi==0时最佳，wi权重由供电区域等级决定）
#   B3：供电网格供电区域独立性（=Σwi*fuzzy(xi)，其中xi=供电网格i所包含的供电区域等级个数，fuzzy满足xi==1时最佳，wi权重由供电区域等级决定）
#   B4：供电网格上级变电站数目（=Σwi*fuzzy(xi)，其中xi=供电网格i上级变电站个数，fuzzy满足xi==2~4时最佳，wi权重由供电区域等级决定）
#   B5：供电网格10kV线路数目（=Σwi*fuzzy(xi)，其中xi=供电网格i的10kV线路个数，fuzzy满足xi<=20时最佳，wi权重由供电区域等级决定）
#   B6：供电网格上级变电站联络率（=Σwi*fuzzy(xi)，其中xi=供电网格i中连接多个上级变电站的负荷点/该网格所有负荷点数目，fuzzy满足xi==1时最佳，wi权重由供电区域等级决定）
#   get_weight_for_index_by_AHP(np.array([[1, 1/2, 1/3, 1/3, 1/3, 1/4], [2, 1, 1/2, 1/2, 1/3, 1/3], [3, 2, 1, 1, 1, 1/2], [3, 2, 1, 1, 2, 1/2], [3, 3, 1, 1/2, 1, 1/2], [4, 3, 2, 2, 2, 1]]))
# C：供电单元
#   C1：供电单元供电范围独立性（=Σwi*fuzzy(xi)，其中xi=该网格与供电单元i存在供电范围重叠的单元个数/该网格所有供电单元数目，fuzzy满足xi==0时最佳，wi权重由供电区域等级决定）
#   C2：供电单元供电区域独立性（=Σwi*fuzzy(xi)，其中xi=供电单元i所包含的供电区域等级个数，fuzzy满足xi==1时最佳，wi权重由供电区域等级决定）
#   C3：供电单元负荷互补性（=Σwi*fuzzy(xi)，其中xi=供电单元i的年负荷率，fuzzy满足xi==1时最佳，wi权重由供电区域等级决定）
#   C4：供电单元10kV线路数目（=Σwi*fuzzy(xi)，其中xi=供电单元i的10kV线路个数，fuzzy满足xi<=4时最佳，wi权重由供电区域等级决定）
#   C5：供电单元上级变电站数目（=Σwi*fuzzy(xi)，其中xi=供电单元i上级变电站个数，fuzzy满足xi==2时最佳，wi权重由供电区域等级决定）
#   get_weight_for_index_by_AHP(np.array([[1, 1, 1/2, 1/2, 1/2], [1, 1, 1/2, 2, 1], [2, 2, 1, 2, 1], [2, 1/2, 1/2, 1, 1], [2, 1, 1, 1, 1]]))


# 权重确定方法：层次分析法+TOPSIS法

import numpy as np
import math
import time
from GIS_object import get_block_index
import sqlite3
import matplotlib.pyplot as plt
import datetime
import networkx as nx
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import save_variable, load_variable


# 评价体系
class Evaluator:
    def __init__(self, GIS_object):
        self.GIS_object = GIS_object

    def cal_partition_index(self, partition_index_config):
        weight = partition_index_config[0]
        fuzzy_parameter = partition_index_config[1]
        index_function = partition_index_config[2]
        result = 0
        for (idx_power_supply_partition, power_supply_partition) in enumerate(self.GIS_object.power_supply_partitions):
            result += weight[idx_power_supply_partition]\
                      * fuzzy_function(index_function(self.GIS_object, idx_power_supply_partition),
                                                 fuzzy_parameter[0],
                                                 fuzzy_parameter[1],
                                                 fuzzy_parameter[2],
                                                 fuzzy_parameter[3])
        return result

    def cal_mesh_index(self, idx_power_supply_partition, mesh_index_config):
        weight = mesh_index_config[0]
        fuzzy_parameter = mesh_index_config[1]
        index_function = mesh_index_config[2]
        result = 0
        for (idx_power_supply_mesh, power_supply_mesh) in enumerate(self.GIS_object.power_supply_meshes[idx_power_supply_partition]):
            result += weight[idx_power_supply_mesh] \
                      * fuzzy_function(index_function(self.GIS_object, idx_power_supply_partition, idx_power_supply_mesh),
                                       fuzzy_parameter[0],
                                       fuzzy_parameter[1],
                                       fuzzy_parameter[2],
                                       fuzzy_parameter[3])
        return result

    def cal_unit_index(self, idx_power_supply_partition, idx_power_supply_mesh, unit_index_config):
        weight = unit_index_config[0]
        fuzzy_parameter = unit_index_config[1]
        index_function = unit_index_config[2]
        result = 0
        for (idx_power_supply_unit, power_supply_unit) in enumerate(self.GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh]):
            result += weight[idx_power_supply_unit] \
                      * fuzzy_function(index_function(self.GIS_object, idx_power_supply_partition, idx_power_supply_mesh, idx_power_supply_unit),
                                       fuzzy_parameter[0],
                                       fuzzy_parameter[1],
                                       fuzzy_parameter[2],
                                       fuzzy_parameter[3])
        return result


def get_weight_for_index_by_AHP(decision_matrix):
    N = np.size(decision_matrix, 0)
    weight = np.zeros(N)
    # 先进行一致性检验
    w, v = np.linalg.eig(decision_matrix)
    lambda_max = np.max(w)
    CI = (lambda_max - N) / (N - 1)
    RI_group = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
    CR = CI / RI_group[N - 1]
    if CR >= 0.1:
        print("判断矩阵未通过一致性检验！请重新确定判断矩阵")
        return weight
    else:
        print("判断矩阵通过一致性检验")
        # 算术平均法求权重
        weight_arithmetic_mean = decision_matrix / np.tile(np.sum(decision_matrix, axis=0), (N, 1))
        weight_arithmetic_mean = np.sum(weight_arithmetic_mean, axis=1) / N
        # 几何平均法求权重
        weight_geometric_mean = 1
        for i in range(N):
            weight_geometric_mean *= decision_matrix[:, i]
        weight_geometric_mean = np.power(weight_geometric_mean, 1/N)
        weight_geometric_mean /= np.sum(weight_geometric_mean)
        # # 特征值法求权重
        # weight_lambda = v[np.where(w == lambda_max)[0][0]]
        # weight_lambda /= np.sum(weight_lambda)
        # # 将三种方法进行平均
        # weight = (weight_arithmetic_mean + weight_geometric_mean + weight_lambda) / 3
        # 将两种方法进行平均
        weight = (weight_arithmetic_mean + weight_geometric_mean) / 2
        return weight


def get_weight_for_index_by_Entropy(index_samples):
    N = np.size(index_samples, 1)
    sample_num = np.size(index_samples, 0)
    weight = np.zeros(N)
    if sample_num <= 1:
        print("样本数不足，无法进行熵权法求权重！")
        return weight
    else:
        # 首先进行标准化
        min_index_samples = np.tile(np.min(index_samples, axis=0), (sample_num, 1))
        max_index_samples = np.tile(np.max(index_samples, axis=0), (sample_num, 1))
        index_samples = (index_samples - min_index_samples) / (max_index_samples - min_index_samples)
        # 然后进行非负平移
        min_index_samples = np.tile(np.min(index_samples, axis=0), (sample_num, 1))
        index_samples = index_samples + (np.absolute(min_index_samples) + 0.01)
        # 用熵权法确认权重
        # 先归一化
        index_samples /= np.tile(np.sum(index_samples, axis=0), (sample_num, 1))
        # 再求取信息熵
        e = -1/math.log(sample_num) * np.sum(index_samples * np.log(index_samples), axis=0)
        # 最后计算权重
        weight = (1 - e) / np.sum(1 - e)
        return weight


def get_score_by_TOPSIS(weight, index_sample):
    N = np.size(weight)
    zmin =np.zeros(N)
    zmax =np.ones(N)
    z = index_sample
    dmin = np.power(np.sum(np.power(z - zmin, 2) * weight), 0.5)
    dmax = np.power(np.sum(np.power(z - zmax, 2) * weight), 0.5)
    score = dmin / (dmax + dmin)
    return score


def get_weight_for_partition(GIS_object):
    weight = np.zeros(len(GIS_object.power_supply_partitions))
    # 读取用户数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    users = cur.fetchall()
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        # 获取属于这个分区的用户id
        user_ids = []
        for (user_id, user) in enumerate(users):
            if GIS_object.power_supply_partition_map[user_id, 2] == idx_power_supply_partition:
                user_ids.append(user_id)
        for user_id in user_ids:
            idx_power_supply_area = int(GIS_object.power_supply_area_map[user_id, 2])
            if GIS_object.power_supply_areas[idx_power_supply_area] == "A_plus":
                weight[idx_power_supply_partition] += 6
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "A":
                weight[idx_power_supply_partition] += 5
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "B":
                weight[idx_power_supply_partition] += 4
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "C":
                weight[idx_power_supply_partition] += 3
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "D":
                weight[idx_power_supply_partition] += 2
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "E":
                weight[idx_power_supply_partition] += 1
    weight /= np.sum(weight)
    return weight


def get_weight_for_mesh(GIS_object, idx_power_supply_partition):
    weight = np.zeros(len(GIS_object.power_supply_meshes[idx_power_supply_partition]))
    for (idx_power_supply_mesh, power_supply_mesh) in enumerate(GIS_object.power_supply_meshes[idx_power_supply_partition]):
        # 获取属于这个网格的用户id
        user_ids_of_partition = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 3].tolist()
        user_ids = []
        for i in range(len(user_ids_of_partition)):
            if GIS_object.power_supply_mesh_map[idx_power_supply_partition][i, 2] == idx_power_supply_mesh:
                user_ids.append(int(user_ids_of_partition[i]))
        for user_id in user_ids:
            idx_power_supply_area = int(GIS_object.power_supply_area_map[user_id, 2])
            if GIS_object.power_supply_areas[idx_power_supply_area] == "A_plus":
                weight[idx_power_supply_mesh] += 6
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "A":
                weight[idx_power_supply_mesh] += 5
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "B":
                weight[idx_power_supply_mesh] += 4
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "C":
                weight[idx_power_supply_mesh] += 3
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "D":
                weight[idx_power_supply_mesh] += 2
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "E":
                weight[idx_power_supply_mesh] += 1
    weight /= np.sum(weight)
    return weight


def get_weight_for_unit(GIS_object, idx_power_supply_partition, idx_power_supply_mesh):
    weight = np.zeros(len(GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh]))
    for (idx_power_supply_unit, power_supply_unit) in enumerate(GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh]):
        # 获取属于这个单元的用户id
        user_ids_of_mesh = GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 3].tolist()
        user_ids = []
        for i in range(len(user_ids_of_mesh)):
            if GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][i, 2] == idx_power_supply_unit:
                user_ids.append(int(user_ids_of_mesh[i]))
        for user_id in user_ids:
            idx_power_supply_area = int(GIS_object.power_supply_area_map[user_id, 2])
            if GIS_object.power_supply_areas[idx_power_supply_area] == "A_plus":
                weight[idx_power_supply_unit] += 6
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "A":
                weight[idx_power_supply_unit] += 5
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "B":
                weight[idx_power_supply_unit] += 4
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "C":
                weight[idx_power_supply_unit] += 3
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "D":
                weight[idx_power_supply_unit] += 2
            elif GIS_object.power_supply_areas[idx_power_supply_area] == "E":
                weight[idx_power_supply_unit] += 1
    weight /= np.sum(weight)
    return weight


# 计算模糊函数的隶属度
def fuzzy_function(x, a, b, c, d):
    if x < a or x > d:
        return 0
    elif a == x and x != b:
        return 0
    elif a == x == b:
        return 1
    elif a < x < b:
        return (x - a) / (b - a)
    elif a != x and x == b:
        return 1
    elif b < x < c:
        return 1
    elif d != x and x == c:
        return 1
    elif c < x < d:
        return (x - c) / (d - c)
    elif c == x == d:
        return 1
    elif d == x and x != c:
        return 0


# 供电分区i的管理者数目
def get_TS_num_of_power_supply_partition(GIS_object, idx_power_supply_partition):
    # 读取用户数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    users = cur.fetchall()
    # 读取用户和供电所连接关系
    path_relationship = load_variable("path_relationship.np")
    # 获取属于这个分区的用户id
    user_ids = []
    for (user_id, user) in enumerate(users):
        if GIS_object.power_supply_partition_map[user_id, 2] == idx_power_supply_partition:
            user_ids.append(user_id)
    # 统计供电分区的管理者个数
    TSs = []
    for user_id in user_ids:
        for (idx_TS, TS) in enumerate(GIS_object.TS_areas_unit):
            if path_relationship[user_id, idx_TS] == 1 and TS not in TSs:
                TSs.append(TS)
    manager_num = [0, 0, 0]
    for TS in TSs:
        if TS.find("I") != -1:
            manager_num[0] = 1
        if TS.find("R") != -1:
            manager_num[1] = 1
        if TS.find("U") != -1:
            manager_num[2] = 1
    return sum(manager_num)


# 供电分区i的行政规划数目
def get_administrator_num_of_power_supply_partition(GIS_object, idx_power_supply_partition):
    # 读取用户数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    users = cur.fetchall()
    # 获取属于这个分区的用户id
    user_ids = []
    for (user_id, user) in enumerate(users):
        if GIS_object.power_supply_partition_map[user_id, 2] == idx_power_supply_partition:
            user_ids.append(user_id)
    # 统计供电分区的行政规划个数
    administrators = []
    for user_id in user_ids:
        if users[user_id][3] not in administrators:
            administrators.append(users[user_id][3])
    return len(administrators)


# 供电分区i的负荷峰值
def get_max_load_of_power_supply_partition(GIS_object, idx_power_supply_partition):
    # 读取用户数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    users = cur.fetchall()
    # 获取属于这个分区的用户id
    user_ids = []
    for (user_id, user) in enumerate(users):
        if GIS_object.power_supply_partition_map[user_id, 2] == idx_power_supply_partition:
            user_ids.append(user_id)
    # 计算负荷曲线
    load_profile = np.zeros(12)
    for user_id in user_ids:
        load_profile += GIS_object.user_load_profile[user_id, :]
    return np.max(load_profile)


# 供电网格i的行政规划数目
def get_administrator_num_of_power_supply_mesh(GIS_object, idx_power_supply_partition, idx_power_supply_mesh):
    # 读取用户数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    users = cur.fetchall()
    # 获取属于这个网格的用户id
    user_ids_of_partition = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 3].tolist()
    user_ids = []
    for i in range(len(user_ids_of_partition)):
        if GIS_object.power_supply_mesh_map[idx_power_supply_partition][i, 2] == idx_power_supply_mesh:
            user_ids.append(int(user_ids_of_partition[i]))
    # 统计供电网格的行政规划个数
    administrators = []
    for user_id in user_ids:
        if users[user_id][3] not in administrators:
            administrators.append(users[user_id][3])
    return len(administrators)


# 供电网格i的供电范围独立性
def get_power_supply_independence_of_power_supply_mesh(GIS_object, idx_power_supply_partition, idx_power_supply_mesh):
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
    # 删除graph中的所有高压变电站
    GIS_object.get_network()
    graph = GIS_object.graph.copy()
    for HV_station in HV_stations:
        node_id = GIS_object.node_name_list.index(HV_station[1])
        graph.remove_node(node_id)
    # 每个网格选出一个负荷节点作为代表
    node_representation_of_mesh = []
    for (_idx_power_supply_mesh, power_supply_mesh) in enumerate(GIS_object.power_supply_meshes[idx_power_supply_partition]):
        # 获取一个这个网格的用户id
        user_ids_of_partition = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 3].tolist()
        user_id = 0
        for i in range(len(user_ids_of_partition)):
            if GIS_object.power_supply_mesh_map[idx_power_supply_partition][i, 2] == _idx_power_supply_mesh:
                user_id = int(user_ids_of_partition[i])
                break
        node_representation_of_mesh.append(GIS_object.node_name_list.index(users[user_id][1]))
    # 获取连接关系
    num_of_connected_meshes = 0
    for (_idx_power_supply_mesh, power_supply_mesh) in enumerate(GIS_object.power_supply_meshes[idx_power_supply_partition]):
        if _idx_power_supply_mesh != idx_power_supply_mesh:
            if nx.has_path(graph, node_representation_of_mesh[idx_power_supply_mesh], node_representation_of_mesh[_idx_power_supply_mesh]):
                num_of_connected_meshes += 1
    return num_of_connected_meshes / (len(GIS_object.power_supply_meshes[idx_power_supply_partition]))


# 供电网格i的供电区域等级数目
def get_power_supply_area_num_of_power_supply_mesh(GIS_object, idx_power_supply_partition, idx_power_supply_mesh):
    # 获取属于这个网格的用户id
    user_ids_of_partition = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 3].tolist()
    user_ids = []
    for i in range(len(user_ids_of_partition)):
        if GIS_object.power_supply_mesh_map[idx_power_supply_partition][i, 2] == idx_power_supply_mesh:
            user_ids.append(int(user_ids_of_partition[i]))
    # 统计供电网格的供电区域等级个数
    areas = []
    for user_id in user_ids:
        if GIS_object.power_supply_areas[int(GIS_object.power_supply_area_map[user_id, 2])] not in areas:
            areas.append(GIS_object.power_supply_areas[int(GIS_object.power_supply_area_map[user_id, 2])])
    return len(areas)


# 供电网格i的上级变电站数目
def get_HV_station_num_of_power_supply_mesh(GIS_object, idx_power_supply_partition, idx_power_supply_mesh):
    # 读取所有高压变电站数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "高压变电站数据"''')
    conn.commit()
    HV_stations = cur.fetchall()
    HV_station_of_user = load_variable("HV_station_of_user.np")
    # 获取属于这个网格的用户id
    user_ids_of_partition = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 3].tolist()
    user_ids = []
    for i in range(len(user_ids_of_partition)):
        if GIS_object.power_supply_mesh_map[idx_power_supply_partition][i, 2] == idx_power_supply_mesh:
            user_ids.append(int(user_ids_of_partition[i]))
    # 统计供电网格的上级变电站个数
    HV_stations_of_mesh = []
    for user_id in user_ids:
        for (HV_station_id, HV_station) in enumerate(HV_stations):
            if HV_station_of_user[user_id, HV_station_id] == 1 and HV_stations[HV_station_id][1] not in HV_stations_of_mesh:
                HV_stations_of_mesh.append(HV_stations[HV_station_id][1])
    return len(HV_stations_of_mesh)


# 供电网格i的10kV线路个数
def get_LV_line_num_of_power_supply_mesh(GIS_object, idx_power_supply_partition, idx_power_supply_mesh):
    # 读取用户数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    users = cur.fetchall()
    # 获取属于这个网格的用户id
    user_ids_of_partition = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 3].tolist()
    user_ids = []
    for i in range(len(user_ids_of_partition)):
        if GIS_object.power_supply_mesh_map[idx_power_supply_partition][i, 2] == idx_power_supply_mesh:
            user_ids.append(int(user_ids_of_partition[i]))
    # 统计供电网格的馈线个数
    feeders = []
    for user_id in user_ids:
        if users[user_id][9] not in feeders:
            feeders.append(users[user_id][9])
    return len(feeders)


# 供电网格i的上级变电站联络率
def get_contact_rate_of_power_supply_mesh(GIS_object, idx_power_supply_partition, idx_power_supply_mesh):
    # 读取所有高压变电站数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "高压变电站数据"''')
    conn.commit()
    HV_stations = cur.fetchall()
    HV_station_of_user = load_variable("HV_station_of_user.np")
    # 获取属于这个网格的用户id
    user_ids_of_partition = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 3].tolist()
    user_ids = []
    for i in range(len(user_ids_of_partition)):
        if GIS_object.power_supply_mesh_map[idx_power_supply_partition][i, 2] == idx_power_supply_mesh:
            user_ids.append(int(user_ids_of_partition[i]))
    # 统计连接了多个上级变电站的用户数目
    user_num = 0
    for user_id in user_ids:
        HV_station_num = 0
        for (HV_station_id, HV_station) in enumerate(HV_stations):
            if HV_station_of_user[user_id, HV_station_id] == 1:
                HV_station_num += 1
        if HV_station_num >= 2:
            user_num += 1
    return user_num / len(user_ids)


# 供电单元i的供电范围独立性
def get_power_supply_independence_of_power_supply_unit(GIS_object, idx_power_supply_partition, idx_power_supply_mesh, idx_power_supply_unit):
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
    # 删除graph中的所有高压变电站
    GIS_object.get_network()
    graph = GIS_object.graph.copy()
    for HV_station in HV_stations:
        node_id = GIS_object.node_name_list.index(HV_station[1])
        graph.remove_node(node_id)
    # 每个单元选出一个负荷节点作为代表
    node_representation_of_unit = []
    for (_idx_power_supply_unit, power_supply_unit) in enumerate(GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh]):
        # 获取一个这个单元的用户id
        user_ids_of_mesh = GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 3].tolist()
        user_id = 0
        for i in range(len(user_ids_of_mesh)):
            if GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][i, 2] == _idx_power_supply_unit:
                user_id = int(user_ids_of_mesh[i])
                break
        node_representation_of_unit.append(GIS_object.node_name_list.index(users[user_id][1]))
    # 获取连接关系
    num_of_connected_units = 0
    for (_idx_power_supply_unit, power_supply_unit) in enumerate(GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh]):
        if _idx_power_supply_unit != idx_power_supply_unit:
            if nx.has_path(graph, node_representation_of_unit[idx_power_supply_unit], node_representation_of_unit[_idx_power_supply_unit]):
                num_of_connected_units += 1
    return num_of_connected_units / (len(GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh]))


# 供电单元i的年负荷率
def get_annual_load_rate_of_power_supply_unit(GIS_object, idx_power_supply_partition, idx_power_supply_mesh, idx_power_supply_unit):
    # 获取属于这个单元的用户id
    user_ids_of_mesh = GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 3].tolist()
    user_ids = []
    for i in range(len(user_ids_of_mesh)):
        if GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][i, 2] == idx_power_supply_unit:
            user_ids.append(int(user_ids_of_mesh[i]))
    # 计算负荷曲线
    load_profile = np.zeros(12)
    for user_id in user_ids:
        load_profile += GIS_object.user_load_profile[user_id, :]
    return np.average(load_profile) / np.max(load_profile)


# 供电单元i的供电区域等级数目
def get_power_supply_area_num_of_power_supply_unit(GIS_object, idx_power_supply_partition, idx_power_supply_mesh, idx_power_supply_unit):
    # 获取属于这个单元的用户id
    user_ids_of_mesh = GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 3].tolist()
    user_ids = []
    for i in range(len(user_ids_of_mesh)):
        if GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][i, 2] == idx_power_supply_unit:
            user_ids.append(int(user_ids_of_mesh[i]))
    # 统计供电单元的供电区域等级个数
    areas = []
    for user_id in user_ids:
        if GIS_object.power_supply_areas[int(GIS_object.power_supply_area_map[user_id, 2])] not in areas:
            areas.append(GIS_object.power_supply_areas[int(GIS_object.power_supply_area_map[user_id, 2])])
    return len(areas)


# 供电单元i的10kV线路个数
def get_LV_line_num_of_power_supply_unit(GIS_object, idx_power_supply_partition, idx_power_supply_mesh, idx_power_supply_unit):
    # 读取用户数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    users = cur.fetchall()
    # 获取属于这个单元的用户id
    user_ids_of_mesh = GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 3].tolist()
    user_ids = []
    for i in range(len(user_ids_of_mesh)):
        if GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][i, 2] == idx_power_supply_unit:
            user_ids.append(int(user_ids_of_mesh[i]))
    # 统计供电网格的馈线个数
    feeders = []
    for user_id in user_ids:
        if users[user_id][9] not in feeders:
            feeders.append(users[user_id][9])
    return len(feeders)


# 供电网格i的上级变电站数目
def get_HV_station_num_of_power_supply_unit(GIS_object, idx_power_supply_partition, idx_power_supply_mesh, idx_power_supply_unit):
    # 读取所有高压变电站数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "高压变电站数据"''')
    conn.commit()
    HV_stations = cur.fetchall()
    HV_station_of_user = load_variable("HV_station_of_user.np")
    # 获取属于这个单元的用户id
    user_ids_of_mesh = GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 3].tolist()
    user_ids = []
    for i in range(len(user_ids_of_mesh)):
        if GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][i, 2] == idx_power_supply_unit:
            user_ids.append(int(user_ids_of_mesh[i]))
    # 统计供电单元的上级变电站个数
    HV_stations_of_unit = []
    for user_id in user_ids:
        for (HV_station_id, HV_station) in enumerate(HV_stations):
            if HV_station_of_user[user_id, HV_station_id] == 1 and HV_stations[HV_station_id][1] not in HV_stations_of_unit:
                HV_stations_of_unit.append(HV_stations[HV_station_id][1])
    return len(HV_stations_of_unit)


if __name__ == '__main__':
    GIS_object = load_variable("power_supply_unit_optimization_20230322_195214.gisobj")
    evaluator = Evaluator(GIS_object)
    # 计算供电分区的指标
    partition_weight = get_weight_for_partition(GIS_object)
    A1_config = [partition_weight, (0, 1, 1, 3), get_TS_num_of_power_supply_partition]
    A2_config = [partition_weight, (0, 1, 1, 3), get_administrator_num_of_power_supply_partition]
    A3_config = [partition_weight, (0, 0, 1000, 1000), get_max_load_of_power_supply_partition]
    a1 = evaluator.cal_partition_index(A1_config)
    a2 = evaluator.cal_partition_index(A2_config)
    a3 = evaluator.cal_partition_index(A3_config)
    # 计算供电分区的得分
    decision_matrix_of_partition_index = np.array([[1, 2, 1], [1/2, 1, 1], [1, 1, 1]])
    weight_of_partition_index = get_weight_for_index_by_AHP(decision_matrix_of_partition_index)
    score_of_partition = get_score_by_TOPSIS(weight=weight_of_partition_index, index_sample=np.array([a1, a2, a3]))
    print(f"供电分区划分合理性得分为{score_of_partition*100}分")

    # 计算供电网格的指标
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    b5 = []
    b6 = []
    index_samples_of_mesh = np.zeros((len(GIS_object.power_supply_partitions), 6))
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        mesh_weight = get_weight_for_mesh(GIS_object, idx_power_supply_partition)
        B1_config = [mesh_weight, (0, 1, 1, 3), get_administrator_num_of_power_supply_mesh]
        B2_config = [mesh_weight, (0, 0, 0, 1), get_power_supply_independence_of_power_supply_mesh]
        B3_config = [mesh_weight, (0, 1, 1, 3), get_power_supply_area_num_of_power_supply_mesh]
        B4_config = [mesh_weight, (2, 2, 4, 6), get_HV_station_num_of_power_supply_mesh]
        B5_config = [mesh_weight, (0, 0, 20, 25), get_LV_line_num_of_power_supply_mesh]
        B6_config = [mesh_weight, (0, 1, 1, 1), get_contact_rate_of_power_supply_mesh]
        b1.append(evaluator.cal_mesh_index(idx_power_supply_partition, B1_config))
        b2.append(evaluator.cal_mesh_index(idx_power_supply_partition, B2_config))
        b3.append(evaluator.cal_mesh_index(idx_power_supply_partition, B3_config))
        b4.append(evaluator.cal_mesh_index(idx_power_supply_partition, B4_config))
        b5.append(evaluator.cal_mesh_index(idx_power_supply_partition, B5_config))
        b6.append(evaluator.cal_mesh_index(idx_power_supply_partition, B6_config))
        index_samples_of_mesh[idx_power_supply_partition, :] = np.array([b1[idx_power_supply_partition],
                                                                         b2[idx_power_supply_partition],
                                                                         b3[idx_power_supply_partition],
                                                                         b4[idx_power_supply_partition],
                                                                         b5[idx_power_supply_partition],
                                                                         b6[idx_power_supply_partition]])
    # 计算供电网格的得分
    decision_matrix_of_mesh_index = np.array([[1, 1/2, 1/3, 1/3, 1/3, 1/4], [2, 1, 1/2, 1/2, 1/3, 1/3], [3, 2, 1, 1, 1, 1/2], [3, 2, 1, 1, 2, 1/2], [3, 3, 1, 1/2, 1, 1/2], [4, 3, 2, 2, 2, 1]])
    weight_of_mesh_index = get_weight_for_index_by_AHP(decision_matrix_of_mesh_index) * get_weight_for_index_by_Entropy(index_samples_of_mesh)
    weight_of_mesh_index /= np.sum(weight_of_mesh_index)
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        score_of_mesh = get_score_by_TOPSIS(weight=weight_of_mesh_index, index_sample=np.array([b1[idx_power_supply_partition],
                                                                                                b2[idx_power_supply_partition],
                                                                                                b3[idx_power_supply_partition],
                                                                                                b4[idx_power_supply_partition],
                                                                                                b5[idx_power_supply_partition],
                                                                                                b6[idx_power_supply_partition]]))
        print(f"供电分区{power_supply_partition}的网格划分合理性得分为{score_of_mesh*100}分")

    # 计算供电单元的指标
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    cnt = 0
    index_samples_of_unit = np.zeros((len(GIS_object.power_supply_partitions)*len(GIS_object.power_supply_meshes[idx_power_supply_partition]), 5))
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        for (idx_power_supply_mesh, power_supply_mesh) in enumerate(GIS_object.power_supply_meshes[idx_power_supply_partition]):
            unit_weight = get_weight_for_unit(GIS_object, idx_power_supply_partition, idx_power_supply_mesh)
            C1_config = [unit_weight, (0, 0, 0, 1), get_power_supply_independence_of_power_supply_unit]
            C2_config = [unit_weight, (0, 1, 1, 3), get_power_supply_area_num_of_power_supply_unit]
            C3_config = [unit_weight, (0, 1, 1, 1), get_annual_load_rate_of_power_supply_unit]
            C4_config = [unit_weight, (0, 0, 4, 8), get_LV_line_num_of_power_supply_unit]
            C5_config = [unit_weight, (0, 2, 2, 4), get_HV_station_num_of_power_supply_unit]
            c1.append(evaluator.cal_unit_index(idx_power_supply_partition, idx_power_supply_mesh, C1_config))
            c2.append(evaluator.cal_unit_index(idx_power_supply_partition, idx_power_supply_mesh, C2_config))
            c3.append(evaluator.cal_unit_index(idx_power_supply_partition, idx_power_supply_mesh, C3_config))
            c4.append(evaluator.cal_unit_index(idx_power_supply_partition, idx_power_supply_mesh, C4_config))
            c5.append(evaluator.cal_unit_index(idx_power_supply_partition, idx_power_supply_mesh, C5_config))
            index_samples_of_unit[cnt, :] =\
                np.array([c1[cnt],
                          c2[cnt],
                          c3[cnt],
                          c4[cnt],
                          c5[cnt]])
            cnt += 1
    # 计算供电单元的得分
    decision_matrix_of_unit_index = np.array([[1, 1, 1/2, 1/2, 1/2], [1, 1, 1/2, 2, 1], [2, 2, 1, 2, 1], [2, 1/2, 1/2, 1, 1], [2, 1, 1, 1, 1]])
    weight_of_unit_index = get_weight_for_index_by_AHP(decision_matrix_of_unit_index) * get_weight_for_index_by_Entropy(index_samples_of_unit)
    weight_of_unit_index /= np.sum(weight_of_unit_index)
    cnt = 0
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        for (idx_power_supply_mesh, power_supply_mesh) in enumerate(GIS_object.power_supply_meshes[idx_power_supply_partition]):
            score_of_unit = get_score_by_TOPSIS(weight=weight_of_unit_index, index_sample=np.array([c1[cnt],
                                                                                                    c2[cnt],
                                                                                                    c3[cnt],
                                                                                                    c4[cnt],
                                                                                                    c5[cnt]]))
            cnt += 1
            print(f"供电网格{power_supply_mesh}的单元划分合理性得分为{score_of_unit*100}分")
