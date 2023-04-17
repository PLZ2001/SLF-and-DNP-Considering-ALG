# 假设上级变电站选址已经确定，上级变电站容量已经足够，中压线路容量已经足够，只规划配变容量和中压线路连接关系
# 研究范围：一个供电网格，已知变电站和规划变电站的位置，各配变的位置及原始容量，各配变连接的用户点，各用户点的负荷预测值、功率因数、异常增长概率密度函数、正常增长误差概率密度函数

# 模型：配变定容+线路选线 混合整数线性规划模型
# 决策量：配变的增容量（连续变量），配变所连接的2个变电站（0-1变量）
# 优化目标：[固定费用+可变费用（与总容量有关）]*贴现公式+运维费用 + [固定费用+可变费用（与线路长度有关）]*贴现公式+运维费用（网损）
# 机会约束：P[(配变原始容量+配变增容量)*功率因素 > 负荷预测值 + 异常增长值 + 正常增长预测误差] > 90%
# 其他约束：（1）线路长度满足最短约束（2）每个单环网包含配变数目不超过100个
# 多场景生成：低异常增长场景，中异常增长场景，高异常增长场景
# 优化方法：gurobi，三种场景分别优化、三种场景按照概率求和后优化


from planning_object import PlanningObject
import numpy as np
import random
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import save_variable, load_variable
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import datetime
from itertools import permutations, combinations
import elkai
import networkx as nx


def euclidean_distance(matrix):
    return np.sqrt(np.sum(np.power(matrix, 2)))


def street_distance(x1, x2, y1, y2, nodes):
    start_node = find_nearest_street_node(x1, y1, nodes)
    end_node = find_nearest_street_node(x2, y2, nodes)
    try:
        node_list = get_shortest_street_path(planning_object, start_node, end_node)
    except:
        return 1E6
    wire_x = np.zeros(len(node_list)+2)
    wire_y = np.zeros(len(node_list)+2)
    wire_x[0] = x1
    wire_y[0] = y1
    for (idx,street_node) in enumerate(node_list):
        wire_x[idx+1] = nodes[street_node]["x"]
        wire_y[idx+1] = nodes[street_node]["y"]
    wire_x[-1] = x2
    wire_y[-1] = y2
    street_dist = 0
    for i in range(len(wire_x)):
        if i == len(wire_x) - 1:
            break
        street_dist += euclidean_distance(np.array([wire_x[i]-wire_x[i+1], wire_y[i]-wire_y[i+1]]))
    return street_dist


def random_pick(local_load, x, pdf, level = 1000):
    if local_load[0] is True:
        pick_pool = load_variable(f"{local_load[1]}.pickpool")
    else:
        # 对pdf进行整数化处理
        pdf = np.around(level * pdf)
        # 生成抽样池
        pick_pool = np.zeros(int(np.sum(pdf)))
        cnt = 0
        for x_id in range(len(x)):
            for pdf_cnt in range(int(pdf[x_id])):
                pick_pool[cnt] = x[x_id]
                cnt = cnt + 1
        save_variable(pick_pool, f"{local_load[1]}.pickpool")
    # 抽样
    pick = random.randint(0, len(pick_pool) - 1)
    return pick_pool[pick]


def get_virtual_transformer_path(L_min, L_group, planning_object, wire_cnt, HV_station_id_i, HV_station_id_j):
    all_node_distance = load_variable(f"all_node_distance_{planning_object.planning_transformer_num_virtual}VTS.np")
    # 接下来是根据L_min反推出单环网所经过的虚拟配变路径
    # 待选路径的总数目
    try_times = planning_object.planning_transformer_num_virtual
    # 生成虚拟配变远近程度
    all_virtual_transformer_distance = np.zeros(
        planning_object.planning_transformer_num_virtual)
    wire_cnt_ = 0
    for (HV_station_id_i_, HV_station_i_) in enumerate(planning_object.planning_HV_station_name):
        for (HV_station_id_j_, HV_station_j_) in enumerate(planning_object.planning_HV_station_name):
            if HV_station_id_i_ < HV_station_id_j_:
                wire_cnt_ += 1
    all_virtual_transformer_distance_matrix = np.zeros(
        (planning_object.planning_transformer_num_virtual, wire_cnt_))
    for virtual_transformer_id in range(planning_object.planning_transformer_num_virtual):
        wire_cnt_ = 0
        for (HV_station_id_i_, HV_station_i_) in enumerate(planning_object.planning_HV_station_name):
            for (HV_station_id_j_, HV_station_j_) in enumerate(
                    planning_object.planning_HV_station_name):
                if HV_station_id_i_ < HV_station_id_j_:
                    all_virtual_transformer_distance_matrix[
                        virtual_transformer_id, wire_cnt_] += all_node_distance[HV_station_id_i_, virtual_transformer_id+len(planning_object.planning_HV_station_name)]
                    all_virtual_transformer_distance_matrix[
                        virtual_transformer_id, wire_cnt_] += all_node_distance[HV_station_id_j_, virtual_transformer_id+len(planning_object.planning_HV_station_name)]
                    wire_cnt_ += 1
        all_virtual_transformer_distance[virtual_transformer_id] = \
            all_virtual_transformer_distance_matrix[virtual_transformer_id, wire_cnt] / np.sum(
                all_virtual_transformer_distance_matrix[virtual_transformer_id, :])
    # 按照从近到远对虚拟配变进行排序
    virtual_transformer_distance_sort = np.argsort(all_virtual_transformer_distance)
    # 根据L_min和L_group确定路径的长度
    length = 0
    for i in range(try_times):
        if L_group[i] - 0.01 <= L_min <= L_group[i] + 0.01:
            length = i + 1
    # 路径长度为0说明路径不存在
    if length != 0:
        # 根据路径长度获取路径对应的虚拟配变名单
        c = virtual_transformer_distance_sort[0:length].tolist()
        # 对dummy点、HV_station_id_i、HV_station_id_j、c中的虚拟配变生成距离矩阵
        node_num = 2 + length + 1
        node_distance = np.zeros((node_num, node_num))
        # dummy与其他节点的距离
        node_distance[0, 1] = 0
        node_distance[0, 2] = 0
        for (iteration, transformer_virtual_id) in enumerate(c):
            node_distance[0, 3 + iteration] = 1E6
        # HV_station_id_i与其他节点的距离
        node_distance[1, 0] = 0
        node_distance[1, 2] = all_node_distance[HV_station_id_i, HV_station_id_j]
        for (iteration, transformer_virtual_id) in enumerate(c):
            node_distance[1, 3 + iteration] = all_node_distance[HV_station_id_i, c[iteration]+len(planning_object.planning_HV_station_name)]
        # HV_station_id_j与其他节点的距离
        node_distance[2, 0] = 0
        node_distance[2, 1] = all_node_distance[HV_station_id_j, HV_station_id_i]
        for (iteration, transformer_virtual_id) in enumerate(c):
            node_distance[2, 3 + iteration] = all_node_distance[HV_station_id_j, c[iteration]+len(planning_object.planning_HV_station_name)]
        # c中的虚拟配变与其他节点的距离
        for (iteration, transformer_virtual_id) in enumerate(c):
            node_distance[3 + iteration, 0] = 1E6
            node_distance[3 + iteration, 1] = all_node_distance[HV_station_id_i, c[iteration]+len(planning_object.planning_HV_station_name)]
            node_distance[3 + iteration, 2] = all_node_distance[HV_station_id_j, c[iteration]+len(planning_object.planning_HV_station_name)]
            for (iteration_, transformer_virtual_id_) in enumerate(c):
                if iteration != iteration_:
                    node_distance[3 + iteration, 3 + iteration_] = all_node_distance[c[iteration]+len(planning_object.planning_HV_station_name), c[iteration_]+len(planning_object.planning_HV_station_name)]
        # 根据距离矩阵求解指定起点终点的旅行商问题
        p_ = elkai.solve_float_matrix(node_distance)
        # 从求解结果获取虚拟配变的连接顺序
        p = [0] * length
        for i in range(length):
            p[i] = c[p_[i + 2] - 3]
        if p_[1] == 2:
            p.reverse()
        return p
    else:
        return []


def print_all_node(planning_object):
    plt.scatter(planning_object.planning_transformer_xy[:, 0], planning_object.planning_transformer_xy[:, 1], c="b")
    plt.scatter(planning_object.planning_HV_station_xy[:, 0], planning_object.planning_HV_station_xy[:, 1], c="r")
    plt.scatter(planning_object.planning_transformer_xy_virtual[:, 0], planning_object.planning_transformer_xy_virtual[:, 1], c="g")
    plt.show()


def print_every_wire(max_out_wire, planning_object, S_X, L_min_X, L_group_X, considering_street):
    # 遍历每一个出线层级
    for out_wire_id in range(max_out_wire):
        # 单环网代号
        wire_cnt = 0
        # 遍历每两个上级变电站之间的单环网
        for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
            for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                if HV_station_id_i < HV_station_id_j:
                    # 属于该单环网的所有配变的坐标
                    planning_transformer_x = []
                    planning_transformer_y = []
                    # 两个上级变电站的坐标
                    planning_HV_station_x = [planning_object.planning_HV_station_xy[HV_station_id_i, 0], planning_object.planning_HV_station_xy[HV_station_id_j, 0]]
                    planning_HV_station_y = [planning_object.planning_HV_station_xy[HV_station_id_i, 1], planning_object.planning_HV_station_xy[HV_station_id_j, 1]]
                    for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
                        if 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_i] <= 1.1 and 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_j] <= 1.1:
                            planning_transformer_x.append(planning_object.planning_transformer_xy[transformer_id, 0])
                            planning_transformer_y.append(planning_object.planning_transformer_xy[transformer_id, 1])

                    # 根据L_min反推出单环网所经过的虚拟配变路径
                    p = get_virtual_transformer_path(L_min_X[out_wire_id, wire_cnt], L_group_X[out_wire_id, wire_cnt, :], planning_object, wire_cnt, HV_station_id_i, HV_station_id_j)
                    if len(p) != 0:
                        # 画出单环网对应的上级变电站和配变的点
                        plt.scatter(planning_transformer_x, planning_transformer_y, c="b")
                        plt.scatter(planning_HV_station_x, planning_HV_station_y, c="r")
                        # 单环网各节点的坐标
                        wire_x = []
                        wire_y = []
                        if considering_street is False:
                            wire_x.append(planning_object.planning_HV_station_xy[HV_station_id_i, 0])
                            for virtual_transformer_id in p:
                                wire_x.append(planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 0])
                            wire_x.append(planning_object.planning_HV_station_xy[HV_station_id_j, 0])
                            wire_y.append(planning_object.planning_HV_station_xy[HV_station_id_i, 1])
                            for virtual_transformer_id in p:
                                wire_y.append(planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 1])
                            wire_y.append(planning_object.planning_HV_station_xy[HV_station_id_j, 1])
                        else:
                            nodes = planning_object.street_graph.nodes.data()

                            wire_x.append(planning_object.planning_HV_station_xy[HV_station_id_i, 0])
                            wire_y.append(planning_object.planning_HV_station_xy[HV_station_id_i, 1])
                            start_node = find_nearest_street_node(planning_object.planning_HV_station_xy[HV_station_id_i, 0], planning_object.planning_HV_station_xy[HV_station_id_i, 1], nodes)
                            end_node = find_nearest_street_node(planning_object.planning_transformer_xy_virtual[p[0], 0], planning_object.planning_transformer_xy_virtual[p[0], 1], nodes)
                            node_list = get_shortest_street_path(planning_object, start_node, end_node)
                            for street_node in node_list:
                                wire_x.append(nodes[street_node]["x"])
                                wire_y.append(nodes[street_node]["y"])

                            for (virtual_transformer_idx, virtual_transformer_id) in enumerate(p):
                                if virtual_transformer_idx == len(p) - 1:
                                    break
                                wire_x.append(planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 0])
                                wire_y.append(planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 1])
                                start_node = find_nearest_street_node(planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 0], planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 1], nodes)
                                end_node = find_nearest_street_node(planning_object.planning_transformer_xy_virtual[p[virtual_transformer_idx+1], 0], planning_object.planning_transformer_xy_virtual[p[virtual_transformer_idx+1], 1], nodes)
                                node_list = get_shortest_street_path(planning_object, start_node, end_node)
                                for street_node in node_list:
                                    wire_x.append(nodes[street_node]["x"])
                                    wire_y.append(nodes[street_node]["y"])

                            wire_x.append(planning_object.planning_transformer_xy_virtual[p[-1], 0])
                            wire_y.append(planning_object.planning_transformer_xy_virtual[p[-1], 1])
                            start_node = find_nearest_street_node(planning_object.planning_transformer_xy_virtual[p[-1], 0], planning_object.planning_transformer_xy_virtual[p[-1], 1], nodes)
                            end_node = find_nearest_street_node(planning_object.planning_HV_station_xy[HV_station_id_j, 0], planning_object.planning_HV_station_xy[HV_station_id_j, 1], nodes)
                            node_list = get_shortest_street_path(planning_object, start_node, end_node)
                            for street_node in node_list:
                                wire_x.append(nodes[street_node]["x"])
                                wire_y.append(nodes[street_node]["y"])

                            wire_x.append(planning_object.planning_HV_station_xy[HV_station_id_j, 0])
                            wire_y.append(planning_object.planning_HV_station_xy[HV_station_id_j, 1])
                        # 绘制单环网
                        plt.plot(wire_x, wire_y, c="r")
                        plt.show()
                    wire_cnt += 1


def print_all_wire(max_out_wire, planning_object, S_X, L_min_X, L_group_X, considering_street):
    # 给每个配变打上所属上级变电站的标签
    selection_label = np.zeros(len(planning_object.planning_transformer_name))
    for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
        for out_wire_id in range(max_out_wire):
            wire_cnt = 0
            for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
                for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                    if HV_station_id_i < HV_station_id_j:
                        if 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_i] <= 1.1 and 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_j] <= 1.1:
                            selection_label[transformer_id] = wire_cnt
                        wire_cnt += 1

    # 画出带标签的配变点
    plt.scatter(planning_object.planning_transformer_xy[:, 0], planning_object.planning_transformer_xy[:, 1], c=selection_label)
    # 画出虚拟配变
    plt.scatter(planning_object.planning_transformer_xy_virtual[:, 0], planning_object.planning_transformer_xy_virtual[:, 1], c="g")
    # 画出上级变电站
    plt.scatter(planning_object.planning_HV_station_xy[:, 0], planning_object.planning_HV_station_xy[:, 1], c="r")
    # 遍历每一个出线层级
    for out_wire_id in range(max_out_wire):
        # 单环网代号
        wire_cnt = 0
        # 遍历每两个上级变电站之间的单环网
        for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
            for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                if HV_station_id_i < HV_station_id_j:
                    # 根据L_min反推出单环网所经过的虚拟配变路径
                    p = get_virtual_transformer_path(L_min_X[out_wire_id, wire_cnt], L_group_X[out_wire_id, wire_cnt, :], planning_object, wire_cnt, HV_station_id_i, HV_station_id_j)
                    if len(p) != 0:
                        # 单环网各节点的坐标
                        wire_x = []
                        wire_y = []
                        if considering_street is False:
                            wire_x.append(planning_object.planning_HV_station_xy[HV_station_id_i, 0])
                            for virtual_transformer_id in p:
                                wire_x.append(planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 0])
                            wire_x.append(planning_object.planning_HV_station_xy[HV_station_id_j, 0])
                            wire_y.append(planning_object.planning_HV_station_xy[HV_station_id_i, 1])
                            for virtual_transformer_id in p:
                                wire_y.append(planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 1])
                            wire_y.append(planning_object.planning_HV_station_xy[HV_station_id_j, 1])
                        else:
                            nodes = planning_object.street_graph.nodes.data()

                            wire_x.append(planning_object.planning_HV_station_xy[HV_station_id_i, 0])
                            wire_y.append(planning_object.planning_HV_station_xy[HV_station_id_i, 1])
                            start_node = find_nearest_street_node(planning_object.planning_HV_station_xy[HV_station_id_i, 0], planning_object.planning_HV_station_xy[HV_station_id_i, 1], nodes)
                            end_node = find_nearest_street_node(planning_object.planning_transformer_xy_virtual[p[0], 0], planning_object.planning_transformer_xy_virtual[p[0], 1], nodes)
                            node_list = get_shortest_street_path(planning_object, start_node, end_node)
                            for street_node in node_list:
                                wire_x.append(nodes[street_node]["x"])
                                wire_y.append(nodes[street_node]["y"])

                            for (virtual_transformer_idx, virtual_transformer_id) in enumerate(p):
                                if virtual_transformer_idx == len(p) - 1:
                                    break
                                wire_x.append(planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 0])
                                wire_y.append(planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 1])
                                start_node = find_nearest_street_node(planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 0], planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 1], nodes)
                                end_node = find_nearest_street_node(planning_object.planning_transformer_xy_virtual[p[virtual_transformer_idx+1], 0], planning_object.planning_transformer_xy_virtual[p[virtual_transformer_idx+1], 1], nodes)
                                node_list = get_shortest_street_path(planning_object, start_node, end_node)
                                for street_node in node_list:
                                    wire_x.append(nodes[street_node]["x"])
                                    wire_y.append(nodes[street_node]["y"])

                            wire_x.append(planning_object.planning_transformer_xy_virtual[p[-1], 0])
                            wire_y.append(planning_object.planning_transformer_xy_virtual[p[-1], 1])
                            start_node = find_nearest_street_node(planning_object.planning_transformer_xy_virtual[p[-1], 0], planning_object.planning_transformer_xy_virtual[p[-1], 1], nodes)
                            end_node = find_nearest_street_node(planning_object.planning_HV_station_xy[HV_station_id_j, 0], planning_object.planning_HV_station_xy[HV_station_id_j, 1], nodes)
                            node_list = get_shortest_street_path(planning_object, start_node, end_node)
                            for street_node in node_list:
                                wire_x.append(nodes[street_node]["x"])
                                wire_y.append(nodes[street_node]["y"])

                            wire_x.append(planning_object.planning_HV_station_xy[HV_station_id_j, 0])
                            wire_y.append(planning_object.planning_HV_station_xy[HV_station_id_j, 1])

                        # 绘制单环网
                        wire_x = np.array(wire_x) + random.randint(-30, 30)
                        wire_y = np.array(wire_y) + random.randint(-30, 30)
                        plt.plot(wire_x, wire_y, c=plt.cm.get_cmap("tab10")(wire_cnt % 10))
                    wire_cnt += 1
    plt.show()


def find_nearest_street_node(x, y, nodes):
    nodes = dict(nodes)
    find_range_x = [50, 200, 500]
    find_range_y = [50, 200, 500]
    for k in range(len(find_range_x)):
        valid_node_id = []
        for node_id in list(nodes.keys()):
            if x-find_range_x[k] <= nodes[node_id]["x"] <= x+find_range_x[k] and y-find_range_y[k] <= nodes[node_id]["y"] <= y+find_range_y[k]:
                valid_node_id.append(node_id)
        if len(valid_node_id) > 0:
            break
    street_node_distance = np.zeros(len(valid_node_id))
    for (idx, node_id) in enumerate(valid_node_id):
        street_node_distance[idx] = euclidean_distance(np.array([nodes[node_id]["x"] - x, nodes[node_id]["y"] - y]))
    street_node_distance_sort = np.argsort(street_node_distance)
    return valid_node_id[street_node_distance_sort[0]]


def get_shortest_street_path(planning_object, start_node, end_node):
    return nx.shortest_path(planning_object.street_graph, start_node, end_node)


if __name__ == '__main__':
    planning_object = PlanningObject()
    nodes = planning_object.street_graph.nodes.data()
    # 虚拟配变数目改变时要重新生成
    # all_node_num = len(planning_object.planning_HV_station_name) + planning_object.planning_transformer_num_virtual
    # all_node_distance = np.zeros((all_node_num, all_node_num))
    # cnt = 0
    # total_cnt = (all_node_num * all_node_num - all_node_num) / 2
    # for iteration_i in range(all_node_num):
    #     for iteration_j in range(all_node_num):
    #         if iteration_i < iteration_j:
    #             print(f"{cnt}/{total_cnt}")
    #             cnt += 1
    #             if iteration_i < len(planning_object.planning_HV_station_name) and iteration_j < len(
    #                     planning_object.planning_HV_station_name):
    #                 all_node_distance[iteration_i, iteration_j] = street_distance(
    #                     planning_object.planning_HV_station_xy[iteration_i, 0],
    #                     planning_object.planning_HV_station_xy[iteration_j, 0],
    #                     planning_object.planning_HV_station_xy[iteration_i, 1],
    #                     planning_object.planning_HV_station_xy[iteration_j, 1],
    #                     nodes)
    #             elif iteration_i < len(planning_object.planning_HV_station_name) and iteration_j >= len(
    #                     planning_object.planning_HV_station_name):
    #                 all_node_distance[iteration_i, iteration_j] = street_distance(
    #                     planning_object.planning_HV_station_xy[iteration_i, 0],
    #                     planning_object.planning_transformer_xy_virtual[
    #                         iteration_j - len(planning_object.planning_HV_station_name), 0],
    #                     planning_object.planning_HV_station_xy[iteration_i, 1],
    #                     planning_object.planning_transformer_xy_virtual[
    #                         iteration_j - len(planning_object.planning_HV_station_name), 1],
    #                     nodes)
    #             elif iteration_i >= len(planning_object.planning_HV_station_name) and iteration_j >= len(
    #                     planning_object.planning_HV_station_name):
    #                 all_node_distance[iteration_i, iteration_j] = street_distance(
    #                     planning_object.planning_transformer_xy_virtual[
    #                         iteration_i - len(planning_object.planning_HV_station_name), 0],
    #                     planning_object.planning_transformer_xy_virtual[
    #                         iteration_j - len(planning_object.planning_HV_station_name), 0],
    #                     planning_object.planning_transformer_xy_virtual[
    #                         iteration_i - len(planning_object.planning_HV_station_name), 1],
    #                     planning_object.planning_transformer_xy_virtual[
    #                         iteration_j - len(planning_object.planning_HV_station_name), 1],
    #                     nodes)
    # for iteration_i in range(all_node_num):
    #     for iteration_j in range(all_node_num):
    #         if iteration_i > iteration_j:
    #             all_node_distance[iteration_i, iteration_j] = all_node_distance[iteration_j, iteration_i]
    # save_variable(all_node_distance, f"all_node_distance_{planning_object.planning_transformer_num_virtual}VTS.np")
    all_node_distance = load_variable(f"all_node_distance_{planning_object.planning_transformer_num_virtual}VTS.np")

    print_all_node(planning_object)

    # 对每个月获得各月的低、中、高典型异常增长率，得到三种典型场景：低异常增长、中异常增长、高异常增长，以及各场景的概率权重
    typical_abnormal_rate = np.zeros((12, 3))
    typical_abnormal_rate_possibility_temp = np.zeros((12, 3))
    for month_id in range(12):
        x = planning_object.abnormal_pdf_x
        pdf = planning_object.abnormal_pdf[month_id, :]
        cdf = planning_object.abnormal_cdf[month_id, :]
        # 获取5%下界和5%上界
        lb_5 = 0
        ub_5 = 0
        for x_id in range(len(x)):
            if cdf[x_id] >= 0.05:
                lb_5 = x[x_id]
                break
        for x_id in range(len(x)):
            if cdf[len(x) - x_id - 1] <= 0.95:
                ub_5 = x[x_id]
                break
        # 获取三分点
        point_1_3 = (ub_5 - lb_5) / 3 + lb_5
        point_2_3 = (ub_5 - lb_5) / 3 * 2 + lb_5
        # 在[lb_5, point_1_3]中取pdf最大处作为低典型异常增长率
        pdf_max = 0
        for x_id in range(len(x)):
            if lb_5 <= x[x_id] <= point_1_3:
                if pdf[x_id] > pdf_max:
                    typical_abnormal_rate[month_id, 0] = x[x_id]
                    typical_abnormal_rate_possibility_temp[month_id, 0] = pdf[x_id]
                    pdf_max = pdf[x_id]
        # 在[point_1_3, point_2_3]中取pdf最大处作为中典型异常增长率
        pdf_max = 0
        for x_id in range(len(x)):
            if point_1_3 <= x[x_id] <= point_2_3:
                if pdf[x_id] > pdf_max:
                    typical_abnormal_rate[month_id, 1] = x[x_id]
                    typical_abnormal_rate_possibility_temp[month_id, 1] = pdf[x_id]
                    pdf_max = pdf[x_id]
        # 在[point_2_3, ub_5]中取pdf最大处作为高典型异常增长率
        pdf_max = 0
        for x_id in range(len(x)):
            if point_2_3 <= x[x_id] <= ub_5:
                if pdf[x_id] > pdf_max:
                    typical_abnormal_rate[month_id, 2] = x[x_id]
                    typical_abnormal_rate_possibility_temp[month_id, 1] = pdf[x_id]
                    pdf_max = pdf[x_id]
    # 对三种典型场景的概率权重进行归一化
    typical_abnormal_rate_possibility = np.ones(3)
    for month_id in range(12):
        typical_abnormal_rate_possibility = typical_abnormal_rate_possibility * typical_abnormal_rate_possibility_temp[month_id, :]
    typical_abnormal_rate_possibility /= np.sum(typical_abnormal_rate_possibility)

    for scene in range(3):
        # Create a new model
        model = gp.Model("TRANSFORMER_PLANNING")

        max_out_wire = 3 # 俩变电站最多出3条不同的单环网

        # 配变的增容量作为决策量，单位kVA，下限是0，上限是100
        C = model.addMVar((len(planning_object.planning_transformer_name),), vtype=GRB.CONTINUOUS, lb=0, name="transformer_capacity_extend")
        # 配变所连接的2个变电站（0-1变量）
        S = model.addMVar((max_out_wire, len(planning_object.planning_transformer_name), len(planning_object.planning_HV_station_name)), vtype=GRB.BINARY, name="transformer_HVstation_relationship")
        # 俩俩高压变电站之间单环网的长度
        wire_cnt = 0
        for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
            for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                if HV_station_id_i < HV_station_id_j:
                    wire_cnt = wire_cnt + 1
        L_min = model.addMVar((max_out_wire, wire_cnt), vtype=GRB.CONTINUOUS, lb=0, name="HVstation_wire_length")
        L_existence = model.addMVar((max_out_wire, wire_cnt), vtype=GRB.BINARY, name="HVstation_wire_existence")
        try_times = planning_object.planning_transformer_num_virtual
        L_group = model.addMVar((max_out_wire, wire_cnt, try_times), vtype=GRB.CONTINUOUS, lb=0, name="HVstation_wire_length_group")
        S_multiply = model.addMVar((max_out_wire, len(planning_object.planning_transformer_name), wire_cnt), vtype=GRB.BINARY, name="transformer_HVstation_relationship_multiply")

        # 目标函数1：配变成本
        cost_transformer = 0
        # 配变：[固定费用+可变费用（与总容量有关）]*贴现公式+运维费用
        fc = 0#100000
        r = 0.05
        m = 1
        mc = 0#600000
        ratio = r * pow(1 + r, m) / (pow(1 + r, m) - 1)
        for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
            cost_transformer = cost_transformer + (fc + 175.91*(planning_object.planning_transformer_capacity_kVA[transformer_id] + C[transformer_id]) + 72993) * ratio + mc

        # 目标函数2：线路成本
        cost_wire = 0
        # 线路：[固定费用+可变费用（与线路长度有关）]*贴现公式+运维费用（网损）
        fc = 0#100000
        r = 0.05
        m = 1
        ratio = r * pow(1 + r, m) / (pow(1 + r, m) - 1)
        mc = 0
        # 因为网损数量级较小，导致优化收敛速度慢，因此通过fix_ratio来提高数量级
        fix_ratio = 1
        for out_wire_id in range(max_out_wire):
            # street_distance_matrix = np.zeros((len(planning_object.planning_transformer_name), len(planning_object.planning_HV_station_name)))
            street_distance_matrix = load_variable("street_distance_matrix.np")
            for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
                # yuan/kWh * ohm/km * h / kV / kV = yuan/k /km /k/k /W/W
                beta_l = 0.5 * 0.45 * 3600 / (12.47 * 12.47 * planning_object.planning_transformer_cos_phi[transformer_id] * planning_object.planning_transformer_cos_phi[transformer_id])
                P = np.max(planning_object.planning_transformer_load_profile[transformer_id, :])
                L_to_HV_station = 0
                for (HV_station_id, HV_station) in enumerate(planning_object.planning_HV_station_name):
                    # street_distance_matrix[transformer_id, HV_station_id] = street_distance(planning_object.planning_HV_station_xy[HV_station_id, 0],
                    #                 planning_object.planning_transformer_xy[transformer_id, 0],
                    #                 planning_object.planning_HV_station_xy[HV_station_id, 1],
                    #                 planning_object.planning_transformer_xy[transformer_id, 1], nodes)
                    L_to_HV_station = L_to_HV_station + street_distance_matrix[transformer_id, HV_station_id] * S[out_wire_id, transformer_id, HV_station_id]
                mc = mc + beta_l * P * P * L_to_HV_station * fix_ratio
                print(transformer_id)
            # save_variable(street_distance_matrix,"street_distance_matrix.np")
        for out_wire_id in range(max_out_wire):
            wire_cnt = 0
            for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
                for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                    if HV_station_id_i < HV_station_id_j:
                        cost_wire = cost_wire + 200 * L_min[out_wire_id, wire_cnt] * 1000 * L_existence[out_wire_id, wire_cnt]
                        wire_cnt = wire_cnt + 1
        cost_wire = (fc + cost_wire) * ratio + mc

        # 总目标函数
        cost = cost_transformer + cost_wire
        model.setObjective(cost, GRB.MINIMIZE)

        # 添加约束
        # 约束0：S_multiply是S元素之间的乘积
        for out_wire_id in range(max_out_wire):
            for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
                wire_cnt = 0
                for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
                    for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                        if HV_station_id_i < HV_station_id_j:
                            model.addConstr(S_multiply[out_wire_id, transformer_id, wire_cnt] <= S[out_wire_id, transformer_id, HV_station_id_i])
                            model.addConstr(S_multiply[out_wire_id, transformer_id, wire_cnt] <= S[out_wire_id, transformer_id, HV_station_id_j])
                            model.addConstr(S_multiply[out_wire_id, transformer_id, wire_cnt] >= S[out_wire_id, transformer_id, HV_station_id_i] + S[out_wire_id, transformer_id, HV_station_id_j] - 1)
                            wire_cnt += 1

        if scene == 0:
            # 约束1：L_min[wire_cnt]是两个变电站之间trytimes条虚拟配变排列路径组L_group_list中的最小合法值
            for out_wire_id in range(max_out_wire):
                wire_cnt = 0
                for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
                    for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                        if HV_station_id_i < HV_station_id_j:
                            try_times = planning_object.planning_transformer_num_virtual
                            L_group_list = [0] * try_times
                            now_times = 0
                            all_virtual_transformer_distance = np.zeros(planning_object.planning_transformer_num_virtual)
                            wire_cnt_ = 0
                            for (HV_station_id_i_, HV_station_i_) in enumerate(planning_object.planning_HV_station_name):
                                for (HV_station_id_j_, HV_station_j_) in enumerate(planning_object.planning_HV_station_name):
                                    if HV_station_id_i_ < HV_station_id_j_:
                                        wire_cnt_ += 1
                            all_virtual_transformer_distance_matrix = np.zeros((planning_object.planning_transformer_num_virtual, wire_cnt_))
                            for virtual_transformer_id in range(planning_object.planning_transformer_num_virtual):
                                wire_cnt_ = 0
                                for (HV_station_id_i_, HV_station_i_) in enumerate(planning_object.planning_HV_station_name):
                                    for (HV_station_id_j_, HV_station_j_) in enumerate(
                                            planning_object.planning_HV_station_name):
                                        if HV_station_id_i_ < HV_station_id_j_:
                                            all_virtual_transformer_distance_matrix[virtual_transformer_id, wire_cnt_] += all_node_distance[HV_station_id_i_, virtual_transformer_id+len(planning_object.planning_HV_station_name)]
                                            all_virtual_transformer_distance_matrix[virtual_transformer_id, wire_cnt_] += all_node_distance[HV_station_id_j_, virtual_transformer_id+len(planning_object.planning_HV_station_name)]
                                            wire_cnt_ += 1
                                all_virtual_transformer_distance[virtual_transformer_id] = all_virtual_transformer_distance_matrix[virtual_transformer_id, wire_cnt] / np.sum(all_virtual_transformer_distance_matrix[virtual_transformer_id, :])
                            virtual_transformer_distance_sort = np.argsort(all_virtual_transformer_distance)
                            for length in range(1, planning_object.planning_transformer_num_virtual + 1):
                                c = virtual_transformer_distance_sort[0:length].tolist()
                                print(f"{now_times}/{try_times}")
                                # 对dummy点、HV_station_id_i、HV_station_id_j、c中的虚拟配变生成距离矩阵
                                node_num = 2 + length + 1
                                node_distance = np.zeros((node_num, node_num))
                                # dummy与其他节点的距离
                                node_distance[0, 1] = 0
                                node_distance[0, 2] = 0
                                for (iteration, transformer_virtual_id) in enumerate(c):
                                    node_distance[0, 3 + iteration] = 1E6
                                # HV_station_id_i与其他节点的距离
                                node_distance[1, 0] = 0
                                node_distance[1, 2] = all_node_distance[HV_station_id_i, HV_station_id_j]
                                for (iteration, transformer_virtual_id) in enumerate(c):
                                    node_distance[1, 3 + iteration] = all_node_distance[HV_station_id_i, c[iteration]+len(planning_object.planning_HV_station_name)]
                                # HV_station_id_j与其他节点的距离
                                node_distance[2, 0] = 0
                                node_distance[2, 1] = all_node_distance[HV_station_id_j, HV_station_id_i]
                                for (iteration, transformer_virtual_id) in enumerate(c):
                                    node_distance[2, 3 + iteration] = all_node_distance[HV_station_id_j, c[iteration]+len(planning_object.planning_HV_station_name)]
                                # c中的虚拟配变与其他节点的距离
                                for (iteration, transformer_virtual_id) in enumerate(c):
                                    node_distance[3 + iteration, 0] = 1E6
                                    node_distance[3 + iteration, 1] = all_node_distance[HV_station_id_i, c[iteration]+len(planning_object.planning_HV_station_name)]
                                    node_distance[3 + iteration, 2] = all_node_distance[HV_station_id_j, c[iteration]+len(planning_object.planning_HV_station_name)]
                                    for (iteration_, transformer_virtual_id_) in enumerate(c):
                                        if iteration != iteration_:
                                            node_distance[3 + iteration, 3 + iteration_] = all_node_distance[c[iteration]+len(planning_object.planning_HV_station_name), c[iteration_]+len(planning_object.planning_HV_station_name)]
                                p_ = elkai.solve_float_matrix(node_distance)
                                p = [0] * length
                                for i in range(length):
                                    p[i] = c[p_[i + 2] - 3]
                                if p_[1] == 2:
                                    p.reverse()
                                # save_variable(p,
                                #               fr"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\5.考虑不确定性的选址定容优化\TSP_result\{planning_object.planning_transformer_num_virtual}VTS-{HV_station_id_i}-{HV_station_id_j}-{now_times}.list")
                                # p = load_variable(
                                #     fr"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\5.考虑不确定性的选址定容优化\TSP_result\{planning_object.planning_transformer_num_virtual}VTS-{HV_station_id_i}-{HV_station_id_j}-{now_times}.list")
                                # 生成 0 ~ length(例20) 的某排列[12, 3, 9, ..., 2, 16]
                                # 获取 HV_station_id_i至12, 12至3,...16至HV_station_id_j的各距离
                                distance = np.zeros(length + 1)
                                distance[0] = all_node_distance[HV_station_id_i, p[0]+len(planning_object.planning_HV_station_name)]
                                for (iteration, transformer_virtual_id) in enumerate(p):
                                    if iteration >= length - 1:
                                        break
                                    distance[iteration + 1] = all_node_distance[p[iteration]+len(planning_object.planning_HV_station_name), p[iteration+1]+len(planning_object.planning_HV_station_name)]
                                distance[length] = all_node_distance[HV_station_id_j, p[length-1]+len(planning_object.planning_HV_station_name)]
                                # Big_M_inside_wire = (属于本次排列的虚拟配变区域至少有1个配变连接对应两个变电站为0，否则大于0) * Big_M + ...
                                # Big_M_outside_wire = (不属于本次排列的虚拟配变区域没有配变连接对应两个变电站时为0，否则大于0) * Big_M + ...
                                Big_M = 100
                                Big_M_inside_wire = 0  # 符合要求则为0，否则是个大数
                                Big_M_outside_wire = 0  # 符合要求则为0，否则是个大数
                                # validity描述了每个虚拟配变区域中的配变，其与变电站的连接关系是否与供电网相匹配
                                validity = [0] * planning_object.planning_transformer_num_virtual
                                for transformer_virtual_id in range(
                                        planning_object.planning_transformer_num_virtual):
                                    if transformer_virtual_id in p:
                                        S_multiply_OR = model.addVar(vtype=GRB.BINARY,
                                                                     name="transformer_HVstation_relationship_multiply_OR")
                                        S_multiply_OR_group = []
                                        for (transformer_id, transformer) in enumerate(
                                                planning_object.planning_transformer_name):
                                            if planning_object.planning_transformer_virtual_label[
                                                transformer_id] == transformer_virtual_id:
                                                S_multiply_OR_group.append(S_multiply[out_wire_id, transformer_id, wire_cnt])
                                        model.addGenConstrOr(S_multiply_OR, S_multiply_OR_group)
                                        validity[transformer_virtual_id] = 1 - S_multiply_OR
                                        Big_M_inside_wire = Big_M_inside_wire + validity[
                                            transformer_virtual_id] * Big_M
                                    else:
                                        S_multiply_OR2 = model.addVar(vtype=GRB.BINARY,
                                                                      name="transformer_HVstation_relationship_multiply_OR2")
                                        S_multiply_OR2_group = []
                                        for (transformer_id, transformer) in enumerate(
                                                planning_object.planning_transformer_name):
                                            if planning_object.planning_transformer_virtual_label[
                                                transformer_id] == transformer_virtual_id:
                                                S_multiply_OR2_group.append(S_multiply[out_wire_id, transformer_id, wire_cnt])
                                        model.addGenConstrOr(S_multiply_OR2, S_multiply_OR2_group)
                                        validity[transformer_virtual_id] = S_multiply_OR2
                                        Big_M_outside_wire = Big_M_outside_wire + validity[
                                            transformer_virtual_id] * Big_M
                                # for (iteration, transformer_virtual_id) in enumerate(p):
                                #     for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
                                #         if planning_object.planning_transformer_virtual_label[transformer_id] == transformer_virtual_id:
                                #             validity[iteration] = validity[iteration] + 2 - S[transformer_id, HV_station_id_i] - S[transformer_id, HV_station_id_j]
                                #     Big_Num = Big_Num + validity[iteration] * Big_M
                                # L = HV_station_id_i至12距离 + ...
                                L = np.sum(distance) / 1000
                                # L 与 Big_M_outside_wire + Big_M_inside_wire 进行组合，若合法则为L，若非法则为大数，加入L_group_list中
                                model.addConstr(L_group[
                                                    out_wire_id, wire_cnt, now_times] <= L + Big_M_outside_wire + Big_M_inside_wire + 0.1)
                                model.addConstr(L_group[
                                                    out_wire_id, wire_cnt, now_times] >= L + Big_M_outside_wire + Big_M_inside_wire - 0.1)
                                L_group_list[now_times] = (L_group[out_wire_id, wire_cnt, now_times])
                                now_times += 1
                            model.addGenConstrMin(L_min[out_wire_id, wire_cnt], L_group_list, constant=100)
                            wire_cnt = wire_cnt + 1


            # 约束1.1：L_existence[wire_cnt]是两个变电站之间是否有路径
            for out_wire_id in range(max_out_wire):
                wire_cnt = 0
                for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
                    for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                        if HV_station_id_i < HV_station_id_j:
                            S_multiply_OR_group = []
                            for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
                                S_multiply_OR_group.append(S_multiply[out_wire_id, transformer_id, wire_cnt])
                            model.addGenConstrOr(L_existence[out_wire_id, wire_cnt], S_multiply_OR_group)
                            wire_cnt += 1

            # # 约束1.1：L_existence[wire_cnt]是两个变电站之间是否有路径
            # for out_wire_id in range(max_out_wire):
            #     wire_cnt = 0
            #     for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
            #         for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
            #             if HV_station_id_i < HV_station_id_j:
            #                 model.addConstr((1-L_existence[out_wire_id, wire_cnt]) * 99.9 <= L_min[out_wire_id, wire_cnt])
            #                 model.addConstr(L_min[out_wire_id, wire_cnt] <= 99.9 + 100000*(1-L_existence[out_wire_id, wire_cnt]))
            #                 wire_cnt += 1

            # # 约束1.2：L_min[wire_cnt]不能大于20km
            # wire_cnt = 0
            # for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
            #     for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
            #         if HV_station_id_i < HV_station_id_j:
            #             model.addConstr(L_min[wire_cnt]*L_existence[wire_cnt] <= 20)
            #             wire_cnt += 1

            # 约束1.2：每条wire的配变数目不超过100个
            for out_wire_id in range(max_out_wire):
                wire_cnt = 0
                for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
                    for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                        if HV_station_id_i < HV_station_id_j:
                            cnt = 0
                            for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
                                cnt += S_multiply[out_wire_id, transformer_id, wire_cnt]
                            model.addConstr(cnt <= 100.1)
                            wire_cnt += 1

            # # 约束1.2：每条存在的wire的配变数目与其他存在的wire差别小于100个
            # wire_cnt = 0
            # for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
            #     for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
            #         if HV_station_id_i < HV_station_id_j:
            #             cnt = 0
            #             for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
            #                 cnt += S_multiply[transformer_id, wire_cnt]
            #             wire_cnt_ = 0
            #             for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
            #                 for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
            #                     if HV_station_id_i < HV_station_id_j:
            #                         cnt_ = 0
            #                         for (transformer_id, transformer) in enumerate(
            #                                 planning_object.planning_transformer_name):
            #                             cnt_ += S_multiply[transformer_id, wire_cnt_]
            #                         model.addConstr(cnt - cnt_ <= 100.1 + 1000000*(1-L_existence[wire_cnt]*L_existence[wire_cnt_]))
            #                         model.addConstr(cnt - cnt_ >= -100.1 - 1000000*(1-L_existence[wire_cnt]*L_existence[wire_cnt_]))
            #                         wire_cnt_ += 1
            #             wire_cnt += 1

            # 约束2：每个配变必须有且只有2个同出线变电站连接
            for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
                cnt = 0
                for out_wire_id in range(max_out_wire):
                    wire_cnt = 0
                    for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
                        for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                            if HV_station_id_i < HV_station_id_j:
                                cnt = cnt + S_multiply[out_wire_id, transformer_id, wire_cnt] * L_existence[out_wire_id, wire_cnt]
                                wire_cnt += 1
                model.addConstr(cnt <= 1.1)
                model.addConstr(cnt >= 0.9)

            # 约束2.1：每个变电站必须至少有1个配变连接
            for (HV_station_id, HV_station) in enumerate(planning_object.planning_HV_station_name):
                cnt = 0
                for out_wire_id in range(max_out_wire):
                    for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
                        cnt = cnt + S[out_wire_id, transformer_id, HV_station_id]
                model.addConstr(cnt >= 0.9)


        # 机会约束3：P[(配变原始容量+配变增容量)*功率因素 > 负荷预测值 + 异常增长值 + 负荷预测值*正常增长预测误差] > 90%
        for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
            # 应该是由去年的正常情景负荷曲线来计算异常增长值，但现在时间紧急，先用今年的正常情景负荷曲线来计算异常增长值
            abnormal_increment = typical_abnormal_rate[:, scene].T * planning_object.planning_transformer_load_profile[transformer_id, :]
            forecast_with_abnormal = np.max(planning_object.planning_transformer_load_profile[transformer_id, :] + abnormal_increment)
            cdf_result = 0
            x = planning_object.forecast_error_pdf_x
            cdf = planning_object.forecast_error_cdf[0, :]
            for x_id in range(len(x)):
                if cdf[len(x) - x_id - 1] <= 0.90:
                    cdf_result = x[len(x) - x_id - 1]
                    break
            C_min = 1000 * (cdf_result * np.max(planning_object.planning_transformer_load_profile[transformer_id, :]) + forecast_with_abnormal)/planning_object.planning_transformer_cos_phi[transformer_id] - planning_object.planning_transformer_capacity_kVA[transformer_id]
            model.addConstr(C[transformer_id] >= C_min)

        # # 约束4：每条单环网的配变数目不能少于平均配变的一半
        # for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
        #     for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
        #         if HV_station_id_i < HV_station_id_j:
        #             cnt = 0
        #             for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
        #                 cnt += S[transformer_id, HV_station_id_i] * S[transformer_id, HV_station_id_j]
        #             model.addConstr(cnt >= len(planning_object.planning_transformer_name) / (wire_cnt * 2))

        # 优化
        model.Params.MIPGap = 0.0005
        # model.Params.TimeLimit = 36000
        date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # save_variable(model, f"model_{planning_object.planning_transformer_num_virtual}VTS_{scene}scene.gp")
        # model = load_variable(f"model_{planning_object.planning_transformer_num_virtual}VTS_{scene}scene.gp")
        model.optimize()

        # 获取结果
        print('Obj:', model.objVal)
        print(C.X)
        print(S.X)
        print(L_min.X)
        print(L_group.X)
        date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_variable(C.X, f"C_{planning_object.planning_transformer_num_virtual}VTS_{scene}scene_{date}.gp")
        if scene == 0:
            save_variable(S.X, f"S_{planning_object.planning_transformer_num_virtual}VTS_{scene}scene_{date}.gp")
            save_variable(L_min.X, f"Lmin_{planning_object.planning_transformer_num_virtual}VTS_{scene}scene_{date}.gp")
            save_variable(L_group.X, f"Lgroup_{planning_object.planning_transformer_num_virtual}VTS_{scene}scene_{date}.gp")
            save_variable(planning_object, f"planning_object_{date}.plaobj")

            print_every_wire(max_out_wire, planning_object, S.X, L_min.X, L_group.X, True)
            print_all_wire(max_out_wire, planning_object, S.X, L_min.X, L_group.X, True)


