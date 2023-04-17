# 假设变电站选址已经确定，线路容量已经足够，只规划变电站容量、配变容量和线路连接关系
# 研究范围：一个供电网格，已知变电站和规划变电站的位置，各配变的位置及原始容量，各配变连接的负荷点，各负荷点的负荷预测值、功率因数、异常增长概率密度函数、正常增长误差概率密度函数

# 上层：变电站选址定容
# 决策量：配变选取的变电站（0-1变量），变电站的容量（连续变量），配变的增容量（连续变量）
# 优化目标：[固定费用+可变费用（与总容量有关）]*贴现公式+运维费用
# 机会约束：P[变电站容量*功率因素 > 负荷预测值 + 异常增长值 + 正常增长预测误差] > 90%
# 机会约束：P[(配变原始容量+配变增容量)*功率因素 > 负荷预测值 + 异常增长值 + 正常增长预测误差] > 90%
# 多场景生成：随机生成各负荷的异常增长率，得到带有异常增长的负荷预测值
# 优化方法：粒子群法，目标是各场景的目标值平均值

# 下层：线路选线
# 决策量：配变连接的多个变电站（0-1变量）
# 优化目标：[固定费用+可变费用（与线路长度有关）]*贴现公式+运维费用（网损）
# 约束：变电站N-1约束
# 优化方法：粒子群法

# 双层优化方法：当上层或下层的目标函数值无法比上一代更好时，终止迭代

import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\4.基于模糊综合评价理论的网格划分技术")
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\2.基于集成学习的空间负荷预测")
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import get_autoencoder1, evaluate, save_variable, load_variable, evaluate_and_get_normal_component
from KDE import h_optimizer, kde
from SLF_forecast import get_load_profile_12_by_name
import sqlite3
import numpy as np
import networkx as nx
from sklearn.cluster import BisectingKMeans
import matplotlib.pyplot as plt


class PlanningObject():
    def __init__(self):
        self.GIS_object = load_variable(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\4.基于模糊综合评价理论的网格划分技术\rationality_evaluation_result_20230325_144634.gisobj")
        self.GIS_object.get_network()
        # 读取用户数据
        conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
        cur = conn.cursor()
        cur.execute('''select * from "入户点数据"''')
        conn.commit()
        self.users = cur.fetchall()
        self.user_names = []
        for user in self.users:
            self.user_names.append(user[1])
        # 读取供电所数据
        cur = conn.cursor()
        cur.execute('''select * from "供电所数据"''')
        conn.commit()
        self.TSs = cur.fetchall()
        self.TS_names = []
        for TS in self.TSs:
            self.TS_names.append(TS[1])
        # 读取高压变电站数据
        cur = conn.cursor()
        cur.execute('''select * from "高压变电站数据"''')
        conn.commit()
        self.HV_stations = cur.fetchall()
        self.HV_station_names = []
        for HV_station in self.HV_stations:
            self.HV_station_names.append(HV_station[1])
        # 读取配电变电站数据
        cur = conn.cursor()
        cur.execute('''select * from "配电变电站数据"''')
        conn.commit()
        self.Transformers = cur.fetchall()
        self.Transformer_names = []
        for Transformer in self.Transformers:
            self.Transformer_names.append(Transformer[1])
        # 读取斯坦纳点数据
        cur = conn.cursor()
        cur.execute('''select * from "斯坦纳点数据"''')
        conn.commit()
        self.SteinerNodes = cur.fetchall()
        self.SteinerNode_names = []
        for SteinerNode in self.SteinerNodes:
            self.SteinerNode_names.append(SteinerNode[1])
        # 读取道路数据
        cur = conn.cursor()
        cur.execute('''select * from "道路数据"''')
        conn.commit()
        self.StreetBranches = cur.fetchall()
        # 生成道路图数据
        self.street_node_name_list = []
        self.street_graph = nx.Graph()
        self.get_street_graph()

        # 获取空间负荷预测结果
        conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\2.基于集成学习的空间负荷预测\空间负荷预测结果.db')
        cur = conn.cursor()
        cur.execute('''select * from "空间负荷预测结果"''')
        conn.commit()
        self.user_loads = cur.fetchall()

        # 获取规划区域网架结构
        self.planning_grid_graph = load_variable("供电分区（industrial-ITS0）-供电网格（IHS0-IHS1-IHS5）.nx")

        # 规划区域涉及变电站（包括已有变电站和规划变电站）的名称
        self.planning_HV_station_name = []
        self.get_planning_HV_station_name()
        # 规划区域涉及变电站（包括已有变电站和规划变电站）的位置信息
        self.planning_HV_station_xy = np.zeros((len(self.planning_HV_station_name), 2))
        self.get_planning_HV_station_xy()

        # 规划区域涉及配变的名称
        self.planning_transformer_name = []
        self.get_planning_transformer_name()
        # 规划区域涉及配变的位置信息
        self.planning_transformer_xy = np.zeros((len(self.planning_transformer_name), 2))
        self.get_planning_transformer_xy()
        # 规划区域涉及虚拟配变的数目（20个）
        self.planning_transformer_num_virtual = 30
        # 规划区域涉及虚拟配变的名称（20个）
        self.planning_transformer_name_virtual = range(self.planning_transformer_num_virtual)
        # 规划区域涉及虚拟配变的位置信息（20个）和配变所属虚拟配变的标签
        self.planning_transformer_xy_virtual = np.zeros((self.planning_transformer_num_virtual, 2))
        self.planning_transformer_virtual_label = np.zeros(len(self.planning_transformer_name))
        self.get_planning_transformer_xy_virtual()
        
        # 规划区域涉及配变的原始容量
        self.planning_transformer_capacity_kVA = np.zeros(len(self.planning_transformer_name))
        self.get_planning_transformer_capacity_kVA()

        # 规划区域涉及终端用户的名称和数目
        self.planning_user_name = []
        self.planning_lv_user_num = 0
        self.planning_mv_user_num = 0
        self.get_planning_user_name()

        # 规划区域涉及终端用户的负荷曲线
        self.planning_user_load_profile = np.zeros((len(self.planning_user_name), 12))
        self.get_planning_user_load_profile()

        # 规划区域涉及终端用户的功率因数
        self.planning_user_cos_phi = np.zeros(len(self.planning_user_name))
        self.get_planning_user_cos_phi()

        # 规划区域涉及配变与负荷点的对应关系
        self.planning_transformer_users = {}
        self.get_planning_transformer_users()

        # 规划区域涉及配变的平均功率因数
        self.planning_transformer_cos_phi = np.zeros(len(self.planning_transformer_name))
        self.get_planning_transformer_cos_phi()

        # 规划区域涉及配变的负荷曲线
        self.planning_transformer_load_profile = np.zeros((len(self.planning_transformer_name), 12))
        self.get_planning_transformer_load_profile()

        # 规划区域的异常增长概率密度函数
        self.abnormal_pdf_x = np.zeros(1)
        self.abnormal_pdf = np.zeros(1)
        self.abnormal_cdf = np.zeros(1)
        self.get_abnormal_pdf()

        # 规划区域的预测误差概率密度函数
        self.forecast_error_pdf_x = np.zeros(1)
        self.forecast_error_pdf = np.zeros(1)
        self.forecast_error_cdf = np.zeros(1)
        self.get_forecast_error_pdf()

    def get_planning_HV_station_name(self):
        all_nodes = nx.nodes(self.planning_grid_graph)
        for node_id in all_nodes:
            if self.GIS_object.node_name_list[node_id].find("HS") != -1:
                self.planning_HV_station_name.append(self.GIS_object.node_name_list[node_id])

    def get_planning_HV_station_xy(self):
        for (id_HV_station, HV_station) in enumerate(self.planning_HV_station_name):
            self.planning_HV_station_xy[id_HV_station, 0] = self.HV_stations[self.HV_station_names.index(HV_station)][4]
            self.planning_HV_station_xy[id_HV_station, 1] = self.HV_stations[self.HV_station_names.index(HV_station)][5]

    def get_planning_transformer_name(self):
        all_nodes = nx.nodes(self.planning_grid_graph)
        for node_id in all_nodes:
            if self.GIS_object.node_name_list[node_id].find("DT") != -1:
                self.planning_transformer_name.append(self.GIS_object.node_name_list[node_id])

    def get_planning_transformer_xy(self):
        for (id_transformer, transfomer) in enumerate(self.planning_transformer_name):
            self.planning_transformer_xy[id_transformer, 0] = self.Transformers[self.Transformer_names.index(transfomer)][4]
            self.planning_transformer_xy[id_transformer, 1] = self.Transformers[self.Transformer_names.index(transfomer)][5]

    def get_planning_transformer_xy_virtual(self):
        kmeans = BisectingKMeans(n_clusters=self.planning_transformer_num_virtual, bisecting_strategy='biggest_inertia')
        kmeans.fit(self.planning_transformer_xy)
        for i in range(self.planning_transformer_num_virtual):
            self.planning_transformer_xy_virtual[i, 0] = kmeans.cluster_centers_[i, 0]
            self.planning_transformer_xy_virtual[i, 1] = kmeans.cluster_centers_[i, 1]
        for (id_transformer, transfomer) in enumerate(self.planning_transformer_name):
            self.planning_transformer_virtual_label[id_transformer] = kmeans.labels_[id_transformer]
        # plt.scatter(self.planning_transformer_xy[:, 0], self.planning_transformer_xy[:, 1], c="b")
        # plt.scatter(self.planning_transformer_xy_virtual[:, 0], self.planning_transformer_xy_virtual[:, 1], c="r")
        # plt.show()

    def get_planning_transformer_capacity_kVA(self):
        for (id_transformer, transfomer) in enumerate(self.planning_transformer_name):
            self.planning_transformer_capacity_kVA[id_transformer] = self.Transformers[self.Transformer_names.index(transfomer)][6]

    def get_planning_user_name(self):
        all_nodes = nx.nodes(self.planning_grid_graph)
        for node_id in all_nodes:
            if self.GIS_object.node_name_list[node_id].find("LV") != -1:
                self.planning_user_name.append(self.GIS_object.node_name_list[node_id])
                self.planning_lv_user_num += 1
            elif self.GIS_object.node_name_list[node_id].find("MV") != -1:
                self.planning_user_name.append(self.GIS_object.node_name_list[node_id])
                self.planning_mv_user_num += 1

    def get_planning_user_load_profile(self):
        for (id_user, user) in enumerate(self.planning_user_name):
            if user.endswith('U'):
                user = user[0:-2]
            self.planning_user_load_profile[id_user, :] = self.GIS_object.user_load_profile[self.user_names.index(user), :]

    def get_planning_user_cos_phi(self):
        for (id_user, user) in enumerate(self.planning_user_name):
            self.planning_user_cos_phi[id_user] = self.users[self.user_names.index(user)][8]

    def get_planning_transformer_cos_phi(self):
        for (id_transformer, transfomer) in enumerate(self.planning_transformer_name):
            cos_phi = 0
            cnt = 0
            for user in self.planning_transformer_users[transfomer]:
                cos_phi += self.planning_user_cos_phi[self.planning_user_name.index(user)]
                cnt += 1
            self.planning_transformer_cos_phi[id_transformer] = cos_phi / cnt

    def get_planning_transformer_load_profile(self):
        for (id_transformer, transfomer) in enumerate(self.planning_transformer_name):
            load_profile = np.zeros(12)
            for user in self.planning_transformer_users[transfomer]:
                load_profile += self.planning_user_load_profile[self.planning_user_name.index(user)]
            self.planning_transformer_load_profile[id_transformer] = load_profile

    def get_planning_transformer_users(self):
        explored_node = []
        for transformer_name in self.planning_transformer_name:
            self.planning_transformer_users[transformer_name] = []
            node_id = self.GIS_object.node_name_list.index(transformer_name)
            self.find_neighbor_user(transformer_name, node_id, explored_node)

    def find_neighbor_user(self, transformer_name, node_id, explored_node):
        explored_node.append(node_id)
        neighbor_nodes = self.planning_grid_graph[node_id]
        for neighbor_node_id in list(neighbor_nodes.keys()):
            if self.GIS_object.node_name_list[neighbor_node_id].find("LV") != -1 or self.GIS_object.node_name_list[neighbor_node_id].find("MV") != -1 and neighbor_node_id not in explored_node:
                self.planning_transformer_users[transformer_name].append(self.GIS_object.node_name_list[neighbor_node_id])
                self.find_neighbor_user(transformer_name, neighbor_node_id, explored_node)
            elif self.GIS_object.node_name_list[neighbor_node_id].find("DM") != -1 and neighbor_node_id not in explored_node:
                self.find_neighbor_user(transformer_name, neighbor_node_id, explored_node)

    def get_abnormal_pdf(self):
        # print("正在生成异常增长样本矩阵...")
        # # 生成样本矩阵
        # months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        # data_len = self.planning_lv_user_num
        # # 样本矩阵
        # _sample_matrix = np.zeros((data_len, 12))
        # # 数据库
        # conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
        # cur = conn.cursor()
        # cur.execute('''select * from "负荷数据表" where "年份" = 2016 OR "年份" = 2017''')
        # conn.commit()
        # results = cur.fetchall()
        # # 获取自编码器模型
        # auto_encoder = get_autoencoder1(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\AutoEncoder_20230125_123858.path")
        # # 建立名字和idx的映射表
        # name_map = {}
        # for idx in range(70407):
        #     name_map[results[idx][1]] = idx
        # cnt = 0
        # for user_name in self.planning_user_name:
        #     if user_name.find("LV") != -1:
        #         if user_name.endswith('U'):
        #             user_name = user_name[0:-2]
        #         print(f"{cnt}/{self.planning_lv_user_num}")
        #         # 获取数据
        #         idx = name_map[user_name.lower()]
        #         increment = np.array(results[idx+70407][33:33+365]) / 1000 - np.array(results[idx][33:33+365]) / 1000
        #         # 输入模型
        #         normal_increment, abnormal_increment, mse = evaluate(_auto_encoder=auto_encoder, _increment=increment)
        #         for index, month in enumerate(months):
        #             start = 0
        #             for month_before in range(1, month):
        #                 start += days[month_before]
        #             end = start + days[month]
        #             # 除以基值变为百分比
        #             _sample_matrix[cnt, index] = np.max(abnormal_increment[start:end]) / np.max((np.array(results[idx][33:33+365]) / 1000)[start:end])
        #         cnt += 1
        # save_variable(_sample_matrix, "abnormal_sample_matrix.np")
        # _sample_matrix = load_variable("abnormal_sample_matrix.np")
        # print(f"正在计算概率密度函数...")
        # dim = np.size(_sample_matrix, 1)
        # _x = np.arange(-1.0, 4.0, 0.0001)
        # _pdf = np.zeros((dim, len(_x)))
        # for idx in range(dim):
        #     print(f"正在生成{idx+1}月的概率密度函数..")
        #     _h = h_optimizer(_sample_matrix=np.reshape(_sample_matrix[:, idx], (-1, 1)))
        #     for i in range(len(_x)):
        #         _pdf[idx, i] = kde(_x[i], np.reshape(_sample_matrix[:, idx], (-1, 1)), _h)
        # dim = np.size(_sample_matrix, 1)
        # _x = np.arange(-1.0, 4.0, 0.0001)
        # _cdf = np.zeros((dim, len(_x)))
        # for idx in range(dim):
        #     print(f"正在生成{idx+1}月的累积分布函数..")
        #     sum = 0
        #     for i in range(len(_x)):
        #         sum += _pdf[idx, i] * 0.0001
        #         _cdf[idx, i] = sum
        # save_variable(_x, "abnormal_pdf_x.np")
        # save_variable(_pdf, "abnormal_pdf.np")
        # save_variable(_cdf, "abnormal_cdf.np")
        _x = load_variable("abnormal_pdf_x.np")
        _pdf = load_variable("abnormal_pdf.np")
        _cdf = load_variable("abnormal_cdf.np")
        self.abnormal_pdf_x = _x
        self.abnormal_pdf = _pdf
        self.abnormal_cdf = _cdf

    # 生成规划区域的预测误差概率密度函数
    def get_forecast_error_pdf(self):
        # print("正在生成预测误差样本矩阵...")
        # # 生成样本矩阵
        # months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        # data_len = self.planning_lv_user_num
        # # 样本矩阵
        # _sample_matrix = np.zeros((data_len, 1))
        # # 数据库
        # conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
        # cur = conn.cursor()
        # cur.execute('''select * from "负荷数据表"''')
        # conn.commit()
        # results = cur.fetchall()
        # auto_encoder = get_autoencoder1(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\AutoEncoder_20230125_123858.path")
        # # 建立名字和idx的映射表
        # name_map = {}
        # for idx in range(70407):
        #     name_map[results[idx][1]] = idx
        # cnt = 0
        # for user_name in self.planning_user_name:
        #     if user_name.find("LV") != -1:
        #         if user_name.endswith('U'):
        #             user_name = user_name[0:-2]
        #         print(f"{cnt}/{self.planning_lv_user_num}")
        #         # 获取数据
        #         idx = name_map[user_name.lower()]
        #
        #         _load_profile_12 = np.zeros(12)
        #         result = [results[idx + 70407*2]]
        #         result1 = [results[idx + 70407]]
        #         result2 = [results[idx]]
        #         load_profile_365 = evaluate_and_get_normal_component(_auto_encoder=auto_encoder, _old_load_profile_365=evaluate_and_get_normal_component(_auto_encoder=auto_encoder, _old_load_profile_365=np.array(result2[0][33:33+365]), _new_load_profile_365=np.array(result1[0][33:33+365])), _new_load_profile_365=np.array(result[0][33:33+365]))
        #         load_profile_365 = load_profile_365 / 1000
        #         for index, month in enumerate(months):
        #             start = 0
        #             for month_before in range(1, month):
        #                 start += days[month_before]
        #             end = start + days[month]
        #             _load_profile_12[index] = np.max(load_profile_365[start:end])
        #         _sample_matrix[cnt, 0] = (np.max(_load_profile_12) - self.user_loads[idx][7]) / self.user_loads[idx][7]
        #         cnt += 1
        # save_variable(_sample_matrix, "forecast_error_sample_matrix.np")
        # _sample_matrix = load_variable("forecast_error_sample_matrix.np")
        # print(f"正在计算概率密度函数...")
        # dim = np.size(_sample_matrix, 1)
        # _x = np.arange(-3.0, 3.0, 0.0001)
        # _pdf = np.zeros((dim, len(_x)))
        # for idx in range(dim):
        #     print(f"正在生成概率密度函数..")
        #     _h = h_optimizer(_sample_matrix=np.reshape(_sample_matrix[:, idx], (-1, 1)))
        #     for i in range(len(_x)):
        #         _pdf[idx, i] = kde(_x[i], np.reshape(_sample_matrix[:, idx], (-1, 1)), _h)
        # dim = np.size(_sample_matrix, 1)
        # _x = np.arange(-3.0, 3.0, 0.0001)
        # _cdf = np.zeros((dim, len(_x)))
        # for idx in range(dim):
        #     print(f"正在生成累积分布函数..")
        #     sum = 0
        #     for i in range(len(_x)):
        #         sum += _pdf[idx, i] * 0.0001
        #         _cdf[idx, i] = sum
        # save_variable(_x, "forecast_error_pdf_x.np")
        # save_variable(_pdf, "forecast_error_pdf.np")
        # save_variable(_cdf, "forecast_error_cdf.np")
        _x = load_variable("forecast_error_pdf_x.np")
        _pdf = load_variable("forecast_error_pdf.np")
        _cdf = load_variable("forecast_error_cdf.np")
        self.forecast_error_pdf_x = _x
        self.forecast_error_pdf = _pdf
        self.forecast_error_cdf = _cdf

    def get_street_graph(self):
        # for (idx, result) in enumerate(self.StreetBranches):
        #     print(f"{idx}/{len(self.StreetBranches)}")
        #     start_node = result[3]+'-'+result[2]
        #     end_node = result[6]+'-'+result[2]
        #     if start_node not in self.street_node_name_list:
        #         self.street_node_name_list.append(start_node)
        #     if end_node not in self.street_node_name_list:
        #         self.street_node_name_list.append(end_node)
        #     self.street_graph.add_nodes_from([
        #         (self.street_node_name_list.index(start_node), {"name": start_node, "x": result[4], "y": result[5]}),
        #         (self.street_node_name_list.index(end_node), {"name": end_node, "x": result[7], "y": result[8]}),
        #     ])
        #     self.street_graph.add_edges_from([
        #         (self.street_node_name_list.index(start_node), self.street_node_name_list.index(end_node)),
        #     ])
        # save_variable(self.street_graph, "street_graph.nx")
        # save_variable(self.street_node_name_list, "street_node_name_list.list")
        self.street_node_name_list = load_variable("street_node_name_list.list")
        self.street_graph = load_variable("street_graph.nx")

if __name__ == '__main__':
    planning_object = PlanningObject()


