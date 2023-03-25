import math
import sqlite3
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyproj import CRS
from pyproj import Transformer
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import save_variable, load_variable


# 坐标系转换
class CRSTransformer:
    def __init__(self):
        self.crs_WebMercator = CRS.from_epsg(3857)  # Web墨卡托投影坐标系
        self.crs_WGS84 = CRS.from_epsg(4326)  # WGS 84地理坐标系
        self.crs_UTM = CRS.from_epsg(32617)  # WGS 84/ UTM zone 17N 投影坐标系
        self.mercator2lnglat = Transformer.from_crs(self.crs_WebMercator, self.crs_WGS84, always_xy=True)
        self.lnglat2mercator = Transformer.from_crs(self.crs_WGS84, self.crs_WebMercator, always_xy=True)
        self.gisxy2mercator = Transformer.from_crs(self.crs_UTM, self.crs_WebMercator, always_xy=True)


# 获取某个点属于哪一个地块（地块编号(block_index_x,block_index_y)，左下角是(0,0)）
def get_block_index(x, y, left_bottom_x, left_bottom_y, right_top_x, right_top_y, horizontal_block_num, vertical_block_num):
    block_width = (right_top_x - left_bottom_x) / horizontal_block_num
    block_height = (right_top_y - left_bottom_y) / vertical_block_num
    block_index_x = (x - left_bottom_x) // block_width
    block_index_y = (y - left_bottom_y) // block_height
    return int(block_index_x), int(block_index_y)


# 计算梯形模糊函数的隶属度
def trapezoid_fuzzy_function(x, a, b, c, d):
    if x < a or x >= d:
        return 0
    elif a <= x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return 1
    elif c <= x < d:
        return (x - c) / (d - c)


class GISObject:
    # 这是一个地图对象，划分为150*150的网格，每个网格可获取其负荷密度
    # 采用墨卡托坐标系
    def __init__(self, horizontal_block_num, vertical_block_num):
        # 左下-80.0108694 35.9606000 右上-79.4898001 36.3386954
        # 坐标系转换器
        self.CRS_transformer = CRSTransformer()
        # 整个地图的四点坐标
        self.left_bottom_x, self.left_bottom_y = self.CRS_transformer.lnglat2mercator.transform(-80.0108694, 35.9606000)
        self.right_top_x, self.right_top_y = self.CRS_transformer.lnglat2mercator.transform(-79.4898001, 36.3386954)
        # 方格个数
        self.horizontal_block_num = horizontal_block_num
        self.vertical_block_num = vertical_block_num
        # 方格大小
        self.block_width = (self.right_top_x - self.left_bottom_x) / self.horizontal_block_num
        self.block_height = (self.right_top_y - self.left_bottom_y) / self.vertical_block_num
        # 用方格表示的负荷密度
        self.load_density = np.zeros((self.horizontal_block_num, self.vertical_block_num))
        # 各方格低压用户数目
        self.lv_user_num = np.zeros((self.horizontal_block_num, self.vertical_block_num))
        # 各方格中压用户数目
        self.mv_user_num = np.zeros((self.horizontal_block_num, self.vertical_block_num))
        # 用方格表示的饱和负荷密度
        self.saturated_load_density = np.zeros((self.horizontal_block_num, self.vertical_block_num))
        # 各方格低压用户年负荷曲线
        self.lv_load_profile = np.zeros((self.horizontal_block_num, self.vertical_block_num, 12))
        # 各方格中压用户年负荷曲线
        self.mv_load_profile = np.zeros((self.horizontal_block_num, self.vertical_block_num, 12))
        # 供电区域初始总类（用于优化）
        self.power_supply_areas = ["A_plus", "A", "B", "C", "D", "E", "None"]
        # 各供电区域划分依据
        self.fuzzy_area = {"A_plus": [30, 100], "A": [15, 30], "B": [6, 15], "C": [1, 6], "D": [0.1, 1], "E": [0, 0.1]}
        # 各方格的供电区域隶属度
        self.power_supply_area_score = np.zeros((len(self.power_supply_areas), self.horizontal_block_num, self.vertical_block_num))
        # 用方格表示的供电区域优化结果
        self.power_supply_area_optimization = np.zeros((len(self.power_supply_areas), self.horizontal_block_num, self.vertical_block_num))
        # 配电线路总图
        self.graph = nx.Graph()
        # 节点名
        self.node_name_list = []
        # 用终端用户表示的供电区域优化结果
        self.power_supply_area_map = np.zeros((70551, 4))
        # 用终端用户表示的行政划分
        self.administrative_division_map = np.zeros((70551, 4))
        # 用终端用户表示的供电所范围
        self.TS_area_map = np.zeros((70551, 4))
        # 行政划分名字
        self.administrative_divisions = []
        # 供电所范围名字
        self.TS_areas = []
        # 供电分区名字
        self.power_supply_partitions = []
        # 用终端用户表示的供电分区
        self.power_supply_partition_map = np.zeros((70551, 4))
        # 供电网格名字
        self.power_supply_meshes = []
        # 用终端用户表示的供电网格
        self.power_supply_mesh_map = np.zeros((70551, 4))
        # 用户年负荷曲线
        self.user_load_profile = np.zeros((70551, 12))

    # 读取数据库，获取所有块的2018年负荷密度（包括低压用户和中压用户）
    def get_all_load_density(self):
        self.load_density = np.zeros((self.horizontal_block_num, self.vertical_block_num))
        # 先计算低压用户的
        self.lv_user_num = np.zeros((self.horizontal_block_num, self.vertical_block_num))
        self.lv_load_profile = np.zeros((self.horizontal_block_num, self.vertical_block_num, 12))

        conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\2.基于集成学习的空间负荷预测\空间负荷预测结果.db')
        cur = conn.cursor()
        cur.execute('''select * from "空间负荷预测结果"''')
        conn.commit()
        results = cur.fetchall()
        for result in results:
            x, y = self.CRS_transformer.lnglat2mercator.transform(result[2], result[3])
            block_index_x, block_index_y = get_block_index(x=x, y=y,
                                                           left_bottom_x=self.left_bottom_x,
                                                           left_bottom_y=self.left_bottom_y,
                                                           right_top_x=self.right_top_x,
                                                           right_top_y=self.right_top_y,
                                                           horizontal_block_num=self.horizontal_block_num,
                                                           vertical_block_num=self.vertical_block_num)
            load_profile = result[52:52+12]
            self.lv_load_profile[block_index_x, block_index_y, :] += np.array(load_profile)
            self.lv_user_num[block_index_x, block_index_y] += 1
        for x in range(self.horizontal_block_num):
            for y in range(self.vertical_block_num):
                if self.lv_user_num[x, y] > 0:
                    self.load_density[x, y] = max(self.lv_load_profile[x, y, :]) * 100000 / (self.lv_user_num[x, y] * 500) # 每个低压用户的平均占地面积按500m^2来计算

        # 再计算中压用户的
        self.mv_user_num = np.zeros((self.horizontal_block_num, self.vertical_block_num))
        self.mv_load_profile = np.zeros((self.horizontal_block_num, self.vertical_block_num, 12))

        conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\中压负荷数据表.db')
        cur = conn.cursor()
        cur.execute('''select * from "负荷数据表" where "年份" = 2018''')
        conn.commit()
        results = cur.fetchall()
        for result in results:
            x, y = self.CRS_transformer.lnglat2mercator.transform(result[3], result[4])
            block_index_x, block_index_y = get_block_index(x=x, y=y,
                                                           left_bottom_x=self.left_bottom_x,
                                                           left_bottom_y=self.left_bottom_y,
                                                           right_top_x=self.right_top_x,
                                                           right_top_y=self.right_top_y,
                                                           horizontal_block_num=self.horizontal_block_num,
                                                           vertical_block_num=self.vertical_block_num)
            load_profile_365 = np.array(result[33:33+365]) / 1000
            load_profile = np.zeros(12)
            months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
            for index, month in enumerate(months):
                start = 0
                for month_before in range(1, month):
                    start += days[month_before]
                end = start + days[month]
                load_profile[index] = np.max(load_profile_365[start:end])
            self.mv_user_num[block_index_x, block_index_y] += 1
            self.mv_load_profile[block_index_x, block_index_y, :] += np.array(load_profile)
        for x in range(self.horizontal_block_num):
            for y in range(self.vertical_block_num):
                if self.mv_user_num[x, y] > 0:
                    self.load_density[x, y] += max(self.mv_load_profile[x, y, :]) * 100000 / (self.mv_user_num[x, y] * 5000) # 每个中压用户的平均占地面积按5000m^2来计算

    # # 根据所有块的2018年负荷密度（包括低压用户和中压用户），用3%年增长率增长10年来估算饱和负荷
    # def get_all_saturated_load_density(self):
    #     self.saturated_load_density = self.load_density * np.power(1.03, 10)

    # 计算得到各网格的供电区域模糊评分
    def calculate_power_supply_area_score(self):
        self.power_supply_area_score = np.zeros((len(self.power_supply_areas), self.horizontal_block_num, self.vertical_block_num))
        for x in range(self.horizontal_block_num):
            for y in range(self.vertical_block_num):
                if self.load_density[x, y] > 0:
                    for (idx, power_supply_area) in enumerate(self.power_supply_areas):
                        if power_supply_area == "None":
                            self.power_supply_area_score[idx, x, y] = 0
                        else:
                            b = self.fuzzy_area[power_supply_area][0]
                            c = self.fuzzy_area[power_supply_area][1]
                            a = b - (c - b) / 3
                            d = c + (c - b) / 3
                            f_x = self.load_density[x, y]
                            self.power_supply_area_score[idx, x, y] = trapezoid_fuzzy_function(x=f_x, a=a, b=b, c=c, d=d)
                else:
                    for (idx, power_supply_area) in enumerate(self.power_supply_areas):
                        if power_supply_area == "None":
                            self.power_supply_area_score[idx, x, y] = 1
                        else:
                            self.power_supply_area_score[idx, x, y] = 0

    # 读取数据库，生成由供电所、高压变电站、配电变电站、入户点组成的无向图结构（https://blog.51cto.com/baoqiangwang/5196365）
    def get_network(self):
        # self.node_name_list = []
        # conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
        # cur = conn.cursor()
        # cur.execute('''select * from "线路数据"''')
        # conn.commit()
        # results = cur.fetchall()
        # self.graph = nx.Graph()
        # for (idx, result) in enumerate(results):
        #     print(idx)
        #     start_node = result[4]
        #     end_node = result[5]
        #     if start_node.endswith("_69"):
        #         start_node = start_node[0:-3]
        #     elif start_node.endswith("_1247"):
        #         start_node = start_node[0:-5]
        #     elif start_node.endswith("LV"):
        #         start_node = start_node[0:-2]
        #     if end_node.endswith("_69"):
        #         end_node = end_node[0:-3]
        #     elif end_node.endswith("_1247"):
        #         end_node = end_node[0:-5]
        #     elif end_node.endswith("LV"):
        #         end_node = end_node[0:-2]
        #     if start_node not in self.node_name_list:
        #         self.node_name_list.append(start_node)
        #     if end_node not in self.node_name_list:
        #         self.node_name_list.append(end_node)
        #     self.graph.add_nodes_from([
        #         (self.node_name_list.index(start_node), {"name": start_node}),
        #         (self.node_name_list.index(end_node), {"name": end_node}),
        #     ])
        #     self.graph.add_edges_from([
        #         (self.node_name_list.index(start_node), self.node_name_list.index(end_node), {"name": result[1]}),
        #     ])
        # save_variable(self.graph, "graph.nx")
        # save_variable(self.node_name_list, "node_name_list.list")
        self.node_name_list = load_variable("node_name_list.list")
        self.graph = load_variable("graph.nx")
        # pos = nx.spring_layout(self.graph)
        # nx.draw(self.graph, pos, with_labels=True, alpha=0.9)
        # plt.show()
        # print(f"节点数目：{self.graph.number_of_nodes()}")
        # print(f"边的数目：{self.graph.number_of_edges()}")

    def get_user_load_profile(self):
        # # 读取用户数据
        # _conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
        # _cur = _conn.cursor()
        # _cur.execute('''select * from "入户点数据"''')
        # _conn.commit()
        # users = _cur.fetchall()
        # self.user_load_profile = np.zeros((len(users), 12))
        # # 先读取低压用户的
        # conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\2.基于集成学习的空间负荷预测\空间负荷预测结果.db')
        # cur = conn.cursor()
        # cur.execute('''select * from "空间负荷预测结果"''')
        # conn.commit()
        # results = cur.fetchall()
        # for result in results:
        #     _cur.execute('''select * from "入户点数据" where "负荷名称" = ?''', (result[1].upper(),))
        #     _conn.commit()
        #     try:
        #         indexes = _cur.fetchall()
        #         for index in indexes:
        #             self.user_load_profile[index[0], :] = result[52:52+12]
        #     except:
        #         pass
        # # 再读取中压用户的
        # conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\中压负荷数据表.db')
        # cur = conn.cursor()
        # cur.execute('''select * from "负荷数据表" where "年份" = 2018''')
        # conn.commit()
        # results = cur.fetchall()
        # for result in results:
        #     _cur.execute('''select * from "入户点数据" where "负荷名称" = ?''', (result[1].upper(),))
        #     _conn.commit()
        #     try:
        #         indexes = _cur.fetchall()
        #         load_profile_365 = np.array(result[33:33+365]) / 1000
        #         load_profile = np.zeros(12)
        #         months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        #         days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        #         for index, month in enumerate(months):
        #             start = 0
        #             for month_before in range(1, month):
        #                 start += days[month_before]
        #             end = start + days[month]
        #             load_profile[index] = np.max(load_profile_365[start:end])
        #         for index in indexes:
        #             self.user_load_profile[index[0], :] = load_profile
        #     except:
        #         pass
        # save_variable(self.user_load_profile, "user_load_profile.np")
        self.user_load_profile = load_variable("user_load_profile.np")


if __name__ == '__main__':
    GIS_object = GISObject(horizontal_block_num=300, vertical_block_num=300)
    GIS_object.get_all_load_density()
    GIS_object.calculate_power_supply_area_score()