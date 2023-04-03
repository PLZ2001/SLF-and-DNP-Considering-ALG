import numpy as np
import time
import sys
sys.path.append(r"F:\FTP\计及负荷异常增长的空间负荷预测与配电网规划\4.基于双层优化的网格划分技术")
from GIS_object import get_block_index
import sqlite3
import matplotlib.pyplot as plt
import datetime
import networkx as nx
import sys
sys.path.append(r"F:\FTP\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import save_variable, load_variable


GIS_object = load_variable(r"F:\FTP\计及负荷异常增长的空间负荷预测与配电网规划\4.基于双层优化的网格划分技术\rationality_evaluation_result_20230325_144634.gisobj")
# 读取用户数据
conn = sqlite3.connect(r'F:\FTP\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
cur = conn.cursor()
cur.execute('''select * from "入户点数据"''')
conn.commit()
users = cur.fetchall()
# 读取供电所数据
cur = conn.cursor()
cur.execute('''select * from "供电所数据"''')
conn.commit()
TSs = cur.fetchall()
# 读取高压变电站数据
cur = conn.cursor()
cur.execute('''select * from "高压变电站数据"''')
conn.commit()
HV_stations = cur.fetchall()
# 读取配电变电站数据
cur = conn.cursor()
cur.execute('''select * from "配电变电站数据"''')
conn.commit()
Transformers = cur.fetchall()
# 读取斯坦纳点数据
cur = conn.cursor()
cur.execute('''select * from "斯坦纳点数据"''')
conn.commit()
SteinerNodes = cur.fetchall()
for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
    for (idx_power_supply_mesh, power_supply_mesh) in enumerate(GIS_object.power_supply_meshes[idx_power_supply_partition]):
        print(f"现在是供电分区（{power_supply_partition}）中的供电网格（{power_supply_mesh}）")
        GIS_object.get_network()
        # 供电网格的粗略范围
        valid_blocks = np.zeros((GIS_object.horizontal_block_num, GIS_object.vertical_block_num))

        # 获取供电网格的用户名单
        user_ids_of_partition = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 3].tolist()
        user_ids = []
        for i in range(len(user_ids_of_partition)):
            if GIS_object.power_supply_mesh_map[idx_power_supply_partition][i, 2] == idx_power_supply_mesh:
                user_ids.append(int(user_ids_of_partition[i]))

        # 计算供电网格的粗略范围
        for user_id in user_ids:
            x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(users[user_id][4], users[user_id][5])
            block_index_x, block_index_y = get_block_index(x=x, y=y,
                                                           left_bottom_x=GIS_object.left_bottom_x,
                                                           left_bottom_y=GIS_object.left_bottom_y,
                                                           right_top_x=GIS_object.right_top_x,
                                                           right_top_y=GIS_object.right_top_y,
                                                           horizontal_block_num=GIS_object.horizontal_block_num,
                                                           vertical_block_num=GIS_object.vertical_block_num)
            valid_blocks[block_index_x, block_index_y] = 1

        # 将供电网格的粗略范围扩大一些
        x_min = 9999999
        x_max = 0
        y_min = 9999999
        y_max = 0
        for x in range(GIS_object.horizontal_block_num):
            for y in range(GIS_object.vertical_block_num):
                if valid_blocks[x, y] == 1:
                    if x < x_min:
                        x_min = x
                    elif x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    elif y > y_max:
                        y_max = y
        extend_blocks = 1
        x_min = x_min - extend_blocks if 0 <= x_min - extend_blocks else 0
        x_max = x_max + extend_blocks if GIS_object.horizontal_block_num - 1 >= x_min + extend_blocks else GIS_object.horizontal_block_num - 1
        y_min = y_min - extend_blocks if 0 <= y_min - extend_blocks else 0
        y_max = y_max + extend_blocks if GIS_object.vertical_block_num - 1 >= y_min + extend_blocks else GIS_object.vertical_block_num - 1
        for x in range(GIS_object.horizontal_block_num):
            for y in range(GIS_object.vertical_block_num):
                if x_min<=x<=x_max and y_min<=y<=y_max:
                    valid_blocks[x, y] = 1
        # 从所有节点中筛选出属于供电网格的节点
        valid_node_id = []
        for user in users:
            x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(user[4], user[5])
            block_index_x, block_index_y = get_block_index(x=x, y=y,
                                                           left_bottom_x=GIS_object.left_bottom_x,
                                                           left_bottom_y=GIS_object.left_bottom_y,
                                                           right_top_x=GIS_object.right_top_x,
                                                           right_top_y=GIS_object.right_top_y,
                                                           horizontal_block_num=GIS_object.horizontal_block_num,
                                                           vertical_block_num=GIS_object.vertical_block_num)
            if valid_blocks[block_index_x, block_index_y] == 1:
                valid_node_id.append(GIS_object.node_name_list.index(user[1]))
        for TS in TSs:
            x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(TS[4], TS[5])
            block_index_x, block_index_y = get_block_index(x=x, y=y,
                                                           left_bottom_x=GIS_object.left_bottom_x,
                                                           left_bottom_y=GIS_object.left_bottom_y,
                                                           right_top_x=GIS_object.right_top_x,
                                                           right_top_y=GIS_object.right_top_y,
                                                           horizontal_block_num=GIS_object.horizontal_block_num,
                                                           vertical_block_num=GIS_object.vertical_block_num)
            if valid_blocks[block_index_x, block_index_y] == 1:
                valid_node_id.append(GIS_object.node_name_list.index(TS[1]))
        for HV_station in HV_stations:
            x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(HV_station[4], HV_station[5])
            block_index_x, block_index_y = get_block_index(x=x, y=y,
                                                           left_bottom_x=GIS_object.left_bottom_x,
                                                           left_bottom_y=GIS_object.left_bottom_y,
                                                           right_top_x=GIS_object.right_top_x,
                                                           right_top_y=GIS_object.right_top_y,
                                                           horizontal_block_num=GIS_object.horizontal_block_num,
                                                           vertical_block_num=GIS_object.vertical_block_num)
            if valid_blocks[block_index_x, block_index_y] == 1:
                valid_node_id.append(GIS_object.node_name_list.index(HV_station[1]))
        for Transformer in Transformers:
            x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(Transformer[4], Transformer[5])
            block_index_x, block_index_y = get_block_index(x=x, y=y,
                                                           left_bottom_x=GIS_object.left_bottom_x,
                                                           left_bottom_y=GIS_object.left_bottom_y,
                                                           right_top_x=GIS_object.right_top_x,
                                                           right_top_y=GIS_object.right_top_y,
                                                           horizontal_block_num=GIS_object.horizontal_block_num,
                                                           vertical_block_num=GIS_object.vertical_block_num)
            if valid_blocks[block_index_x, block_index_y] == 1:
                valid_node_id.append(GIS_object.node_name_list.index(Transformer[1]))
        for SteinerNode in SteinerNodes:
            x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(SteinerNode[4], SteinerNode[5])
            block_index_x, block_index_y = get_block_index(x=x, y=y,
                                                           left_bottom_x=GIS_object.left_bottom_x,
                                                           left_bottom_y=GIS_object.left_bottom_y,
                                                           right_top_x=GIS_object.right_top_x,
                                                           right_top_y=GIS_object.right_top_y,
                                                           horizontal_block_num=GIS_object.horizontal_block_num,
                                                           vertical_block_num=GIS_object.vertical_block_num)
            if valid_blocks[block_index_x, block_index_y] == 1:
                valid_node_id.append(GIS_object.node_name_list.index(SteinerNode[1]))

        # 删除不在供电单元的节点
        graph = GIS_object.graph.copy()
        for (node_id, node) in enumerate(GIS_object.node_name_list):
            if node_id not in valid_node_id:
                graph.remove_node(node_id)
        for node in list(graph.nodes):
            user_node = GIS_object.node_name_list.index(users[user_ids[0]][1])
            if not nx.has_path(graph, node, user_node):
                graph.remove_node(node)
        nx.write_graphml(graph, f"供电分区（{power_supply_partition}）-供电网格（{power_supply_mesh}）.graphml")
        save_variable(graph, f"供电分区（{power_supply_partition}）-供电网格（{power_supply_mesh}）.nx")
        print("生成完毕")
        options = {
            'node_color': 'black',
            'node_size': 10,
            'width': 1,
        }
        plt.rcParams['figure.dpi'] = 500
        pos = nx.spring_layout(graph)
        node_labels = nx.get_node_attributes(graph, 'name')
        nx.draw(graph, pos, **options)
        nx.draw_networkx_labels(graph, pos, node_labels, font_size=3, font_color='r')
        plt.show()

