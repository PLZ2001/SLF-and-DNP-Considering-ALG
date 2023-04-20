import sqlite3
import numpy as np
import pandas as pd
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import save_variable, load_variable
from Transformer_MV_wire_optimize import print_every_wire, print_all_wire, print_all_node, get_virtual_transformer_path, find_nearest_street_node, get_shortest_street_path
import matplotlib.pyplot as plt
import random


def figure_data_1(figure_name):
    planning_object = load_variable("planning_object_20230417_100931.plaobj")
    GIS_object = load_variable(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\4.基于模糊综合评价理论的网格划分技术\rationality_evaluation_result_20230325_144634.gisobj")

    C_X0 = load_variable(f"C_30VTS_0scene_20230417_100931.gp")
    C_X1 = load_variable(f"C_30VTS_1scene_20230417_100943.gp")
    C_X2 = load_variable(f"C_30VTS_2scene_20230417_100955.gp")
    C_X = np.zeros((len(planning_object.planning_transformer_name), 3))
    for j in range(len(planning_object.planning_transformer_name)):
        C_X[j, 0] = C_X0[j]
        C_X[j, 1] = C_X1[j]
        C_X[j, 2] = C_X2[j]
    plt.hist(C_X, range=(0, 500))
    plt.show()

    header = ["配变名称", "维度Lng", "经度Lat", "原始容量（kVA）", "场景0扩容增加（kVA）", "场景0扩容后容量（kVA）", "场景1扩容增加（kVA）", "场景1扩容后容量（kVA）", "场景2扩容增加（kVA）", "场景2扩容后容量（kVA）"]
    for i in range(365):
        header.extend([f"2018年第{i+1}天最大负荷（kW）", f"2018年第{i+1}天扩容前最大负载率", f"2018年第{i+1}天0场景扩容后最大负载率", f"2018年第{i+1}天1场景扩容后最大负载率", f"2018年第{i+1}天2场景扩容后最大负载率"])
    pd_final_table = pd.DataFrame(index=range(len(planning_object.planning_transformer_name)), columns=header)

    # # 获取各配变2018年365天的负荷值
    # # 读取负荷实际数据
    # _conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    # _cur = _conn.cursor()
    # _cur.execute('''select * from "入户点数据"''')
    # _conn.commit()
    # users = _cur.fetchall()
    # user_load_profile = np.zeros((len(users), 365))
    # # 先读取低压用户的
    # conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    # cur = conn.cursor()
    # cur.execute('''select * from "负荷数据表" where "年份" = 2018''')
    # conn.commit()
    # results = cur.fetchall()
    # for result in results:
    #     print(result[0])
    #     _cur.execute('''select * from "入户点数据" where "负荷名称" = ?''', (result[1].upper(),))
    #     _conn.commit()
    #     try:
    #         indexes = _cur.fetchall()
    #         load_profile_365 = np.array(result[33:33+365]) / 1000
    #         for index in indexes:
    #             user_load_profile[index[0], :] = load_profile_365
    #     except:
    #         pass
    # # 再读取中压用户的
    # conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\中压负荷数据表.db')
    # cur = conn.cursor()
    # cur.execute('''select * from "负荷数据表" where "年份" = 2018''')
    # conn.commit()
    # results = cur.fetchall()
    # for result in results:
    #     print(result[0])
    #     _cur.execute('''select * from "入户点数据" where "负荷名称" = ?''', (result[1].upper(),))
    #     _conn.commit()
    #     try:
    #         indexes = _cur.fetchall()
    #         load_profile_365 = np.array(result[33:33+365]) / 1000
    #         for index in indexes:
    #             user_load_profile[index[0], :] = load_profile_365
    #     except:
    #         pass
    #
    # planning_user_load_profile = np.zeros((len(planning_object.planning_user_name), 365))
    # for (id_user, user) in enumerate(planning_object.planning_user_name):
    #     if user.endswith('U'):
    #         user = user[0:-2]
    #     planning_user_load_profile[id_user, :] = user_load_profile[planning_object.user_names.index(user), :]
    #
    # planning_transformer_load_profile = np.zeros((len(planning_object.planning_transformer_name), 365))
    # for (id_transformer, transformer) in enumerate(planning_object.planning_transformer_name):
    #     load_profile = np.zeros(365)
    #     for user in planning_object.planning_transformer_users[transformer]:
    #         load_profile += planning_user_load_profile[planning_object.planning_user_name.index(user)]
    #     planning_transformer_load_profile[id_transformer] = load_profile
    # # 获取完毕：planning_transformer_load_profile
    # save_variable(planning_transformer_load_profile, "2018_planning_transformer_load_profile.np")
    planning_transformer_load_profile = load_variable("2018_planning_transformer_load_profile.np")

    for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
        ts_data = planning_object.Transformers[planning_object.Transformer_names.index(transformer)]
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(ts_data[4], ts_data[5])
        lng = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[0]
        lat = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[1]
        content = [ts_data[1], lng, lat, ts_data[6], C_X[transformer_id, 0], ts_data[6] + C_X[transformer_id, 0], C_X[transformer_id, 1], ts_data[6] + C_X[transformer_id, 1], C_X[transformer_id, 2], ts_data[6] + C_X[transformer_id, 2]]

        for i in range(365):
            content.append(planning_transformer_load_profile[transformer_id, i]*1000)
            content.append(planning_transformer_load_profile[transformer_id, i]*1000 / planning_object.planning_transformer_cos_phi[transformer_id] / (ts_data[6]))
            content.append(planning_transformer_load_profile[transformer_id, i]*1000 / planning_object.planning_transformer_cos_phi[transformer_id] / (ts_data[6] + C_X[transformer_id, 0]))
            content.append(planning_transformer_load_profile[transformer_id, i]*1000 / planning_object.planning_transformer_cos_phi[transformer_id] / (ts_data[6] + C_X[transformer_id, 1]))
            content.append(planning_transformer_load_profile[transformer_id, i]*1000 / planning_object.planning_transformer_cos_phi[transformer_id] / (ts_data[6] + C_X[transformer_id, 2]))

        pd_final_table.loc[transformer_id] = content
    pd_final_table.to_excel(figure_name)


def figure_data_2(figure_name):
    planning_object = load_variable("planning_object_20230418_221117.plaobj")
    GIS_object = load_variable(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\4.基于模糊综合评价理论的网格划分技术\rationality_evaluation_result_20230325_144634.gisobj")

    max_out_wire = 3
    S_X = load_variable(f"S_30VTS_0scene_20230418_221117.gp")
    L_min_X = load_variable(f"Lmin_30VTS_0scene_20230418_221117.gp")
    L_group_X = load_variable(f"Lgroup_30VTS_0scene_20230418_221117.gp")

    print_all_node(planning_object)
    print_every_wire(max_out_wire, planning_object, S_X, L_min_X, L_group_X, True)
    print_all_wire(max_out_wire, planning_object, S_X, L_min_X, L_group_X, True)

    writer = pd.ExcelWriter(figure_name)

    # 配变从属关系
    header = ["配变名称", "维度Lng", "经度Lat", "配变所属开关站代号", "配变所属馈线代号", "配变所属上级变电站名称"]
    pd_final_table = pd.DataFrame(index=range(len(planning_object.planning_transformer_name)), columns=header)
    for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
        ts_data = planning_object.Transformers[planning_object.Transformer_names.index(transformer)]
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(ts_data[4], ts_data[5])
        lng = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[0]
        lat = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[1]
        content = [ts_data[1], lng, lat, f"{int(planning_object.planning_transformer_virtual_label[transformer_id])}号开关站"]
        # 遍历每一个出线层级
        for out_wire_id in range(max_out_wire):
            # 单环网代号
            wire_cnt = 0
            # 遍历每两个上级变电站之间的单环网
            for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
                for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                    if HV_station_id_i < HV_station_id_j:
                        if 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_i] <= 1.1 and 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_j] <= 1.1:
                            content.append(f"{wire_cnt}-{HV_station_i}-{HV_station_j}-{out_wire_id}")
                            content.append(f"{HV_station_i}-{HV_station_j}")
                        wire_cnt += 1
        pd_final_table.loc[transformer_id] = content
    pd_final_table.to_excel(writer, sheet_name=f"配变从属关系")

    # 开关站数据统计
    header = ["开关站代号", "维度Lng", "经度Lat", "开关站接入配变数目", "开关站所属馈线数目", "开关站涉及上级变电站数目"]
    pd_final_table = pd.DataFrame(index=range(planning_object.planning_transformer_num_virtual), columns=header)
    for virtual_transformer_id in range(planning_object.planning_transformer_num_virtual):
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 0], planning_object.planning_transformer_xy_virtual[virtual_transformer_id, 1])
        lng = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[0]
        lat = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[1]
        content = [f"{virtual_transformer_id}号开关站", lng, lat]

        cnt = 0
        for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
            if int(planning_object.planning_transformer_virtual_label[transformer_id]) == virtual_transformer_id:
                cnt += 1
        content.append(cnt)

        wire = []
        for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
            if int(planning_object.planning_transformer_virtual_label[transformer_id]) == virtual_transformer_id:
                # 遍历每一个出线层级
                for out_wire_id in range(max_out_wire):
                    # 单环网代号
                    wire_cnt = 0
                    # 遍历每两个上级变电站之间的单环网
                    for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
                        for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                            if HV_station_id_i < HV_station_id_j:
                                if 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_i] <= 1.1 and 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_j] <= 1.1:
                                    if f"{wire_cnt}-{HV_station_i}-{HV_station_j}-{out_wire_id}" not in wire:
                                        wire.append(f"{wire_cnt}-{HV_station_i}-{HV_station_j}-{out_wire_id}")
                                wire_cnt += 1
        content.append(len(wire))

        HV_station = []
        for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
            if int(planning_object.planning_transformer_virtual_label[transformer_id]) == virtual_transformer_id:
                # 遍历每一个出线层级
                for out_wire_id in range(max_out_wire):
                    # 单环网代号
                    wire_cnt = 0
                    # 遍历每两个上级变电站之间的单环网
                    for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
                        for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                            if HV_station_id_i < HV_station_id_j:
                                if 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_i] <= 1.1 and 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_j] <= 1.1:
                                    if HV_station_i not in HV_station:
                                        HV_station.append(HV_station_i)
                                    if HV_station_j not in HV_station:
                                        HV_station.append(HV_station_j)
                                wire_cnt += 1
        content.append(len(HV_station))

        pd_final_table.loc[virtual_transformer_id] = content
    pd_final_table.to_excel(writer, sheet_name=f"开关站数据统计")

    # 上级变电站数据统计
    header = ["上级变电站名称", "维度Lng", "经度Lat", "联络上级变电站数目", "馈线出线数目", "开关站连接数目", "配变连接数目"]
    pd_final_table = pd.DataFrame(index=range(len(planning_object.planning_HV_station_name)), columns=header)
    for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(planning_object.planning_HV_station_xy[HV_station_id_i, 0], planning_object.planning_HV_station_xy[HV_station_id_i, 1])
        lng = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[0]
        lat = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[1]
        content = [HV_station_i, lng, lat]

        HV_station = []
        for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
            # 遍历每一个出线层级
            for out_wire_id in range(max_out_wire):
                # 遍历每两个上级变电站之间的单环网
                for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                    if HV_station_id_i != HV_station_id_j:
                        if 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_i] <= 1.1 and 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_j] <= 1.1:
                            if HV_station_j not in HV_station:
                                HV_station.append(HV_station_j)
        content.append(len(HV_station))

        wire = []
        for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
            # 遍历每一个出线层级
            for out_wire_id in range(max_out_wire):
                # 遍历每两个上级变电站之间的单环网
                for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                    if HV_station_id_i != HV_station_id_j:
                        if 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_i] <= 1.1 and 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_j] <= 1.1:
                            if f"{HV_station_j}-{out_wire_id}" not in wire:
                                wire.append(f"{HV_station_j}-{out_wire_id}")
        content.append(len(wire))

        virtual_ts = []
        for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
            # 遍历每一个出线层级
            for out_wire_id in range(max_out_wire):
                if 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_i] <= 1.1:
                    if int(planning_object.planning_transformer_virtual_label[transformer_id]) not in virtual_ts:
                        virtual_ts.append(int(planning_object.planning_transformer_virtual_label[transformer_id]))
        content.append(len(virtual_ts))

        cnt = 0
        for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
            # 遍历每一个出线层级
            for out_wire_id in range(max_out_wire):
                if 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_i] <= 1.1:
                    cnt += 1
        content.append(cnt)

        pd_final_table.loc[HV_station_id_i] = content
    pd_final_table.to_excel(writer, sheet_name=f"上级变电站数据统计")

    # 馈线经纬度数据
    wires = []
    for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
        # 遍历每一个出线层级
        for out_wire_id in range(max_out_wire):
            # 单环网代号
            wire_cnt = 0
            # 遍历每两个上级变电站之间的单环网
            for (HV_station_id_i, HV_station_i) in enumerate(planning_object.planning_HV_station_name):
                for (HV_station_id_j, HV_station_j) in enumerate(planning_object.planning_HV_station_name):
                    if HV_station_id_i < HV_station_id_j:
                        if 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_i] <= 1.1 and 0.9 <= S_X[out_wire_id, transformer_id, HV_station_id_j] <= 1.1:
                            if f"{wire_cnt}-{HV_station_i}-{HV_station_j}-{out_wire_id}" not in wires:
                                wires.append(f"{wire_cnt}-{HV_station_i}-{HV_station_j}-{out_wire_id}")
                        wire_cnt += 1
    header = []
    for wire in wires:
        header.append(f"{wire}馈线维度lng")
        header.append(f"{wire}馈线经度lat")
    pd_final_table = pd.DataFrame(index=range(6000), columns=header)

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
                        wire = f"{wire_cnt}-{HV_station_i}-{HV_station_j}-{out_wire_id}"
                        # 单环网各节点的坐标
                        wire_x = []
                        wire_y = []

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

                        wire_x = np.array(wire_x) + random.randint(-30, 30)
                        wire_y = np.array(wire_y) + random.randint(-30, 30)

                        for i in range(len(wire_x)):
                            x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(wire_x[i], wire_y[i])
                            lng = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[0]
                            lat = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[1]
                            wire_x[i] = lng
                            wire_y[i] = lat
                        pd_final_table.loc[0:len(wire_x)-1, f"{wire}馈线维度lng"] = np.array(wire_x)
                        pd_final_table.loc[0:len(wire_y)-1, f"{wire}馈线经度lat"] = np.array(wire_y)
                    wire_cnt += 1
    pd_final_table.to_excel(writer, sheet_name=f"馈线经纬度数据")

    writer.save()
    writer.close()


def figure_data_3(figure_name):
    planning_object = load_variable("planning_object_20230417_100931.plaobj")
    GIS_object = load_variable(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\4.基于模糊综合评价理论的网格划分技术\rationality_evaluation_result_20230325_144634.gisobj")

    # # 获取各配变2016年和2017年365天的负荷值
    # # 读取负荷实际数据
    # _conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    # _cur = _conn.cursor()
    # _cur.execute('''select * from "入户点数据"''')
    # _conn.commit()
    # users = _cur.fetchall()
    # user_load_profile = np.zeros((len(users), 365))
    # # 先读取低压用户的
    # conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    # cur = conn.cursor()
    # cur.execute('''select * from "负荷数据表" where "年份" = 2016''')
    # conn.commit()
    # results = cur.fetchall()
    # for result in results:
    #     print(result[0])
    #     _cur.execute('''select * from "入户点数据" where "负荷名称" = ?''', (result[1].upper(),))
    #     _conn.commit()
    #     try:
    #         indexes = _cur.fetchall()
    #         load_profile_365 = np.array(result[33:33+365]) / 1000
    #         for index in indexes:
    #             user_load_profile[index[0], :] = load_profile_365
    #     except:
    #         pass
    # # 再读取中压用户的
    # conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\中压负荷数据表.db')
    # cur = conn.cursor()
    # cur.execute('''select * from "负荷数据表" where "年份" = 2016''')
    # conn.commit()
    # results = cur.fetchall()
    # for result in results:
    #     print(result[0])
    #     _cur.execute('''select * from "入户点数据" where "负荷名称" = ?''', (result[1].upper(),))
    #     _conn.commit()
    #     try:
    #         indexes = _cur.fetchall()
    #         load_profile_365 = np.array(result[33:33+365]) / 1000
    #         for index in indexes:
    #             user_load_profile[index[0], :] = load_profile_365
    #     except:
    #         pass
    #
    # planning_user_load_profile = np.zeros((len(planning_object.planning_user_name), 365))
    # for (id_user, user) in enumerate(planning_object.planning_user_name):
    #     if user.endswith('U'):
    #         user = user[0:-2]
    #     planning_user_load_profile[id_user, :] = user_load_profile[planning_object.user_names.index(user), :]
    #
    # planning_transformer_load_profile = np.zeros((len(planning_object.planning_transformer_name), 365))
    # for (id_transformer, transformer) in enumerate(planning_object.planning_transformer_name):
    #     load_profile = np.zeros(365)
    #     for user in planning_object.planning_transformer_users[transformer]:
    #         load_profile += planning_user_load_profile[planning_object.planning_user_name.index(user)]
    #     planning_transformer_load_profile[id_transformer] = load_profile
    # # 获取完毕：planning_transformer_load_profile
    # save_variable(planning_transformer_load_profile, "2016_planning_transformer_load_profile.np")
    #
    # # 读取负荷实际数据
    # _conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    # _cur = _conn.cursor()
    # _cur.execute('''select * from "入户点数据"''')
    # _conn.commit()
    # users = _cur.fetchall()
    # user_load_profile = np.zeros((len(users), 365))
    # # 先读取低压用户的
    # conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    # cur = conn.cursor()
    # cur.execute('''select * from "负荷数据表" where "年份" = 2017''')
    # conn.commit()
    # results = cur.fetchall()
    # for result in results:
    #     print(result[0])
    #     _cur.execute('''select * from "入户点数据" where "负荷名称" = ?''', (result[1].upper(),))
    #     _conn.commit()
    #     try:
    #         indexes = _cur.fetchall()
    #         load_profile_365 = np.array(result[33:33+365]) / 1000
    #         for index in indexes:
    #             user_load_profile[index[0], :] = load_profile_365
    #     except:
    #         pass
    # # 再读取中压用户的
    # conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\中压负荷数据表.db')
    # cur = conn.cursor()
    # cur.execute('''select * from "负荷数据表" where "年份" = 2017''')
    # conn.commit()
    # results = cur.fetchall()
    # for result in results:
    #     print(result[0])
    #     _cur.execute('''select * from "入户点数据" where "负荷名称" = ?''', (result[1].upper(),))
    #     _conn.commit()
    #     try:
    #         indexes = _cur.fetchall()
    #         load_profile_365 = np.array(result[33:33+365]) / 1000
    #         for index in indexes:
    #             user_load_profile[index[0], :] = load_profile_365
    #     except:
    #         pass
    #
    # planning_user_load_profile = np.zeros((len(planning_object.planning_user_name), 365))
    # for (id_user, user) in enumerate(planning_object.planning_user_name):
    #     if user.endswith('U'):
    #         user = user[0:-2]
    #     planning_user_load_profile[id_user, :] = user_load_profile[planning_object.user_names.index(user), :]
    #
    # planning_transformer_load_profile = np.zeros((len(planning_object.planning_transformer_name), 365))
    # for (id_transformer, transformer) in enumerate(planning_object.planning_transformer_name):
    #     load_profile = np.zeros(365)
    #     for user in planning_object.planning_transformer_users[transformer]:
    #         load_profile += planning_user_load_profile[planning_object.planning_user_name.index(user)]
    #     planning_transformer_load_profile[id_transformer] = load_profile
    # # 获取完毕：planning_transformer_load_profile
    # save_variable(planning_transformer_load_profile, "2017_planning_transformer_load_profile.np")
    planning_transformer_load_profile_2017 = load_variable("2017_planning_transformer_load_profile.np")
    planning_transformer_load_profile_2016 = load_variable("2016_planning_transformer_load_profile.np")
    planning_transformer_load_profile = load_variable("2018_planning_transformer_load_profile.np")


    # 由趋势外推计算配变2018年最大负荷，从而获取扩容量
    C_X = np.zeros(len(planning_object.planning_transformer_name))
    planning_transformer_2018_linear_load = np.zeros(len(planning_object.planning_transformer_name))
    for (id_transformer, transformer) in enumerate(planning_object.planning_transformer_name):
        planning_transformer_2018_linear_load[id_transformer] = np.max(planning_transformer_load_profile_2017[id_transformer, :]) + (np.max(planning_transformer_load_profile_2017[id_transformer, :]) - np.max(planning_transformer_load_profile_2016[id_transformer, :]))
        C_X[id_transformer] = planning_transformer_2018_linear_load[id_transformer]*1000 / planning_object.planning_transformer_cos_phi[id_transformer] - planning_object.planning_transformer_capacity_kVA[id_transformer]
        if C_X[id_transformer] < 0:
            C_X[id_transformer] = 0

    header = ["配变名称", "维度Lng", "经度Lat", "原始容量（kVA）", "趋势外推扩容增加（kVA）", "趋势外推扩容后容量（kVA）"]
    for i in range(365):
        header.extend([f"2018年第{i+1}天最大负荷（kW）", f"2018年第{i+1}天扩容前最大负载率", f"2018年第{i+1}天趋势外推扩容后最大负载率"])
    pd_final_table = pd.DataFrame(index=range(len(planning_object.planning_transformer_name)), columns=header)



    for (transformer_id, transformer) in enumerate(planning_object.planning_transformer_name):
        ts_data = planning_object.Transformers[planning_object.Transformer_names.index(transformer)]
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(ts_data[4], ts_data[5])
        lng = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[0]
        lat = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[1]
        content = [ts_data[1], lng, lat, ts_data[6], C_X[transformer_id], ts_data[6] + C_X[transformer_id]]

        for i in range(365):
            content.append(planning_transformer_load_profile[transformer_id, i]*1000)
            content.append(planning_transformer_load_profile[transformer_id, i]*1000 / planning_object.planning_transformer_cos_phi[transformer_id] / (ts_data[6]))
            content.append(planning_transformer_load_profile[transformer_id, i]*1000 / planning_object.planning_transformer_cos_phi[transformer_id] / (ts_data[6] + C_X[transformer_id]))
        pd_final_table.loc[transformer_id] = content
    pd_final_table.to_excel(figure_name)



if __name__ == '__main__':
    # figure_data_1("配变扩容优化结果.xlsx")
    # figure_data_3("配变扩容优化结果（趋势外推）.xlsx")
    figure_data_2("选线优化结果.xlsx")


