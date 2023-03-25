import sqlite3
import numpy as np
import pandas as pd
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import save_variable, load_variable


def figure_data_1(figure_name):
    header = ["负荷名称", "维度Lng", "经度Lat", "供电区域名称", "供电区域id", "供电分区名称", "供电分区id", "供电网格名称", "供电网格id", "供电单元名称", "供电单元id"]

    GIS_object = load_variable("rationality_evaluation_result_20230325_144634.gisobj")

    # 读取用户数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    users = cur.fetchall()

    pd_final_table = pd.DataFrame(index=range(len(users)), columns=header)

    contents = [[]]*len(users)
    user_ids = GIS_object.power_supply_area_map[:, 3].tolist()
    for (idx, user_id) in enumerate(user_ids):
        user_id = int(user_id)
        contents[user_id] = [0]*11
        contents[user_id][0] = users[user_id][1]
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(users[user_id][4], users[user_id][5])
        contents[user_id][1] = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[0]
        contents[user_id][2] = GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[1]
        contents[user_id][3] = GIS_object.power_supply_areas[int(GIS_object.power_supply_area_map[idx, 2])]
        contents[user_id][4] = int(GIS_object.power_supply_area_map[idx, 2])
    user_ids = GIS_object.power_supply_partition_map[:, 3].tolist()
    for (idx, user_id) in enumerate(user_ids):
        user_id = int(user_id)
        contents[user_id][5] = GIS_object.power_supply_partitions[int(GIS_object.power_supply_partition_map[idx, 2])]
        contents[user_id][6] = int(GIS_object.power_supply_partition_map[idx, 2])
    cnt = 0
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        user_ids = GIS_object.power_supply_mesh_map[idx_power_supply_partition][:, 3].tolist()
        for (idx, user_id) in enumerate(user_ids):
            user_id = int(user_id)
            contents[user_id][7] = GIS_object.power_supply_meshes[idx_power_supply_partition][int(GIS_object.power_supply_mesh_map[idx_power_supply_partition][idx, 2])]
            contents[user_id][8] = int(GIS_object.power_supply_mesh_map[idx_power_supply_partition][idx, 2]) + cnt
        cnt += len(GIS_object.power_supply_meshes[idx_power_supply_partition])
    cnt = 0
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        for (idx_power_supply_mesh, power_supply_mesh) in enumerate(GIS_object.power_supply_meshes[idx_power_supply_partition]):
            user_ids = GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][:, 3].tolist()
            for (idx, user_id) in enumerate(user_ids):
                user_id = int(user_id)
                contents[user_id][9] = GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh][int(GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][idx, 2])]
                contents[user_id][10] = int(GIS_object.power_supply_unit_map[idx_power_supply_partition][idx_power_supply_mesh][idx, 2]) + cnt
            cnt += len(GIS_object.power_supply_units[idx_power_supply_partition][idx_power_supply_mesh])
    for (idx, content) in enumerate(contents):
        pd_final_table.loc[idx] = content
    pd_final_table.to_excel(figure_name)


def figure_data_2(figure_name):
    header = ["名称", "维度Lng", "经度Lat", "类型代号", "id"]
    GIS_object = load_variable("rationality_evaluation_result_20230325_144634.gisobj")

    # 读取用户数据
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
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

    pd_final_table = pd.DataFrame(index=range(len(users)+len(TSs)+len(HV_stations)+len(Transformers)), columns=header)

    idx = 0
    for (user_id, user) in enumerate(users):
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(user[4], user[5])
        content = [user[1],
                   GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[0],
                   GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[1],
                   0,
                   user[0]]
        pd_final_table.loc[idx] = content
        idx += 1
    for (TS_id, TS) in enumerate(TSs):
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(TS[4], TS[5])
        content = [TS[1],
                   GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[0],
                   GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[1],
                   3,
                   TS[0]]
        pd_final_table.loc[idx] = content
        idx += 1
    for (HV_station_id, HV_station) in enumerate(HV_stations):
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(HV_station[4], HV_station[5])
        content = [HV_station[1],
                   GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[0],
                   GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[1],
                   2,
                   HV_station[0]]
        pd_final_table.loc[idx] = content
        idx += 1
    for (Transformer_id, Transformer) in enumerate(Transformers):
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(Transformer[4], Transformer[5])
        content = [Transformer[1],
                   GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[0],
                   GIS_object.CRS_transformer.mercator2lnglat.transform(x, y)[1],
                   1,
                   Transformer[0]]
        pd_final_table.loc[idx] = content
        idx += 1
    pd_final_table.to_excel(figure_name)


def figure_data_3(figure_name):
    header = ["名称", "类型", "得分"]
    for i in range(6):
        header.append(f"指标{i+1}得分")
        header.append(f"指标{i+1}权重")
    GIS_object = load_variable("rationality_evaluation_result_20230325_144634.gisobj")

    cnt = 1
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        for (idx_power_supply_mesh, power_supply_mesh) in enumerate(GIS_object.power_supply_meshes[idx_power_supply_partition]):
            cnt += 1

    pd_final_table = pd.DataFrame(index=range(cnt), columns=header)

    idx = 0
    content = ["整体", "供电分区划分合理性", GIS_object.score_of_partition*100,
               GIS_object.a1, GIS_object.weight_of_partition_index[0],
               GIS_object.a2, GIS_object.weight_of_partition_index[1],
               GIS_object.a3, GIS_object.weight_of_partition_index[2]]
    content.extend(["NaN"]*6)
    pd_final_table.loc[idx] = content
    idx += 1
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        content = [power_supply_partition, "供电网格划分合理性", GIS_object.score_of_mesh[idx_power_supply_partition]*100,
                   GIS_object.b1[idx_power_supply_partition], GIS_object.weight_of_mesh_index[0],
                   GIS_object.b2[idx_power_supply_partition], GIS_object.weight_of_mesh_index[1],
                   GIS_object.b3[idx_power_supply_partition], GIS_object.weight_of_mesh_index[2],
                   GIS_object.b4[idx_power_supply_partition], GIS_object.weight_of_mesh_index[3],
                   GIS_object.b5[idx_power_supply_partition], GIS_object.weight_of_mesh_index[4],
                   GIS_object.b6[idx_power_supply_partition], GIS_object.weight_of_mesh_index[5]]
        pd_final_table.loc[idx] = content
        idx += 1
    for (idx_power_supply_partition, power_supply_partition) in enumerate(GIS_object.power_supply_partitions):
        for (idx_power_supply_mesh, power_supply_mesh) in enumerate(GIS_object.power_supply_meshes[idx_power_supply_partition]):
            content = [power_supply_mesh, "供电单元划分合理性", GIS_object.score_of_unit[idx_power_supply_partition][idx_power_supply_mesh]*100,
                       GIS_object.c1[idx_power_supply_partition][idx_power_supply_mesh], GIS_object.weight_of_unit_index[0],
                       GIS_object.c2[idx_power_supply_partition][idx_power_supply_mesh], GIS_object.weight_of_unit_index[1],
                       GIS_object.c3[idx_power_supply_partition][idx_power_supply_mesh], GIS_object.weight_of_unit_index[2],
                       GIS_object.c4[idx_power_supply_partition][idx_power_supply_mesh], GIS_object.weight_of_unit_index[3],
                       GIS_object.c5[idx_power_supply_partition][idx_power_supply_mesh], GIS_object.weight_of_unit_index[4]]
            content.extend(["NaN"]*2)
            pd_final_table.loc[idx] = content
            idx += 1
    pd_final_table.to_excel(figure_name)


if __name__ == '__main__':
    # figure_data_1("网格化划分结果.xlsx")
    # figure_data_2("供电所3、高压变电站2、配变1、终端用户0的地理坐标.xlsx")
    figure_data_3("网格划分合理性评估体系评估结果.xlsx")


