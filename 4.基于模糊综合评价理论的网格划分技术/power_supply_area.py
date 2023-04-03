# 划分原则：根据负荷密度进行供电区域划分，要求区域具有凝聚性而不能四处分散

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from GIS_object import GISObject, get_block_index
import matplotlib.pyplot as plt
import datetime
import sqlite3
import sys
sys.path.append(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型")
from AE_evaluate import save_variable, load_variable


def find_nearest_power_supply_area(GIS_object, block_index_x, block_index_y):
    power_supply_area = []
    try_times = 10
    for try_time in range(try_times):
        delta_x_group = range(-(try_time + 1), (try_time + 2))
        delta_y_group = range(-(try_time + 1), (try_time + 2))
        delta_xs = []
        delta_ys = []
        for i in range(len(delta_x_group)):
            for j in range(len(delta_y_group)):
                delta_xs.append(delta_x_group[i])
                delta_ys.append(delta_y_group[j])
        for delta_x in delta_xs:
            for delta_y in delta_ys:
                if 0<=block_index_x+delta_x<GIS_object.horizontal_block_num and 0<=block_index_y+delta_y<GIS_object.vertical_block_num:
                    for (idx_power_supply_area_unit, power_supply_area_unit) in enumerate(GIS_object.power_supply_areas_unit):
                        if GIS_object.power_supply_area_optimization[idx_power_supply_area_unit, block_index_x+delta_x, block_index_y+delta_y] == 1:
                            power_supply_area.append(power_supply_area_unit)
                    if len(power_supply_area) > 0:
                        return power_supply_area
    return power_supply_area


def print_power_supply_area(GIS_object):
    x = GIS_object.power_supply_area_map[:, 0]
    y = GIS_object.power_supply_area_map[:, 1]
    power_supply_area = GIS_object.power_supply_area_map[:, 2]
    plt.scatter(x, y, s=1, c=power_supply_area, cmap="viridis")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    GIS_object = GISObject(horizontal_block_num=75, vertical_block_num=75)
    GIS_object.get_all_load_density()
    # GIS_object.get_all_saturated_load_density()
    GIS_object.calculate_power_supply_area_score()
    GIS_object.get_user_load_profile()

    print(f"最大负荷密度为 {np.max(GIS_object.load_density)} W/m^2")
    print(f"最小负荷密度为 {np.min(GIS_object.load_density)} W/m^2")
    recommend = "None"
    for (idx, power_supply_area) in enumerate(GIS_object.power_supply_areas):
        if GIS_object.fuzzy_area[power_supply_area][1] <= np.max(GIS_object.load_density):
            recommend = power_supply_area
            break
    print(f"建议供电区域最高等级取为{recommend}")

    # Create a new model
    model = gp.Model("POWER_SUPPLY_AREA")

    # 供电区域类型作为决策量
    X = model.addMVar((len(GIS_object.power_supply_areas), GIS_object.horizontal_block_num, GIS_object.vertical_block_num), vtype=GRB.BINARY, name="power_supply_area")
    # 总目标函数
    total_score = 0
    for x in range(GIS_object.horizontal_block_num):
        for y in range(GIS_object.vertical_block_num):
            for (idx, power_supply_area) in enumerate(GIS_object.power_supply_areas):
                if GIS_object.power_supply_area_score[idx, x, y] > 0:
                    total_score += GIS_object.power_supply_area_score[idx, x, y] * X[idx, x, y]
    model.setObjective(total_score, GRB.MAXIMIZE)

    # 添加约束
    # 约束1：每个网格只能是一种供电区域类型
    for x in range(GIS_object.horizontal_block_num):
        for y in range(GIS_object.vertical_block_num):
            cnt = 0
            for (idx, power_supply_area) in enumerate(GIS_object.power_supply_areas):
                cnt += X[idx, x, y]
            model.addConstr(cnt <= 1.1)
            model.addConstr(0.9 <= cnt)

    # 约束2：凝聚性约束
    for (idx, power_supply_area) in enumerate(GIS_object.power_supply_areas):
        for x in range(GIS_object.horizontal_block_num):
            for y in range(GIS_object.vertical_block_num):
                cnt = 0
                # for x_, y_ in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x - 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1)]:
                for x_, y_ in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= x_ < GIS_object.horizontal_block_num and 0 <= y_ < GIS_object.vertical_block_num:
                        cnt += (X[idx, x_, y_] - X[idx, x, y])
                model.addConstr(cnt >= -2)
    for (idx, power_supply_area) in enumerate(GIS_object.power_supply_areas):
        for x in range(GIS_object.horizontal_block_num):
            for y in range(GIS_object.vertical_block_num):
                cnt = 0
                # for x_, y_ in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x - 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1)]:
                for x_, y_ in [(x - 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1)]:
                    if 0 <= x_ < GIS_object.horizontal_block_num and 0 <= y_ < GIS_object.vertical_block_num:
                        cnt += (X[idx, x_, y_] - X[idx, x, y])
                model.addConstr(cnt >= -2)

    # 约束3：要求建议供电区域面积≥0，非建议供电区域的面积则为0
    flag = False
    for (idx, power_supply_area) in enumerate(GIS_object.power_supply_areas):
        if power_supply_area == recommend or flag is True:
            flag = True
            cnt = 0
            for x in range(GIS_object.horizontal_block_num):
                for y in range(GIS_object.vertical_block_num):
                    cnt += X[idx, x, y]
            # model.addConstr(cnt >= 5000000.0 / (GIS_object.block_width * GIS_object.block_height))
            model.addConstr(cnt >= 0)
        else:
            cnt = 0
            for x in range(GIS_object.horizontal_block_num):
                for y in range(GIS_object.vertical_block_num):
                    cnt += X[idx, x, y]
            model.addConstr(cnt <= 0.1)



    # # 约束2：每块供电区域只能有1个
    # for (idx, power_supply_area) in enumerate(GIS_object.power_supply_areas):
    #     num = num_of_island(X[power_supply_area], GIS_object.horizontal_block_num, GIS_object.vertical_block_num)
    #     model.addConstr(num == 1)

    # 优化
    model.Params.MIPGap = 0.0005
    model.Params.TimeLimit = 300
    model.optimize()

    # 获取结果
    print('Obj:', model.objVal)
    GIS_object.power_supply_area_optimization = X.X

    print("正在生成供电区域网格。。。")
    # 给供电区域网格赋值
    GIS_object.power_supply_areas_unit = ["A_plus", "A", "B", "C", "D", "E"]
    GIS_object.power_supply_areas = []
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\3.数据集清洗（续）\规划数据表.db')
    cur = conn.cursor()
    cur.execute('''select * from "入户点数据"''')
    conn.commit()
    results = cur.fetchall()
    # 先统计一共有几种供电区域
    for (idx, result) in enumerate(results):
        power_supply_area = []
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(result[4], result[5])
        block_index_x, block_index_y = get_block_index(x=x, y=y,
                                                       left_bottom_x=GIS_object.left_bottom_x,
                                                       left_bottom_y=GIS_object.left_bottom_y,
                                                       right_top_x=GIS_object.right_top_x,
                                                       right_top_y=GIS_object.right_top_y,
                                                       horizontal_block_num=GIS_object.horizontal_block_num,
                                                       vertical_block_num=GIS_object.vertical_block_num)
        for (idx_power_supply_area_unit, power_supply_area_unit) in enumerate(GIS_object.power_supply_areas_unit):
            if GIS_object.power_supply_area_optimization[idx_power_supply_area_unit, block_index_x, block_index_y] == 1:
                power_supply_area.append(power_supply_area_unit)
        if len(power_supply_area) == 0:
            power_supply_area = find_nearest_power_supply_area(GIS_object, block_index_x, block_index_y)
        power_supply_area = "+".join(power_supply_area)
        if power_supply_area not in GIS_object.power_supply_areas:
            GIS_object.power_supply_areas.append(power_supply_area)
    # 供电区域
    GIS_object.power_supply_area_map = np.zeros((len(results), 4))
    for (idx,result) in enumerate(results):
        x, y = GIS_object.CRS_transformer.gisxy2mercator.transform(result[4], result[5])
        GIS_object.power_supply_area_map[idx, 0] = x
        GIS_object.power_supply_area_map[idx, 1] = y
        GIS_object.power_supply_area_map[idx, 3] = idx
        power_supply_area = []
        block_index_x, block_index_y = get_block_index(x=x, y=y,
                                                       left_bottom_x=GIS_object.left_bottom_x,
                                                       left_bottom_y=GIS_object.left_bottom_y,
                                                       right_top_x=GIS_object.right_top_x,
                                                       right_top_y=GIS_object.right_top_y,
                                                       horizontal_block_num=GIS_object.horizontal_block_num,
                                                       vertical_block_num=GIS_object.vertical_block_num)
        for (idx_power_supply_area_unit, power_supply_area_unit) in enumerate(GIS_object.power_supply_areas_unit):
            if GIS_object.power_supply_area_optimization[idx_power_supply_area_unit, block_index_x, block_index_y] == 1:
                power_supply_area.append(power_supply_area_unit)
        if len(power_supply_area) == 0:
            power_supply_area = find_nearest_power_supply_area(GIS_object, block_index_x, block_index_y)
        power_supply_area = "+".join(power_supply_area)
        GIS_object.power_supply_area_map[idx, 2] = GIS_object.power_supply_areas.index(power_supply_area)

    print_power_supply_area(GIS_object)
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_variable(GIS_object, f"power_supply_area_optimization_{date}.gisobj")




