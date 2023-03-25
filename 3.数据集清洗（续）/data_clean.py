import math

import pandas as pd
import shapefile

# 根据SMART-DS数据集，汇总出 线路数据.csv
# 表头：线路名称 起始点名称 中止点名称 长度（km） 所属馈线 电压等级（kV） 电压级别 相 R R0 L L0 C C0 bbox_lx bbox_ly bbox_rx bbox_ry points_x points_y

# 根据SMART-DS数据集，汇总出 供电所数据.csv
# 表头：变电站名称 x y

# 根据SMART-DS数据集，汇总出 高压变电站数据.csv
# 表头：变电站名称 x y

# 根据SMART-DS数据集，汇总出 配电变电站数据.csv
# 表头：变电站名称 x y 容量（kVA） 相数

# 根据SMART-DS数据集，汇总出 入户点数据.csv
# 表头：负荷名称 x y 电压等级 相数 功率因数 所属馈线

# 根据SMART-DS数据集，汇总出 斯坦纳点数据.csv
# 表头：斯坦纳点名称 x y

if __name__ == '__main__':
    areas = ["GSO"]
    sub_areas = ["industrial", "rural", "urban-suburban"]

    # header = ["线路名称", "区域", "子区域", "起始点名称", "中止点名称", "长度（km）", "所属馈线", "电压等级（kV）", "电压级别", "相", "R", "R0", "X", "X0", "C", "C0", "左下角x", "左下角y", "右上角x", "右上角y", "节点x", "节点y"]
    # final_table = pd.DataFrame(columns=header)
    # for area in areas:
    #     for sub_area in sub_areas:
    #         base_path = fr"GIS\{area}\{sub_area}\Line_N"
    #         print(f"正在读取{area}地区{sub_area}子地区的线路数据...")
    #
    #         sf = shapefile.Reader(base_path)
    #         shapes = sf.shapes()
    #         records = sf.records()
    #         for i in range(len(shapes)):
    #             print(f"{i}/{len(shapes)}")
    #             if records[i].Status == 1:
    #                 content = [records[i].Code,
    #                            area,
    #                            sub_area,
    #                            records[i].NodeA,
    #                            records[i].NodeB,
    #                            records[i].Len,
    #                            records[i].Feeder.replace(" -> ", "--").lower(),
    #                            records[i].NomV,
    #                            records[i].PhasesV[-2:],
    #                            records[i].Phases,
    #                            records[i].R,
    #                            records[i].R0,
    #                            records[i].X,
    #                            records[i].X0,
    #                            records[i].C,
    #                            records[i].C0,
    #                            shapes[i].bbox[0],
    #                            shapes[i].bbox[1],
    #                            shapes[i].bbox[2],
    #                            shapes[i].bbox[3],
    #                            [],
    #                            []]
    #                 for point in shapes[i].points:
    #                     content[-2].append(str(point[0]))
    #                     content[-1].append(str(point[1]))
    #                 content[-2] = ",".join(content[-2])
    #                 content[-1] = ",".join(content[-1])
    #
    #                 final_table.loc[len(final_table)] = content
    # final_table.to_csv("线路数据.csv")

    # header = ["变电站名称", "区域", "子区域", "x", "y"]
    # final_table = pd.DataFrame(columns=header)
    # for area in areas:
    #     for sub_area in sub_areas:
    #         base_path = fr"GIS\{area}\{sub_area}\TransSubstation_N"
    #         print(f"正在读取{area}地区{sub_area}子地区的供电所数据...")
    #
    #         sf = shapefile.Reader(base_path)
    #         shapes = sf.shapes()
    #         records = sf.records()
    #         for i in range(len(shapes)):
    #             print(f"{i}/{len(shapes)}")
    #             content = [records[i].Node,
    #                        area,
    #                        sub_area,
    #                        shapes[i].points[0][0],
    #                        shapes[i].points[0][1]]
    #             final_table.loc[len(final_table)] = content
    # final_table.to_csv("供电所数据.csv")

    # header = ["变电站名称", "区域", "子区域", "x", "y"]
    # final_table = pd.DataFrame(columns=header)
    # for area in areas:
    #     for sub_area in sub_areas:
    #         base_path = fr"GIS\{area}\{sub_area}\HVMVSubstation_N"
    #         print(f"正在读取{area}地区{sub_area}子地区的高压变电站数据...")
    #
    #         sf = shapefile.Reader(base_path)
    #         shapes = sf.shapes()
    #         records = sf.records()
    #         for i in range(len(shapes)):
    #             print(f"{i}/{len(shapes)}")
    #             content = [records[i].Node,
    #                        area,
    #                        sub_area,
    #                        shapes[i].points[0][0],
    #                        shapes[i].points[0][1]]
    #             final_table.loc[len(final_table)] = content
    # final_table.to_csv("高压变电站数据.csv")
    #
    # header = ["变电站名称", "区域", "子区域", "x", "y", "容量（kVA）", "相数"]
    # final_table = pd.DataFrame(columns=header)
    # for area in areas:
    #     for sub_area in sub_areas:
    #         base_path = fr"GIS\{area}\{sub_area}\DistribTransf_N"
    #         print(f"正在读取{area}地区{sub_area}子地区的配电变电站数据...")
    #
    #         sf = shapefile.Reader(base_path)
    #         shapes = sf.shapes()
    #         records = sf.records()
    #         for i in range(len(shapes)):
    #             print(f"{i}/{len(shapes)}")
    #             content = [records[i].Node,
    #                        area,
    #                        sub_area,
    #                        shapes[i].points[0][0],
    #                        shapes[i].points[0][1],
    #                        records[i].Size_kVA,
    #                        records[i].Phases]
    #             final_table.loc[len(final_table)] = content
    # final_table.to_csv("配电变电站数据.csv")
    #
    # header = ["负荷名称", "区域", "子区域", "x", "y", "电压等级", "相数", "功率因数", "所属馈线"]
    # final_table = pd.DataFrame(columns=header)
    # for area in areas:
    #     for sub_area in sub_areas:
    #         base_path = fr"GIS\{area}\{sub_area}\NewConsumerGreenfield_N"
    #         print(f"正在读取{area}地区{sub_area}子地区的入户点数据数据...")
    #
    #         sf = shapefile.Reader(base_path)
    #         shapes = sf.shapes()
    #         records = sf.records()
    #         for i in range(len(shapes)):
    #             print(f"{i}/{len(shapes)}")
    #             content = [records[i].Code,
    #                        area,
    #                        sub_area,
    #                        shapes[i].points[0][0],
    #                        shapes[i].points[0][1],
    #                        records[i].Code[3:5],
    #                        records[i].Phases,
    #                        records[i].DemP_kW / math.sqrt(records[i].DemP_kW*records[i].DemP_kW+records[i].DemQ_kVAr*records[i].DemQ_kVAr),
    #                        records[i].Feeder.replace(" -> ", "--").lower()]
    #             final_table.loc[len(final_table)] = content
    # final_table.to_csv("入户点数据.csv")

    header = ["斯坦纳点名称", "区域", "子区域", "x", "y"]
    final_table = pd.DataFrame(columns=header)
    for area in areas:
        for sub_area in sub_areas:
            base_path = fr"GIS\{area}\{sub_area}\DummyEquip"
            print(f"正在读取{area}地区{sub_area}子地区的斯坦纳点数据...")

            sf = shapefile.Reader(base_path)
            shapes = sf.shapes()
            records = sf.records()
            for i in range(len(shapes)):
                print(f"{i}/{len(shapes)}")
                if records[i].Node.find("DM") != -1:
                    content = [records[i].Node,
                               area,
                               sub_area,
                               shapes[i].points[0][0],
                               shapes[i].points[0][1]]
                    final_table.loc[len(final_table)] = content
    final_table.to_csv("斯坦纳点数据.csv")
