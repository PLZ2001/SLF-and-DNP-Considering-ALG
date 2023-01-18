import sqlite3
import numpy as np
from evaluate import get_autoencoder, evaluate, save_variable, load_variable
import vegas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut


def get_sample_matrix():
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    data_len = 3040
    # 样本矩阵
    _sample_matrix = np.zeros((data_len, 12))
    # 数据库
    conn = sqlite3.connect('负荷数据表（微型）.db')
    cur = conn.cursor()
    # 获取自编码器模型
    auto_encoder = get_autoencoder("AutoEncoder_20230117_132712.path")
    # 获取数据
    cur.execute('''select * from "负荷数据表" where "年份" = 2016 ''')
    conn.commit()
    results = cur.fetchall()
    for idx, result in enumerate(results):
        sample = np.array(result[34:34+365]) / 1000
        # 输入模型
        normal_component, abnormal_component, mse = evaluate(_auto_encoder=auto_encoder, _sample=sample)
        for index, month in enumerate(months):
            start = 0
            for month_before in range(1, month):
                start += days[month_before]
            end = start + days[month]
            _sample_matrix[idx, index] = np.max(abnormal_component[start:end])
    return _sample_matrix


def h_optimizer(_sample_matrix):
    # 窗宽h的计算（https://www.zhihu.com/question/27301358）
    n = np.size(_sample_matrix, 0)
    ave = np.average(_sample_matrix, axis=0)
    ave = np.tile(ave, (n, 1))
    s = np.sqrt(np.sum(np.linalg.norm(_sample_matrix - ave, axis=1)) / (n - 1))
    _h = 1.05 * s * np.power(n, -0.2)
    return _h


def probability(_xl, _xu, _sample_matrix):
    # _xl为下界，_xu为上界
    def _kde(_sample):
        return kde(_sample=_sample, _sample_matrix=_sample_matrix, _h=_h)
        # return kde_new(_sample=_sample, _sample_matrix=_sample_matrix, _h=_h)
    # 计算窗宽h
    _h = h_optimizer(_sample_matrix=_sample_matrix)
    # print(f"h取{_h}")
    # 利用蒙特卡洛高维积分，求区间之间的概率积分
    l_u = []
    for i in range(len(_xl)):
        l_u.append([_xl[i], _xu[i]])
    vegas_integrator = vegas.Integrator(l_u)
    result = vegas_integrator(_kde, nitn=10, neval=1e3)
    # print(result.summary())
    return result.mean, _h


def kde(_sample, _sample_matrix, _h):
    n = np.size(_sample_matrix, 0)
    # 复制扩充
    _sample = np.tile(np.reshape(_sample, (1, -1)), (n, 1))
    # 基于柯西核的多维核密度估计
    # k = 1 / (np.pi * (np.power(np.linalg.norm((_sample - _sample_matrix) / _h, ord=2, axis=1), 2) + 1))
    # 基于高斯核的多维核密度估计
    k = np.exp(-np.power(np.linalg.norm((_sample - _sample_matrix) / _h, axis=1), 2) / 2) / np.sqrt(2 * np.pi)
    return np.average(k / _h)


if __name__ == '__main__':
    # print("正在生成异常增长样本矩阵...")
    # sample_matrix = get_sample_matrix()
    # print("正在保存异常增长样本矩阵...")
    # save_variable(sample_matrix, "sample_matrix.kde")
    print("正在读取异常增长样本矩阵...")
    sample_matrix = load_variable("sample_matrix.kde")
    # sample_matrix的结构必须是行为样本，列为维度
    # 分别每个月计算核密度估计结果，并给出区间概率和概率密度函数图像
    for idx in range(12):
        print(f"正在计算{idx+1}月...")
        # 给出区间概率
        xl = -np.ones(1)*0
        xu = np.ones(1)*0.015
        p, h = probability(_xl=xl, _xu=xu, _sample_matrix=np.reshape(sample_matrix[:, idx], (-1, 1)))
        print(f"{xl}至{xu}的概率是{p*100:2f}%")
        # 给出概率密度函数图像
        x = np.arange(-0.05, 0.05, 0.0005)
        y = np.zeros_like(x)
        for i, xi in enumerate(x):
            y[i] = kde(_sample=xi, _sample_matrix=np.reshape(sample_matrix[:, idx], (-1, 1)), _h=h)
        sns.lineplot(x=x, y=y)
    plt.show()