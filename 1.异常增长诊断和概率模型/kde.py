import sqlite3
import numpy as np
from AE_evaluate import get_autoencoder1, evaluate, save_variable, load_variable
import vegas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut


# 生成样本矩阵
def get_sample_matrix():
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    data_len = 70407
    # 样本矩阵
    _sample_matrix = np.zeros((data_len, 12))
    # 数据库
    conn = sqlite3.connect(r'D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\0.数据集清洗\负荷数据表.db')
    cur = conn.cursor()
    # 获取自编码器模型
    auto_encoder = get_autoencoder1("AutoEncoder_20230118_213844.path")
    # 获取数据
    cur.execute('''select * from "负荷数据表" where "年份" = 2017 ''')
    conn.commit()
    results = cur.fetchall()
    for idx, result in enumerate(results):
        sample = np.array(result[33:33+365]) / 1000
        # 输入模型
        normal_component, abnormal_component, mse = evaluate(_auto_encoder=auto_encoder, _sample=sample)
        for index, month in enumerate(months):
            start = 0
            for month_before in range(1, month):
                start += days[month_before]
            end = start + days[month]
            _sample_matrix[idx, index] = np.max(abnormal_component[start:end])
    return _sample_matrix


# 窗宽计算
def h_optimizer(_sample_matrix):
    # 窗宽h的计算（https://www.zhihu.com/question/27301358）
    n = np.size(_sample_matrix, 0)
    ave = np.average(_sample_matrix, axis=0)
    ave = np.tile(ave, (n, 1))
    s = np.sqrt(np.sum(np.linalg.norm(_sample_matrix - ave, axis=1)) / (n - 1))
    _h = 1.05 * s * np.power(n, -0.2)
    return _h


# 计算异常增长分量处于某个区间的概率
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
    result = vegas_integrator(_kde, nitn=10, neval=200)
    # print(result.summary())
    return result.mean, _h


# 获取某个概率对应的异常增长分量（此处概率的定义是：假如负荷增长，则增长超过某个量的概率，属于条件概率）
def find_abnormal_component(_probability, _sample_matrix):
    dim = np.size(_sample_matrix, 1)
    _xl = -np.ones(dim)*0
    _xu = np.ones(dim)*0.1
    _xl_right = np.ones(dim)*0.1
    _xl_left = -np.ones(dim)*0.1
    for idx in range(dim):
        print(f"正在计算{idx+1}月的{_probability*100}%概率区间...")
        # 先求增长的概率
        _p_increase, _h = probability(_xl=np.array([0]), _xu=np.array([0.1]), _sample_matrix=np.reshape(_sample_matrix[:, idx], (-1, 1)))
        # 用二分法求解
        _p, _h = probability(_xl=np.array([_xl[idx]]), _xu=np.array([_xu[idx]]), _sample_matrix=np.reshape(_sample_matrix[:, idx], (-1, 1)))
        while np.fabs(_p/_p_increase - _probability) > 5E-2:
            if _p/_p_increase < _probability:
                _xl_right[idx] = _xl[idx]
            elif _p/_p_increase > _probability:
                _xl_left[idx] = _xl[idx]
            else:
                break
            _xl[idx] = 0.5 * (_xl_right[idx] + _xl_left[idx])
            _p, _h = probability(_xl=np.array([_xl[idx]]), _xu=np.array([_xu[idx]]), _sample_matrix=np.reshape(_sample_matrix[:, idx], (-1, 1)))
            print(_p/_p_increase - _probability)
    return _xl


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
    # 计算90%、60%、30%概率的异常增长值和概率密度函数图像
    print(f"正在计算概率区间...")
    # 给出90%、60%、30%概率的异常增长值
    abnormal_90 = find_abnormal_component(_probability=0.9, _sample_matrix=sample_matrix)
    save_variable(abnormal_90, "abnormal_90.kde")
    abnormal_60 = find_abnormal_component(_probability=0.6, _sample_matrix=sample_matrix)
    save_variable(abnormal_60, "abnormal_60.kde")
    abnormal_30 = find_abnormal_component(_probability=0.3, _sample_matrix=sample_matrix)
    save_variable(abnormal_30, "abnormal_30.kde")
    abnormal_90 = load_variable("abnormal_90.kde")
    abnormal_60 = load_variable("abnormal_60.kde")
    abnormal_30 = load_variable("abnormal_30.kde")
    sns.lineplot(x=range(1, 13), y=[0]*12)
    sns.lineplot(x=range(1, 13), y=abnormal_90)
    sns.lineplot(x=range(1, 13), y=abnormal_60)
    sns.lineplot(x=range(1, 13), y=abnormal_30)
    plt.show()
    print(f"正在计算概率密度函数...")
    # 给出概率密度函数图像
    for idx in range(12):
        x = np.arange(-0.05, 0.05, 0.0005)
        y = np.zeros_like(x)
        for i, xi in enumerate(x):
            # 计算窗宽h
            h = h_optimizer(_sample_matrix=np.reshape(sample_matrix[:, idx], (-1, 1)))
            y[i] = kde(_sample=xi, _sample_matrix=np.reshape(sample_matrix[:, idx], (-1, 1)), _h=h)
        sns.lineplot(x=x, y=y)
    plt.show()
