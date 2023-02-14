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
    auto_encoder = get_autoencoder1("AutoEncoder_20230125_123858.path")
    # 获取数据
    cur.execute('''select * from "负荷数据表" where "年份" = 2016 OR "年份" = 2017 ''')
    conn.commit()
    results = cur.fetchall()
    for idx in range(data_len):
        print(idx)
        increment = np.array(results[idx+data_len][33:33+365]) / 1000 - np.array(results[idx][33:33+365]) / 1000
        # 输入模型
        normal_increment, abnormal_increment, mse = evaluate(_auto_encoder=auto_encoder, _increment=increment)
        for index, month in enumerate(months):
            start = 0
            for month_before in range(1, month):
                start += days[month_before]
            end = start + days[month]
            # 除以基值变为百分比
            _sample_matrix[idx, index] = np.max(abnormal_increment[start:end]) / np.max((np.array(results[idx][33:33+365]) / 1000)[start:end])
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


# 生成概率密度函数
def generate_pdf(_sample_matrix):
    # dim = np.size(_sample_matrix, 1)
    # _x = np.arange(-1.0, 4.0, 0.0001)
    # _pdf = np.zeros((12, len(_x)))
    # for idx in range(dim):
    #     print(f"正在生成{idx+1}月的概率密度函数..")
    #     _h = h_optimizer(_sample_matrix=np.reshape(_sample_matrix[:, idx], (-1, 1)))
    #     for i in range(len(_x)):
    #         _pdf[idx, i] = kde(_x[i], np.reshape(_sample_matrix[:, idx], (-1, 1)), _h)
    # save_variable(_x, "x.pdf")
    # save_variable(_pdf, "pdf.pdf")
    _x = load_variable(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\x.pdf")
    _pdf = load_variable(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\pdf.pdf")
    print("概率密度函数生成完毕")
    # dim = np.size(_sample_matrix, 1)
    # _x = np.arange(-1.0, 4.0, 0.0001)
    # _cdf = np.zeros((12, len(_x)))
    # for idx in range(dim):
    #     print(f"正在生成{idx+1}月的累积分布函数..")
    #     sum = 0
    #     for i in range(len(_x)):
    #         sum += _pdf[idx, i] * 0.0001
    #         _cdf[idx, i] = sum
    # save_variable(_cdf, "cdf.cdf")
    _cdf = load_variable(r"D:\OneDrive\桌面\毕设\代码\计及负荷异常增长的空间负荷预测与配电网规划\1.异常增长诊断和概率模型\cdf.cdf")
    print("累积分布函数生成完毕")
    return _x, _pdf, _cdf

# 生成概率区间
def generate_probability_range(_sample_matrix):
    def next_l_u(_l, _u):
        __l = _l
        __u = _u
        _l += 1
        _u -= 1
        if _u <= _l:
            return False, 0, 0
        while True:
            if cdf[_l] > 1 - cdf[_u]:
                _u -= 1
                if _u <= _l:
                    return False, 0, 0
                if cdf[_l] <= 1 - cdf[_u]:
                    return True, _l, _u
            if cdf[_l] < 1 - cdf[_u]:
                _l += 1
                if _u <= _l:
                    return False, 0, 0
                if cdf[_l] >= 1 - cdf[_u]:
                    return True, _l, _u

    _x, _pdf, _cdf = generate_pdf(_sample_matrix)

    dim = np.size(_sample_matrix, 1)
    _xl = np.zeros((dim, len(_x)))
    _xu = np.zeros((dim, len(_x)))
    _p_range = np.zeros((dim, len(_x)))
    for idx in range(dim):
        # print(f"正在生成{idx+1}月的概率区间..")
        cdf = _cdf[idx, :]
        l = -1
        u = len(_x)
        success, l, u = next_l_u(l, u)
        cnt = 0
        _p = 0
        while True:
            _xl[idx, cnt] = _x[l]
            _xu[idx, cnt] = _x[u]
            _p_range[idx, cnt] = cdf[u] - cdf[l]
            success, l, u = next_l_u(l, u)
            if not success:
                break
            cnt += 1
    # save_variable(_xl, "xl.p_range")
    # save_variable(_xu, "xu.p_range")
    # save_variable(_p_range, "p_range.p_range")
    # _xl = load_variable("xl.p_range")
    # _xu = load_variable("xu.p_range")
    # _p_range = load_variable("p_range.p_range")
    return _xl, _xu, _p_range


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


# 获取某个概率对应的异常增长分量
def find_abnormal_increment(_probability, _sample_matrix):
    dim = np.size(_sample_matrix, 1)
    _xl, _xu, _p_range = generate_probability_range(_sample_matrix)
    xl = np.zeros(dim)
    xu = np.zeros(dim)
    for idx in range(dim):
        print(f"正在计算{idx+1}月的{_probability*100}%概率区间...")
        for (i, p_range) in enumerate(_p_range[idx, :]):
            if p_range <= (1-_probability):
                xl[idx] = _xl[idx, i]
                xu[idx] = _xu[idx, i]
                break
    return [xl, xu]


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

    print(f"正在计算概率密度函数...")
    # 给出概率密度函数图像
    x, pdf, cdf = generate_pdf(_sample_matrix=sample_matrix)
    for idx in range(12):
        sns.lineplot(x=x, y=pdf[idx, :])
    plt.show()

    # 计算90%、60%、30%概率的异常增长值和概率密度函数图像
    print(f"正在计算概率区间...")
    # 给出90%、60%、30%概率的异常增长值
    abnormal_90 = find_abnormal_increment(_probability=0.9, _sample_matrix=sample_matrix)
    # save_variable(abnormal_90, "abnormal_90.kde")
    abnormal_60 = find_abnormal_increment(_probability=0.6, _sample_matrix=sample_matrix)
    # save_variable(abnormal_60, "abnormal_60.kde")
    abnormal_30 = find_abnormal_increment(_probability=0.3, _sample_matrix=sample_matrix)
    # save_variable(abnormal_30, "abnormal_30.kde")
    abnormal_10 = find_abnormal_increment(_probability=0.1, _sample_matrix=sample_matrix)
    # save_variable(abnormal_10, "abnormal_10.kde")
    # abnormal_90 = load_variable("abnormal_90.kde")
    # abnormal_60 = load_variable("abnormal_60.kde")
    # abnormal_30 = load_variable("abnormal_30.kde")
    # abnormal_10 = load_variable("abnormal_10.kde")
    sns.lineplot(x=range(1, 13), y=[0]*12)
    sns.lineplot(x=range(1, 13), y=abnormal_90[0])
    sns.lineplot(x=range(1, 13), y=abnormal_90[1])
    sns.lineplot(x=range(1, 13), y=abnormal_60[0])
    sns.lineplot(x=range(1, 13), y=abnormal_60[1])
    sns.lineplot(x=range(1, 13), y=abnormal_30[0])
    sns.lineplot(x=range(1, 13), y=abnormal_30[1])
    sns.lineplot(x=range(1, 13), y=abnormal_10[0])
    sns.lineplot(x=range(1, 13), y=abnormal_10[1])
    plt.show()


