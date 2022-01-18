import pandas as pd
import numpy as np
from sklearn.datasets import load_boston  # 导入数据集
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class LinearRegression:
    """实现线性回归"""
    def __init__(self):
        """获取数据维度,分割并处理数据"""
        self.m, self.n = data.shape
        self.y = np.array(data['PRICE']).reshape(self.m, 1)
        self.x = np.hstack((np.ones((self.m, 1)),np.array(data.drop(['PRICE'], axis=1))))
        self.w = np.zeros((1, self.n))

    def calculate(self):
        """计算"""
        self._leastsqaure()

    def _leastsqaure(self):
        """最小二乘法"""
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.x.T, self.x)), self.x.T), self.y).T

    def _gradientdescent(self):
        """梯度下降法"""
        self.alpha = 0.000007    # 学习率
        self.cost = []           # 收集代价，观察趋势
        for i in range(100):
            tmp = self.y - np.dot(self.x, self.w.T)
            self.cost.append(np.mean(tmp ** 2) / 2)
            self.w += self.alpha * np.dot(self.x.T, tmp).T

    def _newton(self):
        """牛顿法"""
        self.w = np.dot(np.dot(np.linalg.inv(np.dot(self.x.T, self.x)), self.x.T),self.y).T

    def showoutput(self):
        """数据可视化"""
        self.predictions = np.matmul(self.x, self.w.T)
        self.x = np.delete(self.x, 0, axis=1)
        if self.n == 3:
            x1, x2 = np.hsplit(self.x, 2)
            fig = plt.figure(figsize=(13, 7))
            ax = Axes3D(fig)
            ax.scatter(x1, x2, self.predictions,c = 'green')
            ax.scatter(x1, x2, self.y, c='red')
            plt.show()
        elif self.n == 2:
            fig = plt.figure(figsize=(13, 7))
            plt.scatter(self.x, self.predictions)
            plt.scatter(self.x, self.y)
            plt.show()

class DataReader:
    """获取数据"""
    def __init__(self, filename):
        """载入数据集"""
        self.data = filename

    def returnData(self):
        """数据筛选，返回筛选后的数据集"""
        data_pd = pd.DataFrame(self.data.data, columns=self.data.feature_names)
        data_pd['PRICE'] = self.data.target
        data_pd = data_pd[['LSTAT', 'PTRATIO', 'PRICE']]
        return data_pd
        # 'LSTAT'， 'PTRATIO', 'RM'为数据集中相关度较高的因素

if __name__ == "__main__":
    data_reader = DataReader(load_boston())
    data = data_reader.returnData()
    lr_model = LinearRegression()
    lr_model.calculate()
    lr_model.showoutput()