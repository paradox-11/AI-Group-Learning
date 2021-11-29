import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression():
    '''
    正则化线性回归
    使用梯度下降法， 牛顿法， 最小二乘法学习参数
    X, theta 都以行为一组
    '''
    def __init__(self, iters, learningRate=0.01, reg=1):
        self.learningRate = learningRate
        self.reg = reg
        self.iters = iters
        self.cost = np.zeros(iters)

    def calculateByGradientDescent(self, data):
        self.dataProcess(data)
        self.theta = self.gradientDescent(self.X, self.y, self.theta)

    def calculateByNewtonMethod(self, data):
        self.dataProcess(data)
        self.theta = self.newtonMethod(self.X, self.y, self.theta)

    def calculateByLeastSqaureMethod(self, data):
        self.dataProcess(data)
        self.theta = self.leastSqaureMethod(self.X, self.y, self.theta)

    def dataProcess(self, data):
        self.X = data[:, :-1]
        self.X = np.insert(self.X, 0, 1, axis=1)
        self.y = data[:, -1:]
        self.theta = np.matrix(np.zeros(self.X.shape[1]))
        self.m = len(self.X)
        self.featureNum = self.X.shape[1] - 1

    def costFunction(self, theta, X, y):
        reg = self.reg * np.sum(np.power(theta, 2)) / (2 * self.m)
        return np.sum(np.power(X * theta.T - y, 2)) / 2 * self.m + reg

    def gradientDescent(self, X, y, theta):
        for i in range(self.iters):
            self.cost[i] = self.costFunction(theta, X, y)
            gradient = X.T * (X * theta.T - y) / self.m + self.reg / self.m * theta.T
            theta = theta * (1 - self.learningRate * self.reg / self.m) - self.learningRate * gradient.T

        return theta

    def newtonMethod(self, X, y, theta):
        for i in range(self.iters):
            self.cost[i] = self.costFunction(theta, X, y)
            gradient_1 = X.T * (X * theta.T - y) / self.m + self.reg / self.m * theta.T
            gradient_2 = X.T * X / self.m + self.reg / self.m * np.identity(X.shape[1])
            theta = theta - (np.linalg.inv(gradient_2) * gradient_1).T 
        
        return theta

    def leastSqaureMethod(self, X, y, theta):
        theta = (np.linalg.inv(X.T * X) * X.T * y).T
        return theta

    def predict(self, theta, X):
        return X * theta.T

    def getFeatureNum(self):
        return self.featureNum

    def getSampleNum(self):
        return self.m

    def getTheta(self):
        return self.theta

    def predict(self, X, theta):
        X = np.insert(X, 0, 1, axis=1)
        return X * theta.T

    def showOutPut(self):
        if (self.featureNum == 1):
            x1 = np.linspace(np.min(self.X[:,1]), np.max(self.X[:,1]), 1000)
            predictions = self.predict(np.matrix(x1).T, self.theta)

            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(self.X[:,1].A, self.y.A, s=50, c='r')
            ax.plot(x1, predictions, c='b',label='Prediction')
            plt.show()

        elif (self.featureNum == 2):
            x1 = np.linspace(np.min(self.X[:,1]), np.max(self.X[:,1]), 1000)
            x2 = np.linspace(np.min(self.X[:,2]), np.max(self.X[:,2]), 1000)
            X_test = np.matrix([x1, x2]).T
            predictions = self.predict(X_test, self.theta)

            fig = plt.figure(figsize=(8,6))
            ax = plt.axes(projection='3d')
            ax.scatter3D(x1,x2,predictions)
            plt.show()
        
        else :
            print('维数过高')

class DataReader():
    '''
    获得包含数据的矩阵
    '''
    def __init__(self):
        pass
    
    def getData(self, path):
        data = pd.read_csv(path)
        data = np.matrix(data.values)
        return data

if __name__ == '__main__':
    data_reader = DataReader()
    data = data_reader.getData('ex1data1.txt') #第一列是城市人口数量，第二列是该城市小吃店利润

    lr_model = LinearRegression(iters=1)
    lr_model.calculateByLeastSqaureMethod(data)
    lr_model.showOutPut()
