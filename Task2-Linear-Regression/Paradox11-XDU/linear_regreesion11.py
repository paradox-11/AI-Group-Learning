import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self,alpha=0.01,iters=1500):
        self.alpha=alpha
        self.iters=iters#需要的变量可以后面再加

    def getdata(self,path,variablename1,variablename2):
        data=pd.read_csv(path,names=[variablename1,variablename2])
        #data.plot(kind="scatter",x=variablename1,y=variablename2,figsize=(12,8))
        #plt.show()#调试代码

        #data=np.insert(data,0,1,axis=1)#三种insert的使用方式
        #data=np.insert(data,0,values=np.ones(data.shape[0]),axis=1)
        data.insert(0,"Ones",1)
        
        self.X=data.iloc[:,:-1]
        self.y=data.iloc[:,-1:]
        self.X=np.matrix(self.X)
        self.y=np.matrix(self.y)
        self.theta=np.matrix(np.array([0,0]))
        self.vaiablename1=variablename1
        self.vaiablename2=variablename2
        self.variable_1=data.iloc[:,1]
        self.variable_2=data.iloc[:,2]

        #print(self.X)#调试代码
        #print("-"*50)
        #print(self.y)
        #print("-"*50)
        #print(self.theta)

        return data
        
    def computecost(self,theta,X,y):
        return np.sum(np.power(X*theta.T-y,2))/(2*len(X))

    def gradientdescent(self,theta,X,y,alpha,iters):
        temp=np.matrix(np.zeros(theta.shape))
        #cost=np.zeros(iters)
        self.cost=np.zeros(iters)
        for i in range(iters):
            error=X*theta.T-y
            grad=X.T*error
            theta=theta-alpha*grad.T/len(X)
            #cost[i]=self.computecost(theta,X,y)
            self.cost[i]=self.computecost(theta,X,y)
        #return theta,cost
        return theta
    
    def leastsquaremethod(self,theta,X,y):
        theta = np.linalg.inv(X.T@X)@X.T@y#X.T@X等价于X.T.dot(X),且此处@可用*代替
        return theta

    def newtonmethod(self,theta,X,y):
        for i in range(self.iters):
            grad=X.T*(X*theta.T-y)/len(X)
            hess=X.T*X/len(X)
            theta=theta-(np.linalg.inv(hess)*grad).T
        return theta

    def calculatebygradientdescent(self):
        self.theta=self.gradientdescent(self.theta,self.X,self.y,self.alpha,self.iters)

    def calculatebyleastsquaremethod(self):
        self.theta=self.leastsquaremethod(self.theta,self.X,self.y).T

    def calculatebynewtonmethod(self):
        self.theta=self.newtonmethod(self.theta,self.X,self.y)

    def showOutput(self):
        x=np.linspace(self.variable_1.min(),self.variable_2.max(),100)
        f=self.theta[0,0]+self.theta[0,1]*x
        fig,ax=plt.subplots(figsize=(12,8))
        ax.plot(x,f,c="r",label="Prediction")
        ax.scatter(self.variable_1,self.variable_2,c="b",s=20,marker="o")
        ax.legend(loc=2)
        ax.set_xlabel(self.vaiablename1)
        ax.set_ylabel(self.vaiablename2)
        ax.set_title("training data")
        plt.show()

if __name__=="__main__":
    path="4\\data4.txt"
    variablename1="Population"
    variablename2="Profit"
    data_operate=LinearRegression()
    data=data_operate.getdata(path,variablename1,variablename2)

    #theta,cost=data_operate.gradientdescent(data_operate.theta,data_operate.X,data_operate.y,data_operate.alpha,data_operate.iters)
    #此处对应另一种封装方法，但封装性不如这种好

    #data_operate.calculatebygradientdescent()#梯度下降法
    #data_operate.calculatebyleastsquaremethod()#最小二乘法
    data_operate.calculatebynewtonmethod()#牛顿法
    data_operate.showOutput()
