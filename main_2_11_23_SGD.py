import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 

class LinearRegression:
    def __init__(self, X, Y, epsilon = 0.001, epoches = 1000, lam = 0.01):
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.epoches = epoches
        self.lam = lam
        self.ineration = []
        self.cost = []
        self.length = len(Y)

    def Cost(self, y_predict):
        return (y_predict-self.Y.T)@(y_predict-self.Y)

    def predict(self, w, bias, x):
        return x@(w.T) + bias
    
    def fit(self):
        w = np.random.randn(1, len(self.X[0]))
        bias = np.random.randn(1)
        for epoches in range(self.epoches):
            cost = 0
            Mix = np.random.permutation(len(self.X))
            for epoch in Mix:
                y_pred = self.predict(w, bias, self.X[epoch])
                dw = self.X[epoch] * (y_pred - self.Y[epoch]) + self.lam * w
                dbias = y_pred - self.Y[epoch]

                w -= self.epsilon * dw
                bias -= self.epsilon * dbias

                cost += (y_pred - self.Y[epoch])**2 + self.lam * np.sum(w**2)

                sys.stdout.write("\rEpoches : {}| {}/{}".format(epoches + 1, epoch + 1, len(self.X)))
            self.cost.append(cost[0])
        
        return w[0], bias[0]
    
    def showCost(self):
        print(self.cost[-1])
        plt.plot(self.cost)
        plt.show()


data = pd.read_csv('DataSet/train.csv')

x = data.iloc[:, 1:11].values
y = data.iloc[:, -1].values.reshape(-1, 1)
LR = LinearRegression(x[:997, :], y[:997, :], 0.0000000000001, 500, 0)
w, bias = LR.fit()
print()
print(LR.predict(w, bias, x[-5]))
print(LR.predict(w, bias, x[-4]))
print(LR.predict(w, bias, x[-3]))
print(LR.predict(w, bias, x[-2]))
print(LR.predict(w, bias, x[-1]))
LR.showCost()

