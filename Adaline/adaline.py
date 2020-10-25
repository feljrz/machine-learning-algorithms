
"""
Algoritimo Adaline
@author: felipe
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random


dataset = pd.DataFrame(pd.read_csv("Basedados_B11.csv", encoding="utf-8" ))
alfa = random() - 0.5
alfa = 0.01
w_new = np.zeros((2,1))
b_new = 0
edge = 0
x = np.array(dataset.iloc[:,:-1])
t = np.array(dataset.iloc[:,-1]).reshape(14,1)


def activation_function(y_liquid):
    return np.where(y_liquid > 0, 1, -1)


#Test
def test(x, y):
    y_pred = x.dot(w_new) + b_new
    y_pred = activation_function(y_pred)
    print("y_predicted| Target")
    print(np.c_[y_pred, y])
    return y_pred


#Train
w_old = w_new.copy()
b_old = 0
change = True
y_li = np.empty(t.shape)
flag = 0
stop = 0
error = []
w_list = []

while(stop != 300 or max(error) < 0.0001):
    stop +=1
    e_lsm = 0
    for i in range(len(t)):
        y_li = (w_old[0]*x[i,0]) + (w_old[1]*x[i,1]) + b_old
        y = y_li
        #Erro
        e_lsm += ((t[i] - y)**2)
        error.append(e_lsm)
        w_new[0] = w_new[0] + alfa * (t[i] - y)*x[i,0]
        w_new[1] = w_new[1] + alfa * (t[i] - y)*x[i,1]
        b_new = b_old + alfa * (t[i] - y)w_old = w_new
        b_old = b_new


y_pred = test(x, t)


#Plot
X_grid = np.array(np.arange(-5, 5, 0.2))
ordena = (w_new[0,:] * X_grid - b_new)/w_new[1,:]
plt.scatter(x= dataset['s1'], y=dataset['s2'], color='blue')
plt.scatter(x=X_grid, y=ordena, color='orange')
plt.show()
tam = len(error)/2
X_grid_2 = np.array(np.arange(-tam,tam))
plt.scatter(X_grid_2, error, color='red')
plt.show()
