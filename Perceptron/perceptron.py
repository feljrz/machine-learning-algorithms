#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algoritimo Perceptron

@author: felipe
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random

dataset = pd.DataFrame(pd.read_csv("Basedados_B11.csv", encoding="utf-8" ))
alfa = random()
w_new = np.zeros((2,1))
b_new = 0
edge = 0
x = np.array(dataset.iloc[:,:-1])
y = np.array(dataset.iloc[:,-1]).reshape(14,1)


def activation_function(y_liquid):
    return np.where(y_liquid > 0, 1, -1)


#Train
def train(x, y, verbose=False):
    global alfa, w_new, b_new, edge
    w_old = w_new.copy()
    b_old = 0
    change = True
    y_li = np.empty(y.shape)
    flag = 0

    while(change):
        y_li = x.dot(w_new) + b_new
        y_li = activation_function(y_li) # Aplicando função de ativação
        #id_changed = np.where(y_li != y)[0] #Indice necessário a se alterar; (não funciona como iterável???)
        for i in range(len(y)):
            if y_li[i] != y[i]:
                w_new = w_old + alfa*((x[i]*y[i]).reshape(2,1))
                b_new = b_old + alfa*y[i]
                flag += 1
            elif np.array_equal(y_li, y):
                change = False
            if verbose:
                print(f'w: {w_new};\ny_li[i]: {y_li[i]};\ny[i]: {y[i]}')
        w_old = w_new
        b_old = b_new


#Test
def test(x, y):
    y_pred = x.dot(w_new) + b_new
    y_pred = activation_function(y_pred)
    print("y_predicted| Target")
    print(np.c_[y_pred, y])
    return y_pred




train(x, y, verbose=True)
y_pred = test(x, y)

#Plot
X_grid = np.array(np.arange(-5, 5, 0.2))
ordena = (w_new[0,:] * X_grid - b_new)/w_new[1,:]

plt.scatter(x= dataset['s1'], y=dataset['s2'], color='blue')
plt.scatter(x=X_grid, y=ordena, color='orange')
plt.show()


