#Algoritimo de Hebb
#Felipe Júnio Rezende 11711ECP007

import numpy as np
import itertools as it

source = np.array([(1, 1), (1, -1), (-1, 1), (-1, -1)])
target = np.array([(1,), (-1,), (-1,), (-1,)])

w_new = np.zeros((2,1))
bias_new = 0
edge = 0

def bipolar_representation(arr):
    for i in range(len(arr)):
        arr[i] = 1 if arr[i] > 0 else -1
    return arr

def binary_generator(digits):
    binlist = []
    for i in range(16):
        b = bin(i)[2:].zfill(digits) #zfill preenche com 0 à esquerda
        num = list(b) #Split dos chars
        for j in range(len(num)): #Parse + representação bipolar
            num[j] = int(num[j])
        num = bipolar_representation(num)
        binlist.append(num)
    return binlist


#Treino
def train(s, t, verbose = False):
    w_old = np.zeros((2,1))
    bias_old = 0
    global w_new, bias_new
    for x, y in zip(s, t):
        w_new[0] = w_old[0] + x[0] * y
        w_new[1] = w_old[1] + x[1] * y
        bias_new = bias_old + y
        w_old = w_new
        if verbose:
            print(f"w_new: {w_new.reshape(1,2)} \nbias_new: {bias_new}")


#Teste
def test(s, t):
    y_li = s.dot(w_new) + bias_new
    y_li = bipolar_representation(y_li)
    for i in range(len(y_li)):
        print(f"y_test{y_li[i]} | y: {t[i]}")


#Testando para um valor
train(source, target, verbose = True)
test(source, target)


#Testando para os 16 valores binários
values = binary_generator(4)
values = np.array(values)

for tar in values:
    train(source, tar)
    test(source, tar)
    print("")


