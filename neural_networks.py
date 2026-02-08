# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 15:09:57 2026
Neural network teaching module for Anirudh

@author: Mani Subramaniyan
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
def activation(x, fun_type):
    match fun_type:
        case 'relu':
            return np.maximum(0, x)
        case 'tanh':
            return np.tanh(x)
        case _:
            raise NotImplementedError('')

def activation_der(x, fun_type):
    match fun_type:
        case 'relu':
            return (x > 0).astype(float)
        case 'tanh':
            return 1 - np.tanh(x)**2
        case _:
             raise NotImplementedError('')
    
#%%    
f = 0.37 # cycles/s
n = 1000
t = np.random.random(n)
A = 0.01
yb = A * np.sin(2 * np.pi * f * t)
y = yb + np.random.randn(n) * 0.001
plt.scatter(t, y)

#%%
n_epoch = 1000
bs = 5 # batch size
n_batch = n // bs
input_dim = 1
n_hidden = 8
output_dim = 1
# Initialize weights
w1 = np.random.randn(input_dim, n_hidden) * np.sqrt(2/input_dim)
w2 = np.random.randn(n_hidden, output_dim) * np.sqrt(2/n_hidden)
b1 = np.random.randn(n_hidden).reshape((1, n_hidden))
b2 = np.random.randn(output_dim).reshape((1, output_dim))
alpha = 0.0001
train = True
yhat = []
fun_type = 'tanh'
loss_epoch = []
for i in range(n_epoch):
    loss = []
    for j in range(n_batch):
        a0 = t[j*bs:(j+1)*bs].reshape((bs, input_dim)) # (bs, input_dim)
        yj = y[j*bs:(j+1)*bs].reshape((bs, output_dim)) # (bs, output_dim)
        # Forward pass
        z1 = np.matmul(a0, w1) + b1       
        a1 = activation(z1, fun_type)
        
        z2 = np.matmul(a1, w2) + b2
        a2 = z2
        
        L = np.mean((yj - a2)**2)
        loss.append(L)
        # Backward pass
        d2 = (1/n_batch) * (yj-a2) * activation_der(z2, fun_type)
        dLdw2 = np.matmul(a1.T, d2)
        
        dLdb2 = np.sum(d2 * 1, 0, keepdims=True)
        
        d1 = np.matmul(d2, w2.T) * activation_der(z1, fun_type)
        dLdw1 = np.matmul(a0.T, d1)
        dLdb1 = np.sum(d1 * 1, 0, keepdims=True)
        
        # Adjust weights
        w1 = w1 - alpha * dLdw1
        b1 = b1 - alpha * dLdb1
        
        w2 = w2 - alpha * dLdw2
        b2 = b2 - alpha * dLdb2
        
    loss_epoch.append(np.mean(loss))
#%% Inference
a0 = t.reshape((-1, 1))

z1 = np.matmul(a0, w1) + b1       
a1 = activation(z1, fun_type)

z2 = np.matmul(a1, w2) + b2
yhat = z2

plt.scatter(t, yhat)