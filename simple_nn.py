# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 19:02:42 2025

@author: Mani Subramaniyan
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
# ------------------------------
# 1. Generate noisy polynomial data
# ------------------------------
np.random.seed(0)
n = 200
x = np.linspace(-2, 2, n).reshape(-1, 1)
y_true = 3*x**5 - 2*x**3 + x
noise = 10 * np.random.randn(n, 1)
y = y_true + noise

# ------------------------------
# 2. Define network architecture
# ------------------------------
input_dim = 1
hidden_dim = 32
output_dim = 1

# Plot data and see
plt.figure(figsize=(8,5))
plt.scatter(x, y, label='Noisy data', color='gray', alpha=0.6)

# ------------------------------
# 2. Define network architecture
# ------------------------------
input_dim = 1
hidden_dim = 32
output_dim = 1

# Initialize weights and biases
A = np.random.randn(input_dim + 1, hidden_dim) * 0.1
B = np.random.randn(hidden_dim + 1, output_dim) * 0.1


# Activation and its derivative
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

# ------------------------------
# 3. Training parameters
# ------------------------------
lr = 0.01
epochs = 5000

# ------------------------------
# 4. Training loop (forward + backward)
# ------------------------------
S = x[:, np.newaxis]
Sa = np.hstack((S, np.ones((n,1))))
for epoch in range(epochs):
    # ---- Forward pass ----    
    psi = S @ A
    X = tanh(psi)
    X = np.hstack((X, np.ones(n, 1)))
    theta = X @ B
    yh = theta
    # Backprop
    dL = yh - y # dL/dyh
    
    # Backward pass
    dB = dL @ X
    dA = dL @ B @ tanh_derivative(psi) @ S
    
#%%
import torch
from torchview import draw_graph
import torch.nn as nn

# Define a simple model (example)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
input_data = torch.randn(1, 10) # Provide a sample input

# Draw the graph
model_graph = draw_graph(model, input_size=input_data.shape, expand_nested=True)

# Display the graph (works well in Jupyter Notebooks or Google Colab)
model_graph.visual_graph
# Or save it to a file
model_graph.visual_graph.render("model_architecture") 
    