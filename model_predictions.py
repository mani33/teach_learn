# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:49:38 2026
classifiers
@author: msubramaniyan
"""
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
#%%
data = fetch_ucirepo(id=33)
X, y = data.data.features, data.data.targets

features = X.columns.to_numpy()

for feat in features:
    # print(feat, X[feat].unique())
    if any(X[feat].isna()): print(feat)

sim = SimpleImputer(missing_values=np.nan).set_output(transform='pandas')
sim.fit(X)
xx = sim.transform(X)
sim2 = IterativeImputer(missing_values=np.nan).set_output(transform='pandas')
x = sim2.fit_transform(X)
#%% Check for class imbalance
h = np.histogram(y, bins=[-np.inf, 0, 1, 2, 3])
#%% Fit a multinomial logistic regression
model = LogisticRegression(multi_class='multinomial', class_weight='balanced')
model.fit(x, y)


