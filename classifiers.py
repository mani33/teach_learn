# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:14:53 2026
binary classifier
@author: msubramaniyan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay
# %% Split data into training and testing
dfo = pd.read_csv(r'P:/teach_learn/data/TCGA_InfoWithGrade.csv')

df = dfo.copy()
target = 'Grade'

# Dummy code categorical variables
cat_var = ['Gender', 'Race']
genes = [f for f in dfo.columns if f not in ['Grade','Gender','Race','Age_at_diagnosis']]
df = pd.get_dummies(df, columns=cat_var, drop_first=True)
df = df.sort_values(by='Grade', kind='stable')
df = df.sort_index(axis=1)
features = [f for f in df.columns if f != target]
df = df[features + [target]]

#%% Build classifier
model = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear')
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(df[features], y, 
                                                    test_size=0.3,
                                                    stratify=y)
model = model.fit(X_train, y_train)
y_score = model.predict_proba(X_test)[:,1]
y_pred = y_score > 0.5
rd = RocCurveDisplay.from_predictions(y_test, y_score)
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm)
cmd.plot()
plt.figure()
plt.stem(features,model.coef_[0,:])
plt.xticks(rotation=45, ha='right')

#%% Correlation
gdf = df[genes + ['Grade']]
plt.figure()
plt.imshow(gdf.corr(), aspect='auto')


