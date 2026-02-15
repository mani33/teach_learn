# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:14:53 2026
binary classifier
@author: msubramaniyan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import RocCurveDisplay, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay
# %% Split data into training and testing
dfo = pd.read_csv(r'data/TCGA_InfoWithGrade.csv')

df = dfo.copy()
target = 'Grade'

# Dummy code categorical variables
cat_var = ['Gender', 'Race']
non_genes = ['Grade','Gender','Race','Age_at_diagnosis']
genes = [f for f in dfo.columns if f not in non_genes]
df = pd.get_dummies(df, columns=cat_var, drop_first=True)
df = df.sort_values(by='Grade', kind='stable')
df = df.sort_index(axis=1)
features = [f for f in df.columns if f != target]
df = df[features + [target]]

#%% Build classifier
# Interaction term builder
pol = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_genes = df[genes]
X_genes_int = pol.fit_transform(X_genes)
int_names = pol.get_feature_names_out(genes)
dfin = pd.DataFrame(X_genes_int, columns=int_names)
non_genes = list(set(df.columns) - set(genes))
dff = pd.concat((df[non_genes], dfin), axis=1)

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
plt.figure()
logits = np.log(y_score/(1-y_score))
#%%
c = np.array(['r']*len(y_test))
c[y_test==0] = 'b'
plt.scatter(np.arange(len(X_test)), logits,20, c)
plt.hlines(0, 0, len(X_test))

#%% Let's do 5-fold nested cv
param_grid = dict(C = [0.01, 0.1, 0.5, 1])
inner_best_model = GridSearchCV(LogisticRegression(solver='saga', penalty='l1', max_iter=1000),
                                param_grid=param_grid,
                                cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=10),
                                return_train_score=True)
m = df['Age_at_diagnosis'].mean()
s = df['Age_at_diagnosis'].std()
df.loc[:,'Age_at_diagnosis'] = (df['Age_at_diagnosis'].values-m)/s
X = df[features].values
y = df['Grade'].values
models = cross_validate(inner_best_model, X, y, scoring='roc_auc', 
                        cv=StratifiedKFold(n_splits=5), return_estimator=True)
                       
                        


