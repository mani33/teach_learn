# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:49:38 2026
classifiers
@author: msubramaniyan
"""
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, average_precision_score
from sklearn.decomposition import PCA
import shap
import umap
from sklearn.preprocessing import StandardScaler
#%%
data = fetch_ucirepo(id=33)
X, ydf = data.data.features, data.data.targets
y = ydf.to_numpy().flatten()
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
ydf.value_counts()
#%% Fit a multinomial logistic regression
model = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000)
#%% 5-fold cv
# My scorer
# custom_scorer = make_scorer(f1_score, average='macro', response_method='predict_proba')
param_grid = dict(C=[0.01, 0.1, 1])
inner_best_model = GridSearchCV(model, param_grid, scoring='f1_macro', cv=StratifiedKFold(n_splits=4))
res = cross_validate(inner_best_model, x, y, cv=StratifiedKFold(n_splits=5), scoring='f1_macro')

#%% Dimension reduction
ss = StandardScaler()
ssf = ss.fit(x)
xs = ssf.transform(x)
pca = PCA().fit(xs)
a = pca.transform(xs)
loadings = pca.components_.T @ np.diag(np.sqrt(pca.explained_variance_))
#%%
model = RandomForestClassifier(min_samples_leaf=5, min_samples_split=6)
model.fit(xs, y)
sm = shap.TreeExplainer(model, feature_names=list(X))
v = np.abs(sm.shap_values(xs)).mean(axis=0).mean(axis=1)
# shap.summary_plot(sm.shap_values(xs)[:,:,4], x, feature_names=list(X), plot_type='bar')
shap.summary_plot(sm.shap_values(xs)[:,:,4], x, feature_names=list(X))
#%% PCA
re = umap.UMAP()
em = re.fit_transform(x)
plt.scatter(em[:,0], em[:,1], c=y)
#%% t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
ft = tsne.fit_transform(xs)
plt.scatter(ft[:,0], ft[:,1], c=y)