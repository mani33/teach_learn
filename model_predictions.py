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
# import shap
# import umap
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
# sm = shap.TreeExplainer(model, feature_names=list(X))
# v = np.abs(sm.shap_values(xs)).mean(axis=0).mean(axis=1)
# shap.summary_plot(sm.shap_values(xs)[:,:,4], x, feature_names=list(X), plot_type='bar')
# shap.summary_plot(sm.shap_values(xs)[:,:,4], x, feature_names=list(X))
#%% PCA
# re = umap.UMAP()
# em = re.fit_transform(x)
# plt.scatter(em[:,0], em[:,1], c=y)
#%% t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
ft = tsne.fit_transform(xs)
plt.scatter(ft[:,0], ft[:,1], c=y)
#%% nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from torch.optim import Adam

class DermDataset(Dataset):
    def __init__(self, feature_df, y, feature_names):
        self.X = feature_df[feature_names].to_numpy()
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        y = torch.tensor(self.y[index]-1, dtype=torch.long)
        return torch.tensor(self.X[index], dtype=torch.float32), y
    
# Define a simple neural network with multiple outputs
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        # 
        curr_size = input_dim
        layers = []
        for hs in hidden_dims:
            layers.append(nn.Linear(curr_size, hs))
            layers.append(nn.BatchNorm1d(hs))
            layers.append(nn.ReLU())
            curr_size = hs
        
        layers.append(nn.Linear(curr_size, output_dim))
        # layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
        
        
     
data = fetch_ucirepo(id=33)
X, ydf = data.data.features, data.data.targets
y = ydf.to_numpy().flatten()
features = X.columns.to_numpy()
imp = IterativeImputer(missing_values=np.nan)
X2 = imp.fit_transform(X)
x = pd.DataFrame(X2, columns=features)
sc = StandardScaler().set_output(transform='pandas').fit(x)
xs = sc.transform(x)

dataset = DermDataset(xs, y, features)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

#%% Train
input_dim = xs.shape[1]
hidden_dims = [128, 64]
output_dim = len(np.unique(y))
model = SimpleNN(input_dim, hidden_dims, output_dim)
model.train()
u, uc = np.unique(y, return_counts=True)
w = uc/y.size
weights = torch.tensor(w, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights)
optim = Adam(params=model.parameters(), betas=[0.9, 0.9], lr=0.0001)

dataset = DermDataset(xs, y, features)
data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

loss_hist = []
loss_beta = 0.9
n_epochs = 500
# Create three data loaders: training, validation and testing
def get_data_loaders(X, y, feature_names):
    # 70:15:15 split
    # Trianing and testing split
    x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True)
    # Split training into training + val
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.5, stratify=y_train_val, shuffle=True)
    
    data_set = DermDataset(x_train, y_train, feature_names)
    train_loader = DataLoader(data_set, batch_size = 16, shuffle=True)

    data_set = DermDataset(x_val, y_val, feature_names)
    val_loader = DataLoader(data_set, batch_size = len(data_set), shuffle=True)
    
    data_set = DermDataset(x_test, y_test, feature_names)
    test_loader = DataLoader(data_set, batch_size = len(data_set), shuffle=True)

    return train_loader, val_loader, test_loader, y_test

train_loader, val_loader, test_loader, y_test = get_data_loaders(xs, y, features)

train_loss_hist = []
val_loss_hist = []

checkpoints = []
for i in range(n_epochs):
    epoch_loss = 0
    val_loss = 0    
    for xb, yb in train_loader:
        optim.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        epoch_loss = loss_beta * epoch_loss + (1 - loss_beta)*loss.item()
        loss.backward()
        optim.step()
        # Compute validation loss
        v_loss_ = []
        with torch.no_grad():
            for xv, yv in val_loader:
                out = model(xv)
                v_loss_.append(criterion(out, yv).item())
        val_loss = val_loss * loss_beta + (1 - loss_beta)*np.mean(v_loss_)
        # Save
    checkpoints.append(dict(state_model = model.state_dict(), state_optim=optim.state_dict(), epoch=i))
    
    train_loss_hist.append(epoch_loss)
    val_loss_hist.append(val_loss)


plt.plot(np.array([train_loss_hist, val_loss_hist]).T, label=['train','val'])
plt.legend()
#%% Testing   
# Pick the best model
best = np.array(checkpoints)[np.argmin(val_loss_hist)]

best_model = SimpleNN(input_dim, hidden_dims, output_dim)
optim = Adam(params=best_model.parameters(), betas=[0.9, 0.9], lr=0.001)

best_model.load_state_dict(best['state_model'])
optim.load_state_dict(best['state_optim'])

best_model.eval()

with torch.no_grad():
    p = []
    for xv, yv in test_loader:
        out = best_model(xv)
        p.append(torch.softmax(out, dim=1).detach().numpy())

p = np.vstack(p)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, RocCurveDisplay, accuracy_score, ConfusionMatrixDisplay, confusion_matrix

y_test_bin = label_binarize(y_test, classes=np.arange(6)+1)

fpr,tpr,th = roc_curve(y_test_bin.flatten(), p.flatten())

rd = RocCurveDisplay.from_predictions(y_test_bin.flatten(), p.flatten())
cm = confusion_matrix(y_test, np.argmax(p, axis=1)+1)
ConfusionMatrixDisplay(cm).plot()

#%% Build a simple CNN
from torchvision import datasets, transforms
from datasets import load_dataset
# train_set = datasets.MNIST(root=r'./data', train=True, download=True)
# test_set = datasets.MNIST(root=r'./data', train=False, download=True)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

dataset = load_dataset("microsoft/cats_vs_dogs", split='train')
# dataset['image'] = dataset['image'][0:400]
# dataset['labels'] = dataset['labels'][0:400]
# Create custom dataset

class CatDogDataset(Dataset):
    def __init__(self, pil_dataset):
        self.Xy = pil_dataset     
        
    def __len__(self):
        return len(self.Xy)
    
    def __getitem__(self, index):
        X = torch.tensor(np.array(self.Xy[index]['image']), dtype=torch.float32)
        X = X.permute(2,0,1)
        X = transforms.Resize((224, 224))(X) # 3 x 224 x 224
        X = X/255.0
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        X = normalize(X)
        y = torch.tensor(np.array(self.Xy[index]['labels']), dtype=torch.float32)
        return X, y
    

my_dataset = CatDogDataset(dataset)

train_loader = DataLoader(my_dataset, batch_size=64, shuffle=True)


#%% Create a CNN
class MyCNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), # 112 x 112
        
        nn.Conv2d(16, 32, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), # 56 x 56 28 x 28
        
        nn.Conv2d(32, 64, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), # 14 x 14
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(28*28*64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, output_dim)
            )
       
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x.view(x.size(0),-1))
        return x
    
    
my_dataset = CatDogDataset(dataset)

train_loader = DataLoader(my_dataset, batch_size=4, shuffle=True)
model = MyCNN(output_dim=1)
optim = Adam(params = model.parameters(), lr=0.001)
# Add a schedule
scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, eta_min=1e-6)
criterion = nn.BCEWithLogitsLoss()
model.train()
for i in range(100):
    for j, (x, y) in enumerate(train_loader):
        if j < 5:
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y.reshape(y.size(0),-1))
            loss.backward()
            optim.step()
            # scheduler.step(i + j/len(train_loader))
    print('loss in ', i, loss.item())

#%% LSTM

class SeqClassifier(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self,x):
        # x - shape is [batch, seq_len, features]
        o, (h, c) = self.lstm(x)
        last_out = h[-1]
        
        return self.fc(last_out)
        
 
#%% SVM
from sklearn import svm

clf = svm.SVC(kernel='linear', C=1)
        

#%% 
from torchvision import models


        
        