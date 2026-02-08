# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 12:11:31 2026
cat versus dog classifier
@author: Mani Subramaniyan
"""
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import nn_utils as nnu

#%%
class CatDogDataset(Dataset):    
    def __init__(self, data_dir, im_shape=(224, 224)):
        super().__init__()
        self.data_dir = data_dir    
        self.image_data = []
        classes = ['cats','dogs']
        self.labels = []
        for i, cls in enumerate(classes):
            class_dir = os.path.join(data_dir, cls)
            im_names = os.listdir(class_dir)
            for im_name in im_names:
                self.labels.append(torch.tensor(i))
                im = cv2.imread(os.path.join(class_dir, im_name))
                im = torch.tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                im = transforms.Resize(im_shape)(im.permute(2,0,1))
                self.image_data.append(im) # 3 x w x h
                print(f'Loaded {im_name} of class {cls}')
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        return self.image_data[idx], self.labels[idx]

class SimpleCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x64
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 16x16            
            )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)) # 2 output classes cat & dog
    
    def forward(self, x):
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x
    
def center_scale_images(im, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    im = im/255.0      
    m = torch.tensor(mean).view(1,3,1,1)
    s = torch.tensor(std).view(1,3,1,1)
    
    return (im - m)/s

def classify_image(im_file, model):
    im = cv2.imread(im_file)
    im_shape = (224, 224)
    im = torch.tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    im = transforms.Resize(im_shape)(im.permute(2,0,1))
    im = im[None, ...]
    im = center_scale_images(im)
    model.eval()
    out = model(im.to(device))
    p = torch.softmax(out,1)
    _, pred = torch.max(out, 1)
    classes = ['cats','dogs']
    return classes[pred], p[0,pred.item()].item()

#%%
if __name__=='__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    data_dir = r'C:\Users\maniv\Documents\for_ani\datasets\cat_vs_dog'
    train_data = CatDogDataset(os.path.join(data_dir, 'train'))
    test_data = CatDogDataset(os.path.join(data_dir, 'test'))
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    #%%
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
  
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
      
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    #%%
    n_epochs = 5
    
    for epoch in range(n_epochs):        
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = center_scale_images(images)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_loader):.4f}")


#%% Inference
    
model.eval()
correct, total = 0, 0
for images, labels in test_loader:
    images = center_scale_images(images)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, pred = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (pred == labels).sum().item()
    
print(f"Validation Accuracy: {100 * correct / total:.2f}%")


#%% Check a test image
# imf = 'C:/Users/maniv/Downloads/SUMI_PASSPORT2.jpg'
# imf = 'C:/Users/maniv/Downloads/SUMI.jpg'
# imf = 'C:/Users/maniv/Downloads/mani.png'
# imf = 'C:/Users/maniv/Downloads/cat.jpg'
# imf = 'C:/Users/maniv/Downloads/girl.jpg'
# imf = 'C:/Users/maniv/Downloads/IMG_0110.jpg'
imf = 'C:/Users/maniv/Downloads/woman_dog.png'
imf = 'C:/Users/maniv/Downloads/woman_cat.png'
c, p = classify_image(imf, model)
print(f'class: {c}, probability: {p:1.2f}')

#%% checking kernels
w = model.features[0].weight.detach().to('cpu').numpy()
for i in range(w.shape[0]):
    plt.subplot(8,8,i+1)
    im = np.transpose(w[i].squeeze(),(1,2,0))
    im = (im - np.min(im))
    im = im/np.max(im)
    plt.imshow(im)
    
w = model.features[3].weight.detach().to('cpu').numpy()
for i in range(w.shape[0]):
    plt.subplot(8,8,i+1)
    im = np.transpose(w[i].squeeze(),(1,2,0))
    im = (im - np.min(im))
    im = im/np.max(im)
    plt.imshow(im)

#%%
#%%
for param in model.parameters():
    param.requires_grad = True
model.eval()

#%%
target_layer = model.layer4[1].conv2
cam = nnu.GradCAM(model, target_layer)
input_tensor = images[[3],:,:,:]
hm = cam.generate_headmap(input_tensor)
im = input_tensor.detach().cpu().numpy().squeeze().transpose((1,2,0))
mn, mx = im.min(), im.max()
im = (255 * (im-mn)/(mx - mn)).astype(np.uint8)
him = nnu.apply_heatmap(im, hm, 0.75)
plt.imshow(him)

#%% Saliency map
sm = nnu.get_saliency_map(model, input_tensor)
im = input_tensor.detach().cpu().numpy().squeeze().transpose((1,2,0))
mn, mx = im.min(), im.max()
im = (255 * (im-mn)/(mx - mn)).astype(np.uint8)
sim = nnu.apply_heatmap(im, sm, 0.25)
plt.imshow(sim)