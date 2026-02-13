# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 08:39:22 2026

@author: Mani Subramaniyan
"""
import torch
import cv2
import numpy as np
import torch.nn.functional as F

#%%
#-----Grad-CAM----
class GradCAM():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, model, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, model, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_headmap(self, input_tensor):
        # Forward pass
        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1).item()
        # backward pass for specific class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        g = self.gradients
        grad_weights = g.mean(dim=[0,2,3])
       
        for i in range(grad_weights.shape[0]):
            self.activations[:,i,:,:] *= grad_weights[i]
        # heat map
        hm = torch.mean(self.activations, dim=1).squeeze()
        hm = F.relu(hm) # keep just positive ones
        hm /= torch.max(hm) # normalize
        
        return hm.detach().cpu().numpy()

def apply_heatmap(original_image, heatmap, alpha, gamma=0):
    hm = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    mn, mx = hm.min(), hm.max()
    hm = (255 * (hm-mn)/(mx - mn))
    hm = np.uint8(255 - hm)
    
    hm_col = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    
    im = cv2.addWeighted(original_image, alpha, hm_col, 1-alpha, gamma)
    
    return im

def get_saliency_map(model, input_tensor):
    input_tensor.requires_grad_()
    out = model(input_tensor)
    class_idx = out.argmax(dim=1).item()
    score = out[0, class_idx]
    model.zero_grad()
    score.backward()
    g = input_tensor.grad.data.abs()
    g, _ = torch.max(g, dim=1)
    g = g.squeeze().cpu().numpy()
    
    return g
                              
def export_onnx_model(model, dummy_input, onnx_file_path):
    """ Export the given model in ONNX format so that it can be visualized using Netron
        Inputs: model - resnet18 model
                onnx_file_path - file name for saving; eg: "resnet18.onnx"
                dummy_input - example: torch.randn(1, 3, 224, 224)
                
                if model is 'slow_fast':
                    dummy_input = [torch.randn(1, 3, 8, 224, 224), torch.randn(1, 3, 32, 224, 224)]
    """    
    # 1. Create a dummy input (Batch=1, Channels=3, H=224, W=224)


    # 2. Export to ONNX   
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,        # Store trained weights in the file
        opset_version=12,          # Use 12+ for better 3D convolution support
        do_constant_folding=True,  # Optimizes the graph
        input_names=['input'],     # Define names for Netron's sidebar
        output_names=['output']
    )
    print(f"Model exported to {onnx_file_path}")   