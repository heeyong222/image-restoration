import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


class PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu3_3'], weights=[1.0]):
        super(PerceptualLoss, self).__init__()
        # Load pretrained VGG16 model
        self.vgg = models.vgg16(pretrained=True).features
        # Move VGG to GPU if available
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()
        self.vgg.eval()  # set to evaluation mode
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.layers = layers
        self.weights = weights
        # Map layer names to indices in VGG16
        self.layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
        }
    
    def forward(self, x, y):
        loss = 0.0
        # Ensure input is on the same device as vgg
        x = x.to(next(self.vgg.parameters()).device)
        y = y.to(next(self.vgg.parameters()).device)
        for layer, weight in zip(self.layers, self.weights):
            layer_index = self.layer_map[layer]
            # Extract features up to the specified layer
            x_features = self.vgg[:layer_index+1](x)
            y_features = self.vgg[:layer_index+1](y)
            loss += weight * nn.functional.l1_loss(x_features, y_features)
        return loss
    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_loss(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)