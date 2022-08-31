#-*- coding:utf-8 -*-  
import math
import torch
import torch.nn as nn
import torchvision
import numpy as np

def range_compressor(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)


class L1MuLoss(nn.Module):
    def __init__(self, mu=5000):
        super(L1MuLoss, self).__init__()
        self.mu = mu

    def forward(self, pred, label):
        mu_pred = range_compressor(pred, self.mu)
        mu_label = range_compressor(label, self.mu)
        return nn.L1Loss()(mu_pred, mu_label)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class JointReconPerceptualLoss(nn.Module):
    def __init__(self, alpha=0.01, mu=5000):
        super(JointReconPerceptualLoss, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.loss_recon = L1MuLoss(self.mu)
        self.loss_vgg = VGGPerceptualLoss(False)

    def forward(self, input, target):
        input_mu = range_compressor(input, self.mu)
        target_mu = range_compressor(target, self.mu)
        loss_recon = self.loss_recon(input, target)
        loss_vgg = self.loss_vgg(input_mu, target_mu)
        loss = loss_recon + self.alpha * loss_vgg
        return loss
