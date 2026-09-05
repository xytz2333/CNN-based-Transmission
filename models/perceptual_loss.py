"""
感知损失（Perceptual Loss）
使用预训练的VGG网络提取特征，计算特征空间的MSE损失
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    基于VGG的感知损失
    """
    def __init__(self, layer_name='relu3_3', device='cpu'):
        super(PerceptualLoss, self).__init__()
        self.device = device
        
        # 加载预训练的VGG16
        vgg = models.vgg16(pretrained=True).features.to(device)
        vgg.eval()
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        # 提取指定层之前的网络
        self.vgg_layers = vgg[:self._get_layer_index(layer_name)]
        self.layer_name = layer_name
        
        # 归一化参数（ImageNet均值和标准差）
        # 使用register_buffer确保张量会随模型一起移动设备
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def _get_layer_index(self, layer_name):
        """
        根据层名获取VGG中的索引
        """
        layer_dict = {
            'relu1_1': 2,
            'relu1_2': 4,
            'relu2_1': 7,
            'relu2_2': 9,
            'relu3_1': 12,
            'relu3_2': 14,
            'relu3_3': 16,
            'relu4_1': 19,
            'relu4_2': 21,
            'relu4_3': 23,
        }
        return layer_dict.get(layer_name, 16)  # 默认relu3_3
    
    def forward(self, x, y):
        """
        计算感知损失
        Args:
            x: 重建图像 [B, 3, H, W]，范围[0, 1]
            y: 原始图像 [B, 3, H, W]，范围[0, 1]
        Returns:
            感知损失值
        """
        # 归一化到ImageNet标准
        x_norm = (x - self.mean) / self.std
        y_norm = (y - self.mean) / self.std
        
        # 提取特征
        x_feat = self.vgg_layers(x_norm)
        y_feat = self.vgg_layers(y_norm)
        
        # 计算MSE损失
        loss = nn.functional.mse_loss(x_feat, y_feat)
        
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失：MSE + 感知损失
    """
    def __init__(self, alpha=0.5, device='cpu'):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # 感知损失权重
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(device=device)
    
    def forward(self, x, y):
        """
        计算组合损失
        Args:
            x: 重建图像
            y: 原始图像
        Returns:
            总损失, MSE损失, 感知损失
        """
        mse = self.mse_loss(x, y)
        perceptual = self.perceptual_loss(x, y)
        total = mse + self.alpha * perceptual
        return total, mse, perceptual


if __name__ == '__main__':
    # 测试感知损失
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = CombinedLoss(alpha=0.5, device=device)
    
    x = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)
    
    total_loss, mse_loss, perc_loss = loss_fn(x, y)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"MSE loss: {mse_loss.item():.4f}")
    print(f"Perceptual loss: {perc_loss.item():.4f}")
