"""
图像质量评价指标
PSNR和SSIM
"""

import torch
import torch.nn.functional as F
import numpy as np


def calculate_psnr(img1, img2):
    """
    计算峰值信噪比
    
    Args:
        img1: 图像1 [B, C, H, W] 或 [C, H, W] 或 [H, W, C] numpy数组
        img2: 图像2
    
    Returns:
        PSNR值 (dB)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # 确保是float类型
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 计算MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # 假设像素值范围是[0, 1]
    PIXEL_MAX = 1.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    
    return psnr


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    计算结构相似性
    基于PyTorch实现
    
    Args:
        img1: 图像1 [B, C, H, W] tensor或numpy数组
        img2: 图像2 [B, C, H, W] tensor或numpy数组
        window_size: 高斯窗口大小
        size_average: 是否对batch和channel取平均
    
    Returns:
        SSIM值
    """
    if not isinstance(img1, torch.Tensor):
        img1 = torch.tensor(img1)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.tensor(img2)
    
    # 统一转换为float32类型，避免类型不匹配错误
    img1 = img1.float()
    img2 = img2.float()
    
    # 确保是4D张量 [B, C, H, W]
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    # 通道数
    C = img1.size(1)
    
    # 创建高斯窗口
    def gaussian_window(window_size, sigma=1.5):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
                             for x in range(window_size)])
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.float().unsqueeze(0).unsqueeze(0)
        window = window / window.sum()
        return window
    
    window = gaussian_window(window_size).to(img1.device)
    window = window.expand(C, 1, window_size, window_size).contiguous()
    
    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=C) - mu1_mu2
    
    # SSIM参数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(dim=(1, 2, 3))


def evaluate_model(model, test_loader, snr_db, channel_type='awgn', device='cpu'):
    """
    在测试集上评估模型性能
    
    Args:
        model: Deep JSCC模型
        test_loader: 测试数据加载器
        snr_db: 信噪比(dB)
        channel_type: 信道类型
        device: 计算设备
    
    Returns:
        平均PSNR, 平均SSIM
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            
            # 前向传播
            recon_images = model(images, snr_db, channel_type)
            
            # 计算PSNR
            batch_psnr = calculate_psnr(images, recon_images)
            total_psnr += batch_psnr * images.size(0)
            
            # 计算SSIM
            batch_ssim = calculate_ssim(images, recon_images)
            total_ssim += batch_ssim * images.size(0)
            
            num_samples += images.size(0)
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    return avg_psnr, avg_ssim


if __name__ == '__main__':
    # 测试指标计算
    img1 = torch.rand(4, 3, 32, 32)
    img2 = torch.rand(4, 3, 32, 32)
    
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    # 相同图像的SSIM应该为1
    ssim_same = calculate_ssim(img1, img1)
    print(f"Same image SSIM: {ssim_same:.4f}")
