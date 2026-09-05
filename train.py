"""
训练脚本
训练Deep JSCC模型
"""

import os
import argparse
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from models import DeepJSCC, CombinedLoss
from utils import get_cifar10_loaders, calculate_psnr, calculate_ssim


def train_epoch(model, train_loader, optimizer, loss_fn, snr_db, channel_type, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_perceptual = 0.0
    num_batches = 0
    
    for images, _ in train_loader:
        images = images.to(device)
        
        # 前向传播
        recon_images = model(images, snr_db, channel_type)
        
        # 计算损失
        loss, mse_loss, perc_loss = loss_fn(recon_images, images)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_perceptual += perc_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_perceptual = total_perceptual / num_batches
    
    return avg_loss, avg_mse, avg_perceptual


def validate(model, test_loader, loss_fn, snr_db, channel_type, device):
    """
    验证模型
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    num_samples = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            
            # 前向传播
            recon_images = model(images, snr_db, channel_type)
            
            # 计算损失
            loss, _, _ = loss_fn(recon_images, images)
            
            # 计算指标
            batch_psnr = calculate_psnr(images, recon_images)
            batch_ssim = calculate_ssim(images, recon_images)
            
            total_loss += loss.item()
            total_psnr += batch_psnr * images.size(0)
            total_ssim += batch_ssim * images.size(0)
            num_batches += 1
            num_samples += images.size(0)
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    return avg_loss, avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description='Train Deep JSCC Model')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--latent_channels', type=int, default=16, help='潜变量通道数')
    parser.add_argument('--snr_db', type=float, default=10.0, help='训练信噪比(dB)')
    parser.add_argument('--channel_type', type=str, default='awgn', choices=['awgn', 'rayleigh'])
    parser.add_argument('--alpha', type=float, default=0.5, help='感知损失权重')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default='./results', help='结果保存目录')
    args = parser.parse_args()
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 加载数据
    print("加载CIFAR-10数据集...")
    train_loader, test_loader, classes = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # 创建模型
    model = DeepJSCC(
        latent_channels=args.latent_channels,
        channel_type=args.channel_type,
        image_size=32
    ).to(device)
    
    # 计算传输率
    transmission_rate = model.get_transmission_rate()
    print(f"传输率: {transmission_rate:.4f} bits/pixel")
    
    # 创建损失函数
    loss_fn = CombinedLoss(alpha=args.alpha, device=device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 训练历史
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    best_psnr = 0.0
    
    # 训练循环
    print(f"\n开始训练，共{args.epochs}轮...")
    print(f"训练SNR: {args.snr_db} dB, 信道类型: {args.channel_type}")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_mse, train_perc = train_epoch(
            model, train_loader, optimizer, loss_fn, 
            args.snr_db, args.channel_type, device
        )
        
        # 验证
        val_loss, val_psnr, val_ssim = validate(
            model, test_loader, loss_fn,
            args.snr_db, args.channel_type, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        val_ssims.append(val_ssim)
        
        # 保存最佳模型
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'args': args,
            }, os.path.join(args.save_dir, 'best_model.pth'))
        
        # 打印信息
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{args.epochs}] "
                  f"Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Perc: {train_perc:.4f}) "
                  f"Val Loss: {val_loss:.4f} "
                  f"Val PSNR: {val_psnr:.2f} dB "
                  f"Val SSIM: {val_ssim:.4f}")
    
    print("-" * 60)
    print(f"训练完成！最佳PSNR: {best_psnr:.2f} dB")
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_psnr': val_psnrs[-1],
        'val_ssim': val_ssims[-1],
        'args': args,
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_psnrs, label='Val PSNR', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(val_ssims, label='Val SSIM', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Validation SSIM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, 'training_curves.png'), dpi=150)
    print(f"训练曲线已保存到 {args.result_dir}/training_curves.png")
    
    # 保存训练历史
    np.savez(os.path.join(args.result_dir, 'training_history.npz'),
             train_losses=train_losses,
             val_losses=val_losses,
             val_psnrs=val_psnrs,
             val_ssims=val_ssims)
    print(f"训练历史已保存到 {args.result_dir}/training_history.npz")


if __name__ == '__main__':
    main()
