"""
测试脚本
测试Deep JSCC模型在不同SNR和信道条件下的性能
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from models import DeepJSCC
from utils import get_cifar10_loaders, calculate_psnr, calculate_ssim


def test_model(model, test_loader, snr_db, channel_type, device):
    """
    测试模型在指定SNR和信道下的性能
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
            
            # 计算指标
            batch_psnr = calculate_psnr(images, recon_images)
            batch_ssim = calculate_ssim(images, recon_images)
            
            total_psnr += batch_psnr * images.size(0)
            total_ssim += batch_ssim * images.size(0)
            num_samples += images.size(0)
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    return avg_psnr, avg_ssim


def visualize_results(model, test_loader, snr_db, channel_type, device, save_path, num_samples=5):
    """
    可视化重建结果
    """
    model.eval()
    
    # 获取一批数据
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(device)
    
    # 重建
    with torch.no_grad():
        recon_images = model(images, snr_db, channel_type)
    
    # 转换为numpy
    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    recon_np = recon_images.cpu().numpy().transpose(0, 2, 3, 1)
    
    # 绘制
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        # 原始图像
        axes[0, i].imshow(images_np[i])
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # 重建图像
        axes[1, i].imshow(recon_np[i])
        psnr = calculate_psnr(images[i], recon_images[i])
        axes[1, i].set_title(f'Recon\nPSNR: {psnr:.1f}dB')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Channel: {channel_type}, SNR: {snr_db} dB', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"可视化结果已保存到 {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Deep JSCC Model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth', 
                        help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--result_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--snr_range', type=str, default='0,5,10,15,20', 
                        help='测试SNR范围，逗号分隔')
    args = parser.parse_args()
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 加载数据
    print("加载CIFAR-10测试集...")
    _, test_loader, classes = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint['args']
    
    model = DeepJSCC(
        latent_channels=model_args.latent_channels,
        channel_type=model_args.channel_type,
        image_size=32
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型加载成功，训练epoch: {checkpoint['epoch']}")
    
    # 传输率
    transmission_rate = model.get_transmission_rate()
    print(f"传输率: {transmission_rate:.4f} bits/pixel")
    
    # 解析SNR范围
    snr_list = [float(x) for x in args.snr_range.split(',')]
    
    # 测试不同信道和SNR
    channel_types = ['awgn', 'rayleigh']
    results = {}
    
    print("\n开始测试...")
    print("-" * 60)
    
    for channel_type in channel_types:
        psnr_list = []
        ssim_list = []
        
        print(f"\n信道类型: {channel_type}")
        
        for snr_db in snr_list:
            psnr, ssim = test_model(model, test_loader, snr_db, channel_type, device)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            print(f"  SNR: {snr_db:5.1f} dB | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
        
        results[channel_type] = {
            'snr': snr_list,
            'psnr': psnr_list,
            'ssim': ssim_list
        }
    
    print("\n" + "-" * 60)
    print("测试完成！")
    
    # 保存结果
    np.savez(os.path.join(args.result_dir, 'test_results.npz'),
             awgn_psnr=results['awgn']['psnr'],
             awgn_ssim=results['awgn']['ssim'],
             rayleigh_psnr=results['rayleigh']['psnr'],
             rayleigh_ssim=results['rayleigh']['ssim'],
             snr_list=snr_list,
             transmission_rate=transmission_rate)
    print(f"测试结果已保存到 {args.result_dir}/test_results.npz")
    
    # 绘制PSNR-SNR曲线
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(snr_list, results['awgn']['psnr'], 'o-', label='AWGN Channel', linewidth=2)
    plt.plot(snr_list, results['rayleigh']['psnr'], 's-', label='Rayleigh Channel', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs SNR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(snr_list, results['awgn']['ssim'], 'o-', label='AWGN Channel', linewidth=2)
    plt.plot(snr_list, results['rayleigh']['ssim'], 's-', label='Rayleigh Channel', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('SSIM')
    plt.title('SSIM vs SNR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.result_dir, 'psnr_snr_curves.png'), dpi=150)
    print(f"PSNR-SNR曲线已保存到 {args.result_dir}/psnr_snr_curves.png")
    
    # 可视化重建结果
    print("\n生成可视化结果...")
    for channel_type in channel_types:
        for snr_db in [5, 15]:
            save_path = os.path.join(args.result_dir, f'visual_{channel_type}_snr{snr_db}.png')
            visualize_results(model, test_loader, snr_db, channel_type, device, save_path)
    
    print("\n所有测试完成！")


if __name__ == '__main__':
    main()
