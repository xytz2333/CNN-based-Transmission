"""
传统对比方法：PCA特征压缩 + BCH信道编码
用于与Deep JSCC方法进行对比

"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI环境下使用Agg后端
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import get_cifar10_loaders, calculate_psnr, calculate_ssim
from utils.channel import awgn_channel, rayleigh_channel, bch_encode, bch_decode


def fit_pca(train_images, n_components):
    """
    在训练集上拟合PCA模型
    
    Args:
        train_images: 训练图像 [N, C, H, W]
        n_components: PCA主成分数量
    
    Returns:
        拟合好的PCA模型
    """
    N, C, H, W = train_images.shape
    
    # 将图像展平为向量
    images_flat = train_images.reshape(N, -1)
    
    # 拟合PCA
    pca = PCA(n_components=n_components)
    pca.fit(images_flat)
    
    print(f"PCA拟合完成，解释方差比: {sum(pca.explained_variance_ratio_):.4f}")
    
    return pca


def pca_transform(images, pca):
    """
    使用已拟合的PCA模型进行压缩
    
    Args:
        images: 图像数据 [N, C, H, W]
        pca: 已拟合的PCA模型
    
    Returns:
        压缩后的特征 [N, n_components]
    """
    N, C, H, W = images.shape
    
    # 将图像展平为向量
    images_flat = images.reshape(N, -1)
    
    # PCA变换
    features = pca.transform(images_flat)
    
    return features


def pca_reconstruct(features, pca, original_shape):
    """
    PCA重建
    
    Args:
        features: 压缩特征 [N, n_components]
        pca: 已拟合的PCA模型
        original_shape: 原始图像形状 (C, H, W)
    
    Returns:
        重建图像 [N, C, H, W]
    """
    # 逆变换
    images_flat = pca.inverse_transform(features)
    
    # 重塑为图像形状
    N = features.shape[0]
    images = images_flat.reshape(N, *original_shape)
    
    # 裁剪到[0, 1]范围
    images = np.clip(images, 0, 1)
    
    # 转换为float32，减少内存使用并与PyTorch默认类型一致
    images = images.astype(np.float32)
    
    return images


def pca_bch_pipeline(images, pca, snr_db, channel_type='awgn', bch_n=15, bch_k=7):
    """
    PCA + BCH 完整传输流程
    
    Args:
        images: 输入图像 [N, C, H, W]
        pca: 已拟合的PCA模型
        snr_db: 信噪比(dB)
        channel_type: 信道类型
        bch_n: BCH码长
        bch_k: BCH信息位长度
    
    Returns:
        重建图像
    """
    N, C, H, W = images.shape
    original_shape = (C, H, W)
    
    # 1. PCA压缩
    features = pca_transform(images, pca)
    
    # 2. BCH编码
    # 对每个特征进行编码
    encoded_features = []
    for i in range(N):
        encoded = bch_encode(features[i], n=bch_n, k=bch_k)
        encoded_features.append(encoded)
    encoded_features = np.array(encoded_features)
    
    # 3. 信道传输
    if channel_type == 'awgn':
        noisy_features = awgn_channel(encoded_features, snr_db)
    elif channel_type == 'rayleigh':
        noisy_features = rayleigh_channel(encoded_features, snr_db)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")
    
    # 4. BCH解码
    decoded_features = []
    for i in range(N):
        decoded = bch_decode(noisy_features[i], n=bch_n, k=bch_k)
        decoded_features.append(decoded)
    decoded_features = np.array(decoded_features)
    
    # 5. PCA重建
    recon_images = pca_reconstruct(decoded_features, pca, original_shape)
    
    return recon_images


def calculate_transmission_rate(n_components, image_size=32, bch_n=15, bch_k=7):
    """
    计算传输率
    
    Args:
        n_components: PCA主成分数
        image_size: 图像尺寸
        bch_n: BCH码长
        bch_k: BCH信息位长度
    
    Returns:
        传输率 bits/pixel
    """
    # 源像素数
    source_pixels = image_size * image_size * 3
    
    # 信道符号数
    # 每个PCA成分编码为bch_n个符号
    channel_symbols = n_components * (bch_n / bch_k)
    
    # 假设每个符号用16比特表示
    bits_per_pixel = (channel_symbols * 16) / source_pixels
    
    return bits_per_pixel


def main():
    parser = argparse.ArgumentParser(description='PCA + BCH Baseline')
    parser.add_argument('--n_components', type=int, default=64, help='PCA主成分数')
    parser.add_argument('--batch_size', type=int, default=1000, help='批次大小')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--result_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--snr_range', type=str, default='0,5,10,15,20', 
                        help='测试SNR范围，逗号分隔')
    parser.add_argument('--bch_n', type=int, default=15, help='BCH码长')
    parser.add_argument('--bch_k', type=int, default=7, help='BCH信息位长度')
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 加载数据
    print("加载CIFAR-10数据集...")
    train_loader, test_loader, classes = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # 获取训练数据，用于拟合PCA
    print("加载训练集用于PCA拟合...")
    train_images = []
    for images, _ in train_loader:
        train_images.append(images.numpy())
    train_images = np.concatenate(train_images, axis=0)
    print(f"训练集大小: {train_images.shape}")
    
    # 获取测试数据
    print("加载测试集...")
    test_images = []
    for images, _ in test_loader:
        test_images.append(images.numpy())
    test_images = np.concatenate(test_images, axis=0)
    print(f"测试集大小: {test_images.shape}")
    
    # 在训练集上拟合PCA模型
    print(f"\n在训练集上拟合PCA (n_components={args.n_components})...")
    pca = fit_pca(train_images, args.n_components)
    
    # 计算传输率
    transmission_rate = calculate_transmission_rate(
        args.n_components, 
        image_size=32,
        bch_n=args.bch_n, 
        bch_k=args.bch_k
    )
    print(f"传输率: {transmission_rate:.4f} bits/pixel")
    
    # 解析SNR范围
    snr_list = [float(x) for x in args.snr_range.split(',')]
    
    # 测试不同信道和SNR
    channel_types = ['awgn', 'rayleigh']
    results = {}
    
    print("\n开始测试PCA+BCH方法...")
    print("-" * 60)
    
    for channel_type in channel_types:
        psnr_list = []
        ssim_list = []
        
        print(f"\n信道类型: {channel_type}")
        
        for snr_db in snr_list:
            # 运行PCA+BCH（使用训练好的PCA模型）
            recon_images = pca_bch_pipeline(
                test_images, 
                pca=pca,
                snr_db=snr_db,
                channel_type=channel_type,
                bch_n=args.bch_n,
                bch_k=args.bch_k
            )
            
            # 计算指标
            psnr = calculate_psnr(test_images, recon_images)
            ssim = calculate_ssim(test_images, recon_images)
            
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
    np.savez(os.path.join(args.result_dir, 'baseline_results.npz'),
             awgn_psnr=results['awgn']['psnr'],
             awgn_ssim=results['awgn']['ssim'],
             rayleigh_psnr=results['rayleigh']['psnr'],
             rayleigh_ssim=results['rayleigh']['ssim'],
             snr_list=snr_list,
             n_components=args.n_components,
             transmission_rate=transmission_rate)
    print(f"基线结果已保存到 {args.result_dir}/baseline_results.npz")
    
    # 可视化重建结果
    print("\n生成可视化结果...")
    num_samples = 5
    sample_images = test_images[:num_samples]
    
    for channel_type in channel_types:
        for snr_db in [5, 15]:
            # 使用训练好的PCA模型进行重建
            recon_samples = pca_bch_pipeline(
                sample_images,
                pca=pca,
                snr_db=snr_db,
                channel_type=channel_type,
                bch_n=args.bch_n,
                bch_k=args.bch_k
            )
            
            # 绘制
            fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
            
            for i in range(num_samples):
                # 原始图像
                axes[0, i].imshow(sample_images[i].transpose(1, 2, 0))
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # 重建图像
                axes[1, i].imshow(recon_samples[i].transpose(1, 2, 0))
                psnr = calculate_psnr(sample_images[i:i+1], recon_samples[i:i+1])
                axes[1, i].set_title(f'Recon\nPSNR: {psnr:.1f}dB')
                axes[1, i].axis('off')
            
            plt.suptitle(f'PCA+BCH - {channel_type} Channel, SNR: {snr_db} dB', fontsize=12)
            plt.tight_layout()
            save_path = os.path.join(args.result_dir, f'baseline_visual_{channel_type}_snr{snr_db}.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"可视化结果已保存到 {save_path}")
    
    print("\n所有基线测试完成！")


if __name__ == '__main__':
    main()
