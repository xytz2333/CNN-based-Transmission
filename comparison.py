"""
对比分析脚本
对比Deep JSCC和PCA+BCH方法的性能
生成对比图表
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(deep_jscc_results, baseline_results, save_dir):
    """
    绘制对比图
    """
    snr_list = deep_jscc_results['snr_list']
    
    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR对比
    ax = axes[0]
    ax.plot(snr_list, deep_jscc_results['awgn_psnr'], 'o-', color='#1f77b4', 
            linewidth=2, label='Deep JSCC (AWGN)')
    ax.plot(snr_list, baseline_results['awgn_psnr'], 's--', color='#ff7f0e', 
            linewidth=2, label='PCA+BCH (AWGN)')
    ax.plot(snr_list, deep_jscc_results['rayleigh_psnr'], '^-', color='#2ca02c', 
            linewidth=2, label='Deep JSCC (Rayleigh)')
    ax.plot(snr_list, baseline_results['rayleigh_psnr'], 'v--', color='#d62728', 
            linewidth=2, label='PCA+BCH (Rayleigh)')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # SSIM对比
    ax = axes[1]
    ax.plot(snr_list, deep_jscc_results['awgn_ssim'], 'o-', color='#1f77b4', 
            linewidth=2, label='Deep JSCC (AWGN)')
    ax.plot(snr_list, baseline_results['awgn_ssim'], 's--', color='#ff7f0e', 
            linewidth=2, label='PCA+BCH (AWGN)')
    ax.plot(snr_list, deep_jscc_results['rayleigh_ssim'], '^-', color='#2ca02c', 
            linewidth=2, label='Deep JSCC (Rayleigh)')
    ax.plot(snr_list, baseline_results['rayleigh_ssim'], 'v--', color='#d62728', 
            linewidth=2, label='PCA+BCH (Rayleigh)')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title('SSIM Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'comparison_plot.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"对比图已保存到 {save_path}")


def print_comparison_table(deep_jscc_results, baseline_results):
    """
    打印对比表格
    """
    snr_list = deep_jscc_results['snr_list']
    
    print("\n" + "=" * 80)
    print("性能对比表")
    print("=" * 80)
    
    # AWGN信道
    print("\n【AWGN信道】")
    print(f"{'SNR(dB)':<10} {'Deep JSCC PSNR':<18} {'PCA+BCH PSNR':<18} {'PSNR提升':<12} "
          f"{'Deep JSCC SSIM':<18} {'PCA+BCH SSIM':<18} {'SSIM提升':<12}")
    print("-" * 80)
    
    for i, snr in enumerate(snr_list):
        dj_psnr = deep_jscc_results['awgn_psnr'][i]
        bl_psnr = baseline_results['awgn_psnr'][i]
        dj_ssim = deep_jscc_results['awgn_ssim'][i]
        bl_ssim = baseline_results['awgn_ssim'][i]
        
        psnr_gain = dj_psnr - bl_psnr
        ssim_gain = dj_ssim - bl_ssim
        
        print(f"{snr:<10.1f} {dj_psnr:<18.2f} {bl_psnr:<18.2f} {psnr_gain:<12.2f} "
              f"{dj_ssim:<18.4f} {bl_ssim:<18.4f} {ssim_gain:<12.4f}")
    
    # 瑞利信道
    print("\n【瑞利衰落信道】")
    print(f"{'SNR(dB)':<10} {'Deep JSCC PSNR':<18} {'PCA+BCH PSNR':<18} {'PSNR提升':<12} "
          f"{'Deep JSCC SSIM':<18} {'PCA+BCH SSIM':<18} {'SSIM提升':<12}")
    print("-" * 80)
    
    for i, snr in enumerate(snr_list):
        dj_psnr = deep_jscc_results['rayleigh_psnr'][i]
        bl_psnr = baseline_results['rayleigh_psnr'][i]
        dj_ssim = deep_jscc_results['rayleigh_ssim'][i]
        bl_ssim = baseline_results['rayleigh_ssim'][i]
        
        psnr_gain = dj_psnr - bl_psnr
        ssim_gain = dj_ssim - bl_ssim
        
        print(f"{snr:<10.1f} {dj_psnr:<18.2f} {bl_psnr:<18.2f} {psnr_gain:<12.2f} "
              f"{dj_ssim:<18.4f} {bl_ssim:<18.4f} {ssim_gain:<12.4f}")
    
    # 传输率对比
    print("\n【传输率对比】")
    print(f"Deep JSCC: {deep_jscc_results['transmission_rate']:.4f} bits/pixel")
    print(f"PCA+BCH:   {baseline_results['transmission_rate']:.4f} bits/pixel")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Comparison Analysis')
    parser.add_argument('--deep_jscc_results', type=str, default='./results/test_results.npz',
                        help='Deep JSCC测试结果路径')
    parser.add_argument('--baseline_results', type=str, default='./results/baseline_results.npz',
                        help='基线方法结果路径')
    parser.add_argument('--result_dir', type=str, default='./results', help='结果保存目录')
    args = parser.parse_args()
    
    # 加载结果
    print("加载结果...")
    deep_jscc_results = np.load(args.deep_jscc_results)
    baseline_results = np.load(args.baseline_results)
    
    # 打印对比表格
    print_comparison_table(deep_jscc_results, baseline_results)
    
    # 绘制对比图
    print("\n生成对比图...")
    plot_comparison(deep_jscc_results, baseline_results, args.result_dir)
    
    # 保存对比数据
    comparison_data = {
        'snr_list': deep_jscc_results['snr_list'],
        'deep_jscc_awgn_psnr': deep_jscc_results['awgn_psnr'],
        'deep_jscc_awgn_ssim': deep_jscc_results['awgn_ssim'],
        'deep_jscc_rayleigh_psnr': deep_jscc_results['rayleigh_psnr'],
        'deep_jscc_rayleigh_ssim': deep_jscc_results['rayleigh_ssim'],
        'baseline_awgn_psnr': baseline_results['awgn_psnr'],
        'baseline_awgn_ssim': baseline_results['awgn_ssim'],
        'baseline_rayleigh_psnr': baseline_results['rayleigh_psnr'],
        'baseline_rayleigh_ssim': baseline_results['rayleigh_ssim'],
        'deep_jscc_rate': deep_jscc_results['transmission_rate'],
        'baseline_rate': baseline_results['transmission_rate'],
    }
    
    np.savez(os.path.join(args.result_dir, 'comparison_data.npz'), **comparison_data)
    print(f"对比数据已保存到 {args.result_dir}/comparison_data.npz")
    
    print("\n对比分析完成！")


if __name__ == '__main__':
    main()
