"""
信道模型工具函数
用于传统方法的信道模拟
"""

import numpy as np


def power_normalize(x):
    """
    功率归一化
    """
    power = np.mean(x ** 2)
    return x / np.sqrt(power + 1e-8)


def awgn_channel(signal, snr_db):
    """
    AWGN信道
    
    Args:
        signal: 输入信号
        snr_db: 信噪比(dB)
    
    Returns:
        受噪声干扰的信号
    """
    # 功率归一化
    signal_norm = power_normalize(signal)
    
    # 计算噪声标准差
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std = np.sqrt(1.0 / snr_linear)
    
    # 添加噪声
    noise = np.random.randn(*signal.shape) * noise_std
    
    return signal_norm + noise


def rayleigh_channel(signal, snr_db):
    """
    瑞利衰落信道
    
    Args:
        signal: 输入信号
        snr_db: 信噪比(dB)
    
    Returns:
        受衰落和噪声干扰的信号
    """
    # 功率归一化
    signal_norm = power_normalize(signal)
    
    # 生成瑞利衰落系数
    h_real = np.random.randn()
    h_imag = np.random.randn()
    h_mag = np.sqrt(h_real ** 2 + h_imag ** 2) / np.sqrt(2.0)
    
    # 信号经过衰落
    signal_faded = signal_norm * h_mag
    
    # 添加AWGN
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std = np.sqrt(1.0 / snr_linear)
    noise = np.random.randn(*signal.shape) * noise_std
    
    return signal_faded + noise


def bch_encode(data, n=15, k=7):
    """
    简化的BCH编码
    实际BCH编码较复杂，用重复编码模拟信道编码效果
    
    Args:
        data: 输入数据
        n: 码长
        k: 信息位长度
    
    Returns:
        编码后的数据
    """
    # 简化：使用重复编码
    repetition = n // k
    encoded = np.repeat(data, repetition)
    return encoded


def bch_decode(encoded, n=15, k=7):
    """
    简化的BCH解码
    使用多数判决解码
    
    Args:
        encoded: 编码后的数据
        n: 码长
        k: 信息位长度
    
    Returns:
        解码后的数据
    """
    repetition = n // k
    # 重塑为 [k, repetition]
    reshaped = encoded.reshape(-1, repetition)
    # 取平均（多数判决的连续版本）
    decoded = np.mean(reshaped, axis=1)
    return decoded


if __name__ == '__main__':
    # 测试信道
    signal = np.random.randn(100)
    
    # AWGN信道
    noisy_awgn = awgn_channel(signal, 10)
    print(f"AWGN信道 - 输入功率: {np.mean(signal**2):.4f}, 输出功率: {np.mean(noisy_awgn**2):.4f}")
    
    # 瑞利信道
    noisy_rayleigh = rayleigh_channel(signal, 10)
    print(f"瑞利信道 - 输入功率: {np.mean(signal**2):.4f}, 输出功率: {np.mean(noisy_rayleigh**2):.4f}")
    
    # 测试BCH编解码
    data = np.random.rand(10)
    encoded = bch_encode(data, 15, 5)
    decoded = bch_decode(encoded, 15, 5)
    print(f"BCH编解码 - 原始长度: {len(data)}, 编码长度: {len(encoded)}, 解码长度: {len(decoded)}")
    print(f"编解码误差: {np.mean((data - decoded)**2):.6f}")
