"""
Deep Joint Source-Channel Coding (Deep JSCC) Model
基于CNN的深度联合源信道编码模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    CNN编码器：将图像压缩为信道特征
    输入: [B, 3, H, W]
    输出: [B, C, H', W'] 压缩后的特征图
    """
    def __init__(self, in_channels=3, latent_channels=16):
        super(Encoder, self).__init__()
        
        # 编码器网络
        self.encoder = nn.Sequential(
            # 第一层: 3 -> 32
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            
            # 第二层: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
            # 第三层: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            
            # 第四层: 128 -> latent_channels (压缩到目标维度)
            nn.Conv2d(128, latent_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(latent_channels),
            nn.Tanh()  # 输出范围[-1, 1]，便于信道传输
        )
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """
    CNN解码器：从受噪声干扰的特征中重建图像
    输入: [B, C, H', W']
    输出: [B, 3, H, W] 重建图像
    """
    def __init__(self, out_channels=3, latent_channels=16):
        super(Decoder, self).__init__()
        
        # 解码器网络
        self.decoder = nn.Sequential(
            # 第一层: latent_channels -> 128
            nn.Conv2d(latent_channels, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            
            # 第二层: 128 -> 64，上采样
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
            # 第三层: 64 -> 32，上采样
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            
            # 第四层: 32 -> out_channels，上采样
            nn.ConvTranspose2d(32, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # 输出范围[0, 1]
        )
    
    def forward(self, x):
        return self.decoder(x)


class ChannelLayer(nn.Module):
    """
    信道层：模拟无线信道传输
    支持AWGN信道和瑞利衰落信道
    """
    def __init__(self, channel_type='awgn'):
        super(ChannelLayer, self).__init__()
        self.channel_type = channel_type
    
    def power_normalize(self, x):
        """
        功率归一化：确保传输信号的平均功率为1
        """
        # 计算每个样本的功率
        power = torch.mean(x ** 2, dim=(1, 2, 3), keepdim=True)
        # 归一化
        x_normalized = x / torch.sqrt(power + 1e-8)
        return x_normalized
    
    def awgn_channel(self, x, snr_db):
        """
        AWGN信道：加性高斯白噪声
        """
        # 计算噪声标准差
        # SNR(dB) = 10 * log10(P_signal / P_noise)
        # P_noise = P_signal / 10^(SNR/10)
        # 由于信号功率归一化为1，所以噪声方差 = 1 / 10^(SNR/10)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = torch.sqrt(torch.tensor(1.0 / snr_linear, device=x.device))
        
        # 添加高斯噪声
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    def rayleigh_channel(self, x, snr_db):
        """
        瑞利衰落信道：多径衰落 + AWGN
        """
        batch_size = x.shape[0]
        
        # 生成瑞利衰落系数（每个样本一个衰落系数，简化为平坦衰落）
        # 瑞利分布 = sqrt(N(0,1)^2 + N(0,1)^2) / sqrt(2)
        h_real = torch.randn(batch_size, 1, 1, 1, device=x.device)
        h_imag = torch.randn(batch_size, 1, 1, 1, device=x.device)
        h_mag = torch.sqrt(h_real ** 2 + h_imag ** 2) / torch.sqrt(torch.tensor(2.0, device=x.device))
        
        # 信号经过衰落信道
        x_faded = x * h_mag
        
        # 添加AWGN
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = torch.sqrt(torch.tensor(1.0 / snr_linear, device=x.device))
        noise = torch.randn_like(x_faded) * noise_std
        
        return x_faded + noise
    
    def forward(self, x, snr_db, channel_type=None):
        """
        前向传播
        Args:
            x: 输入信号
            snr_db: 信噪比(dB)
            channel_type: 信道类型，'awgn'或'rayleigh'，为None时使用默认值
        Returns:
            受信道影响后的信号
        """
        if channel_type is None:
            channel_type = self.channel_type
        
        # 功率归一化
        x_normalized = self.power_normalize(x)
        
        # 通过信道
        if channel_type == 'awgn':
            x_out = self.awgn_channel(x_normalized, snr_db)
        elif channel_type == 'rayleigh':
            x_out = self.rayleigh_channel(x_normalized, snr_db)
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")
        
        return x_out


class DeepJSCC(nn.Module):
    """
    深度联合源信道编码模型
    端到端训练：编码器 -> 信道 -> 解码器
    """
    def __init__(self, in_channels=3, out_channels=3, latent_channels=16, 
                 channel_type='awgn', image_size=32):
        super(DeepJSCC, self).__init__()
        
        self.encoder = Encoder(in_channels, latent_channels)
        self.decoder = Decoder(out_channels, latent_channels)
        self.channel = ChannelLayer(channel_type)
        self.latent_channels = latent_channels
        self.image_size = image_size
        
        # 计算压缩后的特征图尺寸
        # 3次下采样(stride=2)，所以尺寸变为原来的1/8
        self.feature_size = image_size // 8
    
    def encode(self, x):
        """编码"""
        return self.encoder(x)
    
    def decode(self, x):
        """解码"""
        return self.decoder(x)
    
    def forward(self, x, snr_db, channel_type=None):
        """
        端到端前向传播
        Args:
            x: 输入图像 [B, 3, H, W]
            snr_db: 信噪比(dB)
            channel_type: 信道类型
        Returns:
            重建图像
        """
        # 编码
        z = self.encode(x)
        
        # 通过信道
        z_noisy = self.channel(z, snr_db, channel_type)
        
        # 解码
        x_recon = self.decode(z_noisy)
        
        return x_recon
    
    def get_transmission_rate(self):
        """
        计算传输率 k/n
        k: 源符号数（图像像素数 * 通道数）
        n: 信道符号数（特征元素数）
        假设每个特征元素用16比特传输（简化计算）
        """
        # 源符号数：图像像素数 * 通道数
        source_symbols = self.image_size * self.image_size * 3
        
        # 信道符号数：特征图元素数
        channel_symbols = self.latent_channels * self.feature_size * self.feature_size
        
        # 传输率：比特数/像素数
        # 假设每个信道符号用16比特表示（简化）
        bits_per_pixel = (channel_symbols * 16) / source_symbols
        
        return bits_per_pixel


if __name__ == '__main__':
    # 测试模型
    model = DeepJSCC(latent_channels=16, image_size=32)
    x = torch.randn(2, 3, 32, 32)
    
    # 测试AWGN信道
    x_recon = model(x, snr_db=10, channel_type='awgn')
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_recon.shape}")
    print(f"Transmission rate: {model.get_transmission_rate():.4f} bits/pixel")
    
    # 测试瑞利信道
    x_recon_rayleigh = model(x, snr_db=10, channel_type='rayleigh')
    print(f"Rayleigh output shape: {x_recon_rayleigh.shape}")
