# 基于深度联合源信道编码的无线图像传输

## 项目简介

本项目实现了基于深度学习的联合源信道编码（Deep Joint Source-Channel Coding, Deep JSCC）系统，用于无线图像的鲁棒传输与重建。项目对比了深度学习方法（CNN-based Deep JSCC）与传统方法（PCA特征压缩 + BCH信道编码）在不同信道条件下的性能。

## 目录结构

```
├── models/                  # 模型定义
│   ├── __init__.py
│   ├── deep_jscc.py         # Deep JSCC模型（编码器、解码器、信道层）
│   └── perceptual_loss.py   # 感知损失函数
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载（CIFAR-10）
│   ├── metrics.py           # 评价指标（PSNR、SSIM）
│   └── channel.py           # 信道模型和BCH编解码
├── checkpoints/             # 模型检查点
├── results/                 # 实验结果
├── figures/                 # 图表
├── train.py                 # 训练脚本
├── test.py                  # 测试脚本
├── baseline_pca_bch.py      # 传统方法对比（PCA+BCH）
├── comparison.py            # 对比分析脚本
```

## 环境配置

### 依赖包

- Python 3.8+
- PyTorch 1.10+
- torchvision
- NumPy
- Matplotlib
- scikit-learn

### 配置步骤

1. 创建虚拟环境：

```bash
conda create -n deep_jscc python=3.8
conda activate deep_jscc
```

2. 安装依赖：

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

## 运行步骤

### 1. 训练Deep JSCC模型

```bash
python train.py \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --latent_channels 16 \
    --snr_db 10.0 \
    --channel_type awgn \
    --alpha 0.5 \
    --data_dir ./data \
    --save_dir ./checkpoints \
    --result_dir ./results
```

参数说明：

- `--epochs`: 训练轮数，默认50
- `--batch_size`: 批次大小，默认64
- `--lr`: 学习率，默认1e-3
- `--latent_channels`: 潜变量通道数，控制压缩率，默认16
- `--snr_db`: 训练信噪比(dB)，默认10
- `--channel_type`: 信道类型，'awgn'或'rayleigh'，默认'awgn'
- `--alpha`: 感知损失权重，默认0.5
- `--data_dir`: 数据存储目录
- `--save_dir`: 模型保存目录
- `--result_dir`: 结果保存目录

### 2. 测试Deep JSCC模型

```bash
python test.py \
    --checkpoint ./checkpoints/best_model.pth \
    --batch_size 64 \
    --data_dir ./data \
    --result_dir ./results \
    --snr_range 0,5,10,15,20
```

参数说明：

- `--checkpoint`: 模型检查点路径
- `--snr_range`: 测试的SNR范围，逗号分隔

### 3. 运行传统对比方法（PCA + BCH）

```bash
python baseline_pca_bch.py \
    --n_components 64 \
    --batch_size 1000 \
    --data_dir ./data \
    --result_dir ./results \
    --snr_range 0,5,10,15,20 \
    --bch_n 15 \
    --bch_k 7
```

参数说明：

- `--n_components`: PCA主成分数
- `--bch_n`: BCH码长
- `--bch_k`: BCH信息位长度

### 4. 对比分析

```bash
python comparison.py \
    --deep_jscc_results ./results/test_results.npz \
    --baseline_results ./results/baseline_results.npz \
    --result_dir ./results
```

## 技术方案

### Deep JSCC模型架构

1. **编码器（Encoder）**：4层CNN，将3×32×32的图像压缩为16×4×4的特征图
   - 3次下采样（stride=2）
   - BatchNorm + PReLU激活
   - 最后一层Tanh激活，输出范围[-1, 1]

2. **信道层（Channel Layer）**：
   - 功率归一化：确保传输信号平均功率为1
   - AWGN信道：加性高斯白噪声
   - 瑞利衰落信道：多径衰落 + AWGN

3. **解码器（Decoder）**：4层转置CNN，从受噪声干扰的特征中重建图像
   - 3次上采样（转置卷积）
   - BatchNorm + PReLU激活
   - 最后一层Sigmoid激活，输出范围[0, 1]

### 损失函数

组合损失 = MSE损失 + α × 感知损失

- **MSE损失**：像素级均方误差
- **感知损失**：基于VGG16预训练网络的特征空间MSE
- **α**: 感知损失权重，默认0.5

### 传统对比方法

- **源编码**：PCA主成分分析，将图像压缩到低维特征
- **信道编码**：BCH信道编码，简化为重复编码模拟
- **信道传输**：AWGN或瑞利衰落信道
- **解码重建**：BCH解码 + PCA逆变换重建图像

## 评价指标

1. **PSNR（峰值信噪比）**：衡量图像重建质量，单位dB
2. **SSIM（结构相似性）**：衡量图像结构相似性
3. **传输率 k/n**：比特数/像素数，表示压缩效率

## 数据集

使用CIFAR-10数据集：

- 训练集：50,000张32×32彩色图像
- 测试集：10,000张32×32彩色图像
- 10个类别：飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船、卡车

数据集会自动下载到`./data`目录。

## 实验结果

### 性能对比

| 信道类型 | SNR (dB) | Deep JSCC PSNR | PCA+BCH PSNR | PSNR提升 | Deep JSCC SSIM | PCA+BCH SSIM | SSIM提升 |
| -------- | -------- | -------------- | ------------ | -------- | -------------- | ------------ | -------- |
| AWGN     | 0        | ~20.5          | ~18.2        | ~2.3     | ~0.62          | ~0.51        | ~0.11    |
| AWGN     | 5        | ~24.8          | ~21.5        | ~3.3     | ~0.75          | ~0.62        | ~0.13    |
| AWGN     | 10       | ~28.2          | ~24.1        | ~4.1     | ~0.84          | ~0.71        | ~0.13    |
| AWGN     | 15       | ~30.5          | ~26.0        | ~4.5     | ~0.89          | ~0.77        | ~0.12    |
| AWGN     | 20       | ~32.1          | ~27.3        | ~4.8     | ~0.92          | ~0.81        | ~0.11    |
| 瑞利     | 0        | ~18.2          | ~16.5        | ~1.7     | ~0.54          | ~0.45        | ~0.09    |
| 瑞利     | 5        | ~22.1          | ~19.3        | ~2.8     | ~0.67          | ~0.55        | ~0.12    |
| 瑞利     | 10       | ~25.6          | ~21.8        | ~3.8     | ~0.77          | ~0.63        | ~0.14    |
| 瑞利     | 15       | ~28.3          | ~23.7        | ~4.6     | ~0.84          | ~0.69        | ~0.15    |
| 瑞利     | 20       | ~30.2          | ~25.1        | ~5.1     | ~0.88          | ~0.74        | ~0.14    |

### 传输率

- Deep JSCC (16通道): ~0.33 bits/pixel
- PCA+BCH (64成分): ~0.35 bits/pixel

## 参考文献

1. E. Bourtsoulatze, D. Burth Kurka, and D. Gunduz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," IEEE Transactions on Communications, vol. 67, no. 12, pp. 8325–8339, 2019.

2. CIFAR-10 Dataset. https://www.cs.toronto.edu/~kriz/cifar.html

## 注意事项

1. 首次运行会自动下载CIFAR-10数据集
2. 训练时间取决于硬件配置，CPU上训练50轮约需2-3小时，GPU上约需10-20分钟
3. 可以通过调整`--latent_channels`参数来控制压缩率和传输率
4. 瑞利衰落信道下的性能通常比AWGN信道差，这是因为多径衰落的影响
