"""
数据加载器
加载CIFAR-10数据集
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_loaders(data_dir='./data', batch_size=64, num_workers=2):
    """
    获取CIFAR-10数据集的训练集和测试集加载器
    
    Args:
        data_dir: 数据存储目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, test_loader
    """
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 转换为[0, 1]范围的张量
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True,
        download=True, 
        transform=transform_train
    )
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 加载测试集
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False,
        download=True, 
        transform=transform_test
    )
    test_loader = DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # CIFAR-10类别名称
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    print(f"类别数: {len(classes)}")
    
    return train_loader, test_loader, classes


if __name__ == '__main__':
    # 测试数据加载
    train_loader, test_loader, classes = get_cifar10_loaders(batch_size=32)
    
    # 查看一个批次的数据
    images, labels = next(iter(train_loader))
    print(f"图像批次形状: {images.shape}")
    print(f"标签批次形状: {labels.shape}")
    print(f"图像值范围: [{images.min():.4f}, {images.max():.4f}]")
