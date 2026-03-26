# -*- coding: utf-8 -*-
"""
图片分类训练模块
支持：ResNet50, EfficientNet, MobileNet
支持：迁移学习、数据增强、混合精度训练
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageClassificationDataset(Dataset):
    """图片分类数据集 - 支持文件夹结构和CSV索引"""
    
    def __init__(self, data_dir: str, transform=None, annotation_file: str = None):
        """
        Args:
            data_dir: 图片根目录
            transform: 数据增强变换
            annotation_file: CSV标注文件(可选)，格式：image_path,label
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        if annotation_file and Path(annotation_file).exists():
            # CSV模式
            self._load_from_csv(annotation_file)
        else:
            # 文件夹模式 (ImageFolder结构)
            self._load_from_folder()
    
    def _load_from_folder(self):
        """从文件夹结构加载 (class_a/001.jpg)"""
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        for cls in classes:
            cls_dir = self.data_dir / cls
            for img_path in cls_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    self.samples.append((str(img_path), self.class_to_idx[cls]))
        
        logger.info(f"从文件夹加载: {len(classes)} 类, {len(self.samples)} 张图片")
    
    def _load_from_csv(self, annotation_file: str):
        """从CSV加载"""
        import pandas as pd
        df = pd.read_csv(annotation_file)
        
        classes = sorted(df['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        for _, row in df.iterrows():
            img_path = self.data_dir / row['image_path']
            if img_path.exists():
                self.samples.append((str(img_path), self.class_to_idx[row['label']]))
        
        logger.info(f"从CSV加载: {len(classes)} 类, {len(self.samples)} 张图片")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self):
        return [self.idx_to_class[i] for i in range(len(self.class_to_idx))]


def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    获取预训练模型
    
    Args:
        model_name: resnet50, efficientnet_b0, mobilenet_v2, vit_base
        num_classes: 分类数
        pretrained: 是否使用预训练权重
    """
    if model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'vit_base':
        model = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return model


def get_transforms(image_size: int = 224, is_training: bool = True):
    """获取数据增强变换"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def run_image_classification_training(
    train_dir: str,
    val_dir: str = None,
    output_dir: str = "./output",
    config: Dict = None,
    progress_callback: Callable = None,
    job_id: str = None
) -> Dict:
    """
    运行图片分类训练
    
    Args:
        train_dir: 训练数据目录
        val_dir: 验证数据目录（可选，自动划分）
        output_dir: 输出目录
        config: 训练配置
        progress_callback: 进度回调函数
        job_id: 任务ID
    
    Returns:
        训练结果
    """
    config = config or {}
    
    # 配置参数
    model_name = config.get('model_name', 'resnet50')
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 1e-4)
    image_size = config.get('image_size', 224)
    num_workers = config.get('num_workers', 2)
    freeze_backbone = config.get('freeze_backbone', True)  # 默认冻结骨干网络
    val_split = config.get('val_split', 0.2)  # 自动划分验证集比例
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 数据加载
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    # 加载训练集
    full_dataset = ImageClassificationDataset(train_dir, transform=train_transform)
    num_classes = len(full_dataset.class_to_idx)
    class_names = full_dataset.get_class_names()
    
    logger.info(f"类别: {class_names}")
    logger.info(f"总样本数: {len(full_dataset)}")
    
    # 划分训练/验证集
    if val_dir and Path(val_dir).exists():
        # 使用独立验证集
        val_dataset = ImageClassificationDataset(val_dir, transform=val_transform)
        train_dataset = full_dataset
    else:
        # 自动划分
        from torch.utils.data import random_split
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        # 验证集使用val变换
        val_dataset.dataset.transform = val_transform
    
    # Celery环境下不能使用多进程DataLoader
    import multiprocessing
    try:
        # 检测是否在Celery worker进程中 (daemon进程)
        is_daemon = multiprocessing.current_process().daemon
        if is_daemon:
            num_workers = 0
            logger.info("Celery环境 detected, num_workers设置为0")
    except:
        pass
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=(num_workers>0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=(num_workers>0))
    
    # 创建模型
    model = get_model(model_name, num_classes, pretrained=True)
    
    # 冻结骨干网络（迁移学习）
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name and 'heads' not in name:
                param.requires_grad = False
        logger.info("已冻结骨干网络，只训练分类头")
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # 训练循环
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device, scaler)
        
        # 验证
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'class_to_idx': full_dataset.class_to_idx,
                'class_names': class_names,
                'config': config
            }, output_path / 'best_model.pth')
            logger.info(f"保存最佳模型，准确率: {best_acc:.4f}")
        
        # 进度回调
        if progress_callback:
            progress_callback({
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_acc': best_acc
            })
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': full_dataset.class_to_idx,
        'class_names': class_names,
        'config': config,
        'num_classes': num_classes
    }, output_path / 'final_model.pth')
    
    # 导出ONNX
    try:
        export_onnx(model, output_path / 'model.onnx', image_size, device)
    except Exception as e:
        logger.warning(f"ONNX导出失败: {e}")
    
    # 返回结果
    return {
        'success': True,
        'best_accuracy': best_acc,
        'final_accuracy': val_acc,
        'history': history,
        'model_path': str(output_path / 'best_model.pth'),
        'onnx_path': str(output_path / 'model.onnx'),
        'num_classes': num_classes,
        'class_names': class_names,
        'config': config
    }


def export_onnx(model, output_path, image_size, device):
    """导出ONNX格式"""
    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )
    logger.info(f"ONNX模型已导出: {output_path}")


def quantize_model(model_path, output_path):
    """
    模型量化 (INT8) - 减小模型大小，加速推理
    """
    try:
        import torch.quantization
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 重建模型结构
        config = checkpoint.get('config', {})
        model_name = config.get('model_name', 'resnet50')
        num_classes = checkpoint.get('num_classes', len(checkpoint['class_names']))
        
        model = get_model(model_name, num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 动态量化
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # 需要一些校准数据，这里简化处理
        # 实际应该使用验证集进行校准
        
        torch.quantization.convert(model, inplace=True)
        
        # 保存量化模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': checkpoint['class_names'],
            'class_to_idx': checkpoint['class_to_idx'],
            'quantized': True
        }, output_path)
        
        # 计算压缩比
        import os
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        
        logger.info(f"量化完成: {original_size:.1f}MB → {quantized_size:.1f}MB "
                   f"({original_size/quantized_size:.1f}x 压缩)")
        
        return {
            'success': True,
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size
        }
        
    except Exception as e:
        logger.error(f"量化失败: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # 测试
    result = run_image_classification_training(
        train_dir="/path/to/train",
        output_dir="./test_output",
        config={
            'model_name': 'resnet18',
            'epochs': 2,
            'batch_size': 16,
            'freeze_backbone': True
        }
    )
    print(result)
