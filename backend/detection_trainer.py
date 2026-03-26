# -*- coding: utf-8 -*-
"""
目标检测训练模块 - 基于YOLOv8
支持：缺陷定位、物体检测、实例分割
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入ultralytics，如果未安装则给出提示
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics未安装，目标检测功能不可用。运行: pip install ultralytics")

import torch
import numpy as np
from PIL import Image
import cv2


class ObjectDetectionTrainer:
    """目标检测训练器"""
    
    SUPPORTED_MODELS = {
        'yolov8n': {'name': 'YOLOv8 Nano', 'size': '6MB', 'speed': '最快', 'ap': '37.3'},
        'yolov8s': {'name': 'YOLOv8 Small', 'size': '22MB', 'speed': '快', 'ap': '44.9'},
        'yolov8m': {'name': 'YOLOv8 Medium', 'size': '54MB', 'speed': '中等', 'ap': '50.2'},
        'yolov8l': {'name': 'YOLOv8 Large', 'size': '89MB', 'speed': '慢', 'ap': '52.9'},
    }
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 训练配置
                - model: 模型类型 (yolov8n/s/m/l)
                - epochs: 训练轮数
                - batch_size: 批次大小
                - image_size: 输入尺寸
                - confidence_threshold: 置信度阈值
                - iou_threshold: NMS IoU阈值
        """
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("请先安装ultralytics: pip install ultralytics")
        
        self.config = config
        self.model_name = config.get('model', 'yolov8s')
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 16)
        self.image_size = config.get('image_size', 640)
        self.conf_threshold = config.get('confidence_threshold', 0.25)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {self.device}")
    
    def prepare_dataset(self, data_dir: str, output_dir: str) -> str:
        """
        准备数据集为YOLO格式
        
        输入结构:
            data_dir/
            ├── images/
            │   ├── train/
            │   └── val/
            └── labels/
                ├── train/
                └── val/
        
        YOLO格式:
            class_id center_x center_y width height (归一化0-1)
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检测数据集结构
        images_dir = data_dir / 'images'
        labels_dir = data_dir / 'labels'
        
        if not images_dir.exists():
            # 尝试自动转换图片分类格式到检测格式
            logger.info("未找到images目录，尝试从分类格式转换...")
            self._convert_classification_to_detection(data_dir, output_dir)
            images_dir = output_dir / 'images'
            labels_dir = output_dir / 'labels'
        
        # 获取类别列表
        classes = self._get_classes(labels_dir)
        
        # 创建data.yaml
        yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val
test:  # 可选

nc: {len(classes)}
names: {classes}
"""
        yaml_path = output_dir / 'data.yaml'
        yaml_path.write_text(yaml_content)
        
        logger.info(f"数据集准备完成: {yaml_path}")
        logger.info(f"类别: {classes}")
        
        return str(yaml_path)
    
    def _convert_classification_to_detection(self, data_dir: Path, output_dir: Path):
        """将图片分类格式转换为目标检测格式（每张图一个框覆盖全图）"""
        logger.info("将分类数据转换为检测格式...")
        
        # 创建目录
        (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        # 遍历类别文件夹
        class_id = 0
        for class_dir in sorted(data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            # 划分训练/验证集 (80/20)
            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # 处理训练集
            for img_path in train_images:
                self._process_image(img_path, output_dir, 'train', class_id, class_name)
            
            # 处理验证集
            for img_path in val_images:
                self._process_image(img_path, output_dir, 'val', class_id, class_name)
            
            class_id += 1
            logger.info(f"  {class_name}: {len(train_images)}训练 / {len(val_images)}验证")
    
    def _process_image(self, img_path: Path, output_dir: Path, split: str, class_id: int, class_name: str):
        """处理单张图片"""
        # 复制图片
        dst_img = output_dir / 'images' / split / f"{class_name}_{img_path.name}"
        import shutil
        shutil.copy(img_path, dst_img)
        
        # 创建标签（全图框）
        # YOLO格式: class_id center_x center_y width height
        label_path = output_dir / 'labels' / split / f"{class_name}_{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            # 默认框覆盖图片中心 80% 区域
            f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
    
    def _get_classes(self, labels_dir: Path) -> List[str]:
        """从labels目录获取类别列表"""
        # 尝试从data.yaml读取，或从目录结构推断
        classes = []
        train_labels = labels_dir / 'train'
        if train_labels.exists():
            for label_file in train_labels.glob('*.txt'):
                with open(label_file) as f:
                    for line in f:
                        class_id = int(line.strip().split()[0])
                        while len(classes) <= class_id:
                            classes.append(f"class_{len(classes)}")
        
        if not classes:
            classes = ['object']
        
        return classes
    
    def train(self, data_yaml: str, output_dir: str, progress_callback: Callable = None) -> Dict:
        """
        训练模型
        
        Args:
            data_yaml: 数据集配置文件路径
            output_dir: 输出目录
            progress_callback: 进度回调函数
        
        Returns:
            训练结果
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载预训练模型
        logger.info(f"加载模型: {self.model_name}")
        model = YOLO(f'{self.model_name}.pt')
        
        # 训练参数
        train_args = {
            'data': data_yaml,
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.image_size,
            'device': self.device,
            'project': str(output_path),
            'name': 'train',
            'exist_ok': True,
            'verbose': True,
        }
        
        # 添加回调
        if progress_callback:
            def on_train_epoch_end(trainer):
                epoch = trainer.epoch
                metrics = trainer.metrics
                box_map = metrics.get('metrics/mAP50-95(B)', 0)
                
                progress_callback({
                    'epoch': epoch + 1,
                    'total_epochs': self.epochs,
                    'box_map': box_map,
                    'progress': int((epoch + 1) / self.epochs * 100)
                })
            
            model.add_callback('on_train_epoch_end', on_train_epoch_end)
        
        # 开始训练
        logger.info("开始训练...")
        results = model.train(**train_args)
        
        # 保存结果
        best_model_path = output_path / 'train' / 'weights' / 'best.pt'
        final_model_path = output_path / 'best_model.pt'
        
        if best_model_path.exists():
            import shutil
            shutil.copy(best_model_path, final_model_path)
        
        # 导出ONNX
        onnx_path = output_path / 'model.onnx'
        try:
            model.export(format='onnx', imgsz=self.image_size)
            exported_onnx = output_path / 'train' / 'weights' / 'best.onnx'
            if exported_onnx.exists():
                shutil.copy(exported_onnx, onnx_path)
        except Exception as e:
            logger.warning(f"ONNX导出失败: {e}")
        
        # 返回结果
        metrics = results.results_dict if hasattr(results, 'results_dict') else {}
        
        return {
            'success': True,
            'model_path': str(final_model_path),
            'onnx_path': str(onnx_path) if onnx_path.exists() else None,
            'metrics': {
                'box_map': metrics.get('metrics/mAP50-95(B)', 0),
                'box_map50': metrics.get('metrics/mAP50(B)', 0),
                'precision': metrics.get('metrics/precision(B)', 0),
                'recall': metrics.get('metrics/recall(B)', 0),
            },
            'config': self.config,
            'epochs_trained': self.epochs,
            'device': self.device
        }
    
    def predict(self, model_path: str, image_path: str, save: bool = True) -> List[Dict]:
        """
        使用训练好的模型进行预测
        
        Args:
            model_path: 模型路径
            image_path: 图片路径
            save: 是否保存结果图
        
        Returns:
            检测结果列表
        """
        model = YOLO(model_path)
        
        results = model.predict(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save,
            device=self.device
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                detections.append({
                    'class_id': int(box.cls),
                    'class_name': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    'bbox_normalized': box.xyxyn[0].tolist()  # 归一化坐标
                })
        
        return detections


def run_object_detection_training(
    train_dir: str,
    output_dir: str,
    config: Dict,
    progress_callback: Callable = None,
    job_id: str = None
) -> Dict:
    """
    运行目标检测训练（Celery任务调用）
    """
    logger.info(f"[ObjectDetection] 开始训练任务: {job_id}")
    
    try:
        trainer = ObjectDetectionTrainer(config)
        
        # 准备数据集
        data_yaml = trainer.prepare_dataset(train_dir, output_dir)
        
        # 训练
        result = trainer.train(data_yaml, output_dir, progress_callback)
        
        logger.info(f"[ObjectDetection] 训练完成: {job_id}")
        return result
        
    except Exception as e:
        logger.error(f"[ObjectDetection] 训练失败: {e}")
        raise


if __name__ == "__main__":
    # 测试
    config = {
        'model': 'yolov8n',
        'epochs': 10,
        'batch_size': 16,
        'image_size': 640
    }
    
    result = run_object_detection_training(
        train_dir='/path/to/data',
        output_dir='./test_output',
        config=config
    )
    print(result)
