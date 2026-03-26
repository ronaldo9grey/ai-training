# -*- coding: utf-8 -*-
"""
模型可解释性模块 (XAI)
支持：Grad-CAM (图像)、SHAP (表格)、LIME (文本)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAMExplainer:
    """Grad-CAM 图像可解释性分析器"""
    
    def __init__(self, model, target_layer: str = None):
        """
        Args:
            model: PyTorch模型
            target_layer: 目标层名称 (如 'layer4', 'features')
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 自动查找合适的层
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        
        # 注册钩子
        self._register_hooks()
    
    def _find_target_layer(self) -> str:
        """自动查找最后一个卷积层"""
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Sequential)):
                return name
        return 'layer4'  # 默认ResNet最后一层
    
    def _register_hooks(self):
        """注册前向/反向传播钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 查找目标层
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                logger.info(f"Grad-CAM注册层: {name}")
                break
    
    def generate_heatmap(self, image: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        生成Grad-CAM热力图
        
        Args:
            image: 输入图片张量 (1, C, H, W)
            target_class: 目标类别，None则使用预测类别
        
        Returns:
            热力图数组 (H, W)，值域0-1
        """
        image = image.requires_grad_(True)
        
        # 前向传播
        output = self.model(image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # 计算Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # 全局平均池化获取权重
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # 加权求和
        cam = torch.zeros_like(activations[0])
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # 归一化到0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # 调整尺寸到原图大小
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
        
        return cam
    
    def visualize(self, image: np.ndarray, heatmap: np.ndarray, 
                  alpha: float = 0.5) -> np.ndarray:
        """
        将热力图叠加到原图
        
        Args:
            image: 原图 (H, W, 3)，值域0-255或0-1
            heatmap: 热力图 (H, W)，值域0-1
            alpha: 透明度
        
        Returns:
            叠加后的图像
        """
        # 确保图片格式正确
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        # 热力图转彩色
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 叠加
        result = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        
        return result
    
    def explain_to_base64(self, image: torch.Tensor, 
                          original_image: np.ndarray = None) -> Dict:
        """
        生成解释结果并转为base64
        
        Returns:
            {
                'heatmap': 'data:image/png;base64,xxx',
                'overlay': 'data:image/png;base64,xxx',
                'target_class': int,
                'confidence': float
            }
        """
        with torch.enable_grad():
            heatmap = self.generate_heatmap(image)
        
        # 获取预测
        with torch.no_grad():
            output = self.model(image)
            probs = F.softmax(output, dim=1)
            target_class = output.argmax(dim=1).item()
            confidence = probs[0, target_class].item()
        
        # 可视化
        if original_image is None:
            # 从tensor还原
            original_image = image[0].permute(1, 2, 0).cpu().numpy()
            original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
            original_image = (original_image * 255).astype(np.uint8)
        
        overlay = self.visualize(original_image, heatmap)
        
        # 转为base64
        def array_to_base64(arr: np.ndarray) -> str:
            pil_img = Image.fromarray(arr)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        
        # 热力图单独保存
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        return {
            'heatmap': array_to_base64(heatmap_colored),
            'overlay': array_to_base64(overlay),
            'target_class': target_class,
            'confidence': confidence,
            'heatmap_data': heatmap.tolist()  # 原始数据供前端使用
        }


class SHAPExplainer:
    """SHAP 表格数据可解释性分析器"""
    
    def __init__(self, model, feature_names: List[str] = None):
        """
        Args:
            model: sklearn模型或预测函数
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        
        try:
            import shap
            self.shap = shap
            self.explainer = None
        except ImportError:
            logger.warning("SHAP未安装，使用简化版解释")
            self.shap = None
    
    def explain(self, X: np.ndarray, background: np.ndarray = None) -> Dict:
        """
        计算SHAP值
        
        Args:
            X: 待解释样本 (n_samples, n_features)
            background: 背景数据用于对比
        
        Returns:
            {
                'shap_values': [...],
                'feature_importance': [...],
                'base_value': float,
                'prediction': float
            }
        """
        if self.shap is None:
            return self._explain_simple(X)
        
        # 使用KernelExplainer (模型无关)
        if self.explainer is None:
            if background is None:
                background = X[:10] if len(X) >= 10 else X
            self.explainer = self.shap.KernelExplainer(self.model.predict, background)
        
        shap_values = self.explainer.shap_values(X)
        
        # 特征重要性 (绝对值平均)
        importance = np.abs(shap_values).mean(axis=0)
        
        # 构建结果
        feature_names = self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        return {
            'shap_values': shap_values.tolist(),
            'feature_importance': [
                {'feature': name, 'importance': float(imp)}
                for name, imp in zip(feature_names, importance)
            ],
            'base_value': float(self.explainer.expected_value),
            'predictions': self.model.predict(X).tolist()
        }
    
    def _explain_simple(self, X: np.ndarray) -> Dict:
        """简化版解释 (特征重要性)"""
        # 使用置换重要性近似
        baseline_pred = self.model.predict(X)
        importances = []
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_pred = self.model.predict(X_permuted)
            importance = np.mean(np.abs(baseline_pred - permuted_pred))
            importances.append(importance)
        
        feature_names = self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        return {
            'feature_importance': [
                {'feature': name, 'importance': float(imp)}
                for name, imp in zip(feature_names, importances)
            ],
            'method': 'permutation_importance',
            'predictions': baseline_pred.tolist()
        }


class TextExplainer:
    """文本可解释性分析器 (基于attention或LIME)"""
    
    def explain_attention(self, tokens: List[str], 
                          attention_weights: np.ndarray) -> List[Dict]:
        """
        基于Attention权重解释
        
        Args:
            tokens: 分词结果
            attention_weights: Attention权重 (seq_len,)
        
        Returns:
            [
                {'token': 'xxx', 'weight': 0.8, 'index': 0},
                ...
            ]
        """
        # 归一化
        weights = attention_weights / (attention_weights.sum() + 1e-8)
        
        return [
            {
                'token': token,
                'weight': float(weight),
                'index': i,
                'highlight': weight > 0.1  # 高亮阈值
            }
            for i, (token, weight) in enumerate(zip(tokens, weights))
        ]
    
    def highlight_text(self, text: str, words_importance: List[Tuple[str, float]]) -> str:
        """
        生成高亮文本 (HTML格式)
        
        Args:
            text: 原始文本
            words_importance: [(word, importance), ...]
        
        Returns:
            HTML字符串
        """
        html_parts = []
        for word, importance in words_importance:
            # 根据重要性设置颜色强度
            if importance > 0.3:
                color = f'rgba(255, 0, 0, {min(importance, 0.8)})'
                html_parts.append(f'<span style="background-color: {color}; padding: 2px;">{word}</span>')
            elif importance > 0.1:
                color = f'rgba(255, 165, 0, {min(importance, 0.6)})'
                html_parts.append(f'<span style="background-color: {color}; padding: 2px;">{word}</span>')
            else:
                html_parts.append(word)
        
        return ' '.join(html_parts)


def explain_image_classification(model_path: str, image_path: str, 
                                 target_layer: str = None) -> Dict:
    """
    解释图像分类模型的预测
    
    Args:
        model_path: 模型路径 (.pth)
        image_path: 图片路径
        target_layer: 目标层名称
    
    Returns:
        包含热力图和解释信息的结果
    """
    import torch
    from torchvision import transforms
    from PIL import Image
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 这里需要根据实际模型结构调整
    # 简化版：假设是标准的torchvision模型
    model = checkpoint.get('model', None)
    if model is None:
        raise ValueError("无法从checkpoint加载模型")
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    image_array = np.array(image.resize((224, 224)))
    
    # 创建解释器
    explainer = GradCAMExplainer(model, target_layer)
    
    # 生成解释
    result = explainer.explain_to_base64(image_tensor, image_array)
    
    return result


def explain_tabular_prediction(model, X: np.ndarray, 
                               feature_names: List[str] = None,
                               background: np.ndarray = None) -> Dict:
    """
    解释表格数据预测
    
    Args:
        model: sklearn模型
        X: 待解释样本
        feature_names: 特征名称
        background: 背景数据
    
    Returns:
        SHAP解释结果
    """
    explainer = SHAPExplainer(model, feature_names)
    return explainer.explain(X, background)


if __name__ == "__main__":
    # 测试
    print("XAI模块加载成功")
