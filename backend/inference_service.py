# -*- coding: utf-8 -*-
"""
统一推理服务 - 支持NLP模型(transformers)和ML模型(sklearn)
带智能内存管理：LRU淘汰 + 内存阈值 + 空闲超时
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import json
import logging
import joblib
import numpy as np
import pandas as pd
import psutil
import time
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 内存管理配置
MAX_MEMORY_PERCENT = 80  # 内存使用超过80%触发清理
MAX_MODELS_IN_MEMORY = 5  # 最多同时驻留5个模型
MODEL_IDLE_TIMEOUT = 1800  # 30分钟空闲自动卸载（秒）


class ModelInfo:
    """模型信息包装类，记录使用统计"""
    def __init__(self, model_data: Dict, model_type: str, model_path: str):
        self.data = model_data
        self.model_type = model_type
        self.model_path = model_path
        self.last_access_time = time.time()
        self.access_count = 0
        self.created_at = time.time()
    
    def touch(self):
        """更新访问时间"""
        self.last_access_time = time.time()
        self.access_count += 1
    
    def get_memory_size(self) -> int:
        """估算模型内存占用（字节）"""
        if self.model_type == 'nlp':
            # NLP模型：通过参数数量估算
            model = self.data.get('model')
            if model:
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                return param_size
        return 100 * 1024 * 1024  # 默认100MB


class UnifiedInferenceService:
    """统一推理服务 - 带智能内存管理"""
    
    def __init__(self):
        self.models = OrderedDict()  # OrderedDict用于LRU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()
        
        # 启动内存监控线程
        self.monitor_thread = threading.Thread(target=self._memory_monitor, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"推理服务初始化，设备: {self.device}")
        logger.info(f"内存管理: 阈值{MAX_MEMORY_PERCENT}%, 最多{MAX_MODELS_IN_MEMORY}个模型, 空闲{MODEL_IDLE_TIMEOUT}秒卸载")
    
    def _get_memory_percent(self) -> float:
        """获取当前内存使用百分比"""
        return psutil.virtual_memory().percent
    
    def _memory_monitor(self):
        """内存监控线程 - 后台定期检查和清理"""
        while True:
            try:
                time.sleep(60)  # 每分钟检查一次
                
                with self.lock:
                    current_time = time.time()
                    memory_percent = self._get_memory_percent()
                    
                    # 1. 检查内存阈值
                    if memory_percent > MAX_MEMORY_PERCENT:
                        logger.warning(f"内存使用过高 ({memory_percent:.1f}%)，触发清理")
                        self._cleanup_lru(keep_newest=2)  # 保留最新的2个
                        continue
                    
                    # 2. 检查空闲超时
                    to_remove = []
                    for model_id, model_info in self.models.items():
                        idle_time = current_time - model_info.last_access_time
                        if idle_time > MODEL_IDLE_TIMEOUT:
                            to_remove.append(model_id)
                    
                    for model_id in to_remove:
                        logger.info(f"模型 {model_id} 空闲超过{MODEL_IDLE_TIMEOUT}秒，自动卸载")
                        self._unload_model_unlocked(model_id)
                    
                    # 3. 检查模型数量限制
                    if len(self.models) > MAX_MODELS_IN_MEMORY:
                        excess = len(self.models) - MAX_MODELS_IN_MEMORY
                        logger.info(f"模型数量超限，清理{excess}个最少使用的模型")
                        self._cleanup_lru(keep_newest=MAX_MODELS_IN_MEMORY)
                        
            except Exception as e:
                logger.error(f"内存监控线程异常: {e}")
    
    def _cleanup_lru(self, keep_newest: int = 1):
        """LRU清理：保留最新使用的N个模型"""
        with self.lock:
            # 按最后访问时间排序
            sorted_models = sorted(
                self.models.items(),
                key=lambda x: x[1].last_access_time,
                reverse=True
            )
            
            # 卸载旧的模型
            for model_id, _ in sorted_models[keep_newest:]:
                self._unload_model_unlocked(model_id)
    
    def _unload_model_unlocked(self, model_id: str):
        """卸载模型（无锁版本，调用前需获取锁）"""
        if model_id not in self.models:
            return
        
        model_info = self.models[model_id]
        
        # 删除模型释放内存
        del self.models[model_id]
        
        # NLP模型需要清理CUDA缓存
        if model_info.model_type == 'nlp' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"模型 {model_id} 已卸载，当前内存: {self._get_memory_percent():.1f}%")
    
    def get_memory_status(self) -> Dict:
        """获取内存状态"""
        memory = psutil.virtual_memory()
        
        models_info = []
        with self.lock:
            for model_id, model_info in self.models.items():
                models_info.append({
                    "model_id": model_id,
                    "model_type": model_info.model_type,
                    "access_count": model_info.access_count,
                    "idle_seconds": int(time.time() - model_info.last_access_time),
                    "memory_size_mb": model_info.get_memory_size() // (1024 * 1024)
                })
        
        return {
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024 ** 3),
            "memory_total_gb": memory.total / (1024 ** 3),
            "memory_available_gb": memory.available / (1024 ** 3),
            "loaded_models_count": len(self.models),
            "max_models_limit": MAX_MODELS_IN_MEMORY,
            "models": models_info
        }
    
    def load_model(self, model_path: str, model_id: str, model_type: str = 'nlp'):
        """
        加载模型到内存（带内存检查）
        """
        with self.lock:
            # 检查是否已加载
            if model_id in self.models:
                self.models[model_id].touch()
                logger.info(f"模型 {model_id} 已在内存中（命中缓存）")
                return self.models[model_id].data
            
            # 检查内存是否充足
            memory_percent = self._get_memory_percent()
            if memory_percent > MAX_MEMORY_PERCENT:
                logger.warning(f"内存使用率高 ({memory_percent:.1f}%)，尝试清理旧模型")
                self._cleanup_lru(keep_newest=max(1, MAX_MODELS_IN_MEMORY // 2))
            
            # 加载模型
            if model_type == 'nlp':
                model_data = self._load_nlp_model(model_path)
            elif model_type == 'ml':
                model_data = self._load_ml_model(model_path)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 包装为ModelInfo
            model_info = ModelInfo(model_data, model_type, model_path)
            self.models[model_id] = model_info
            
            logger.info(f"模型 {model_id} ({model_type}) 已加载，当前内存: {self._get_memory_percent():.1f}%")
            return model_data
    
    def _load_nlp_model(self, model_path: str) -> Dict:
        """加载NLP模型 (transformers)"""
        final_path = Path(model_path) / "final"
        label_path = Path(model_path) / "label_mapping.json"
        
        if not final_path.exists():
            raise ValueError(f"模型文件不存在: {final_path}")
        
        # 加载标签映射
        id2label = {}
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
                id2label = {int(k): v for k, v in label_map['id2label'].items()}
        
        # 加载模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(final_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(final_path))
        model.to(self.device)
        model.eval()
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'id2label': id2label,
            'num_labels': len(id2label)
        }
    
    def _load_ml_model(self, model_path: str) -> Dict:
        """加载ML模型 (sklearn)"""
        model_file = Path(model_path) / "model.joblib"
        config_file = Path(model_path) / "config.json"
        
        if not model_file.exists():
            raise ValueError(f"模型文件不存在: {model_file}")
        
        # 加载模型包
        model_package = joblib.load(model_file)
        
        # 加载配置
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        return {
            'model': model_package.get('model'),
            'scaler': model_package.get('scaler'),
            'label_encoder': model_package.get('label_encoder'),
            'feature_names': model_package.get('feature_names'),
            'config': config,
            'task_type': config.get('task_type', 'classification')
        }
    
    def predict(self, inputs: Union[List[str], pd.DataFrame, np.ndarray], model_id: str) -> List[Dict]:
        """
        统一预测接口（带访问统计和监控）
        """
        import time
        from metrics_collector import record_inference_metrics
        
        start_time = time.time()
        
        with self.lock:
            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 未加载，请先调用load_model")
            
            # 更新访问时间
            self.models[model_id].touch()
            
            model_info = self.models[model_id]
            model_data = model_info.data
            model_type = model_info.model_type
        
        try:
            # 预测时不需要持有锁
            if model_type == 'nlp':
                results = self._predict_nlp(inputs, model_data)
            elif model_type == 'ml':
                results = self._predict_ml(inputs, model_data)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 记录监控指标
            latency_ms = (time.time() - start_time) * 1000
            confidence = None
            if results and len(results) > 0:
                confidence = results[0].get('confidence')
            
            record_inference_metrics(
                model_type=model_type,
                latency_ms=latency_ms,
                confidence=confidence,
                error=False
            )
            
            return results
            
        except Exception as e:
            # 记录错误
            latency_ms = (time.time() - start_time) * 1000
            record_inference_metrics(
                model_type=model_type,
                latency_ms=latency_ms,
                error=True
            )
            raise
    
    def _predict_nlp(self, texts: List[str], model_info: Dict) -> List[Dict]:
        """NLP模型预测"""
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        id2label = model_info['id2label']
        
        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 构建结果
        results = []
        for i, text in enumerate(texts):
            pred_id = torch.argmax(probs[i]).item()
            confidence = probs[i][pred_id].item()
            
            all_probs = {id2label[j]: round(probs[i][j].item(), 4) for j in range(len(id2label))}
            
            results.append({
                'input': text[:100],  # 截断显示
                'prediction': id2label[pred_id],
                'confidence': round(confidence, 4),
                'all_probabilities': all_probs
            })
        
        return results
    
    def _predict_ml(self, X: Union[pd.DataFrame, np.ndarray], model_info: Dict) -> List[Dict]:
        """ML模型预测"""
        model = model_info['model']
        scaler = model_info['scaler']
        label_encoder = model_info['label_encoder']
        config = model_info['config']
        task_type = config.get('task_type', 'classification')
        
        # 确保是DataFrame
        if isinstance(X, np.ndarray):
            feature_names = model_info.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
            X = pd.DataFrame(X, columns=feature_names)
        
        # 选择特征列
        feature_names = model_info.get('feature_names')
        if feature_names:
            X = X[feature_names]
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 编码分类特征
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # 标准化
        if scaler:
            X_scaled = scaler.transform(X.values)
        else:
            X_scaled = X.values
        
        # 预测
        if task_type == 'classification':
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
            
            results = []
            for i, pred in enumerate(predictions):
                label = label_encoder.inverse_transform([pred])[0] if label_encoder else str(pred)
                result = {
                    'prediction': label,
                    'confidence': round(float(probabilities[i].max()), 4) if probabilities is not None else None
                }
                if probabilities is not None:
                    result['probabilities'] = {
                        (label_encoder.inverse_transform([j])[0] if label_encoder else str(j)): round(float(p), 4)
                        for j, p in enumerate(probabilities[i])
                    }
                results.append(result)
            return results
        
        elif task_type == 'regression':
            predictions = model.predict(X_scaled)
            return [{'prediction': round(float(p), 4)} for p in predictions]
        
        elif task_type == 'anomaly_detection':
            predictions = model.predict(X_scaled)
            scores = model.decision_function(X_scaled) if hasattr(model, 'decision_function') else None
            return [{
                'is_anomaly': bool(pred == 1 or pred == -1),  # isolation forest returns -1/1
                'anomaly_score': round(float(scores[i]), 4) if scores is not None else None
            } for i, pred in enumerate(predictions)]
        
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
    
    def predict_file(self, file_path: str, model_id: str, output_path: str = None) -> Dict:
        """
        对文件进行批量预测
        
        Args:
            file_path: 输入文件路径
            model_id: 模型ID
            output_path: 输出文件路径（可选）
            
        Returns:
            预测结果统计
        """
        # 读取文件
        ext = Path(file_path).suffix.lower()
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        model_info = self.models[model_id]
        
        # 根据模型类型选择输入
        if model_info['model_type'] == 'nlp':
            # NLP模型：假设第一列是文本
            text_col = df.columns[0]
            texts = df[text_col].astype(str).tolist()
            results = self._predict_nlp(texts, model_info)
            
            # 添加预测结果到DataFrame
            df['prediction'] = [r['prediction'] for r in results]
            df['confidence'] = [r['confidence'] for r in results]
        else:
            # ML模型：使用特征列
            results = self._predict_ml(df, model_info)
            
            # 添加预测结果到DataFrame
            if model_info['task_type'] == 'classification':
                df['prediction'] = [r['prediction'] for r in results]
                df['confidence'] = [r['confidence'] for r in results]
            elif model_info['task_type'] == 'regression':
                df['prediction'] = [r['prediction'] for r in results]
            elif model_info['task_type'] == 'anomaly_detection':
                df['is_anomaly'] = [r['is_anomaly'] for r in results]
                df['anomaly_score'] = [r['anomaly_score'] for r in results]
        
        # 保存结果
        if output_path:
            df.to_csv(output_path, index=False)
        
        return {
            'total_samples': len(df),
            'predictions': results[:10],  # 只返回前10条作为预览
            'output_file': output_path
        }
    
    def unload_model(self, model_id: str):
        """手动卸载模型释放内存"""
        with self.lock:
            self._unload_model_unlocked(model_id)

    def list_loaded_models(self) -> List[Dict]:
        """列出已加载的模型"""
        with self.lock:
            return [
                {
                    'model_id': mid,
                    'model_type': info.model_type,
                    'access_count': info.access_count,
                    'idle_seconds': int(time.time() - info.last_access_time)
                }
                for mid, info in self.models.items()
            ]


# 导入LabelEncoder用于ML模型
from sklearn.preprocessing import LabelEncoder

# 全局推理服务实例
inference_service = UnifiedInferenceService()
