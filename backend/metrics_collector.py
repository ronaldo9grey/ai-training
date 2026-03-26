# -*- coding: utf-8 -*-
"""
Prometheus 监控指标收集器
收集系统性能、模型预测质量、训练任务等指标
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
import time
import psutil
from typing import Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """监控指标收集器"""
    
    def __init__(self):
        # 1. 预测相关指标
        self.prediction_counter = Counter(
            'ai_prediction_total',
            '总预测次数',
            ['model_type', 'status']  # status: success, error
        )
        
        self.prediction_latency = Histogram(
            'ai_prediction_latency_seconds',
            '预测延迟（秒）',
            ['model_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.prediction_confidence = Histogram(
            'ai_prediction_confidence',
            '预测置信度分布',
            ['model_type'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # 2. 训练任务指标
        self.training_counter = Counter(
            'ai_training_total',
            '训练任务总数',
            ['task_type', 'status']  # status: completed, failed
        )
        
        self.training_duration = Histogram(
            'ai_training_duration_seconds',
            '训练耗时（秒）',
            ['task_type'],
            buckets=[60, 300, 600, 1800, 3600, 7200, 18000]
        )
        
        self.model_accuracy = Gauge(
            'ai_model_accuracy',
            '模型准确率',
            ['model_id', 'project_id']
        )
        
        # 3. 系统资源指标
        self.system_memory_percent = Gauge(
            'ai_system_memory_percent',
            '系统内存使用率'
        )
        
        self.system_cpu_percent = Gauge(
            'ai_system_cpu_percent',
            '系统CPU使用率'
        )
        
        self.loaded_models_count = Gauge(
            'ai_loaded_models_count',
            '当前加载的模型数量'
        )
        
        # 4. 业务指标
        self.active_projects = Gauge(
            'ai_active_projects',
            '活跃项目数'
        )
        
        self.dataset_size = Gauge(
            'ai_dataset_size_bytes',
            '数据集大小（字节）',
            ['dataset_id', 'project_id']
        )
        
        # 5. 版本信息
        self.version_info = Info(
            'ai_platform',
            '平台版本信息'
        )
        self.version_info.info({'version': '2.0.0', 'build': '20240321'})
        
    def record_prediction(self, model_type: str, latency: float, 
                         confidence: float = None, success: bool = True):
        """记录预测指标"""
        status = 'success' if success else 'error'
        self.prediction_counter.labels(model_type=model_type, status=status).inc()
        self.prediction_latency.labels(model_type=model_type).observe(latency)
        
        if confidence is not None and success:
            self.prediction_confidence.labels(model_type=model_type).observe(confidence)
    
    def record_training(self, task_type: str, duration: float, 
                       accuracy: float = None, success: bool = True):
        """记录训练指标"""
        status = 'completed' if success else 'failed'
        self.training_counter.labels(task_type=task_type, status=status).inc()
        self.training_duration.labels(task_type=task_type).observe(duration)
    
    def update_model_accuracy(self, model_id: str, project_id: str, accuracy: float):
        """更新模型准确率"""
        self.model_accuracy.labels(model_id=model_id, project_id=project_id).set(accuracy)
    
    def update_system_metrics(self, loaded_models: int = 0):
        """更新系统资源指标"""
        self.system_memory_percent.set(psutil.virtual_memory().percent)
        self.system_cpu_percent.set(psutil.cpu_percent(interval=1))
        self.loaded_models_count.set(loaded_models)
    
    def update_active_projects(self, count: int):
        """更新活跃项目数"""
        self.active_projects.set(count)


# 全局收集器实例
metrics = MetricsCollector()


def monitor_prediction(model_type: str):
    """预测监控装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                
                # 提取置信度（如果有）
                confidence = None
                if isinstance(result, dict):
                    confidence = result.get('confidence')
                
                metrics.record_prediction(
                    model_type=model_type,
                    latency=latency,
                    confidence=confidence,
                    success=True
                )
                return result
            except Exception as e:
                latency = time.time() - start_time
                metrics.record_prediction(
                    model_type=model_type,
                    latency=latency,
                    success=False
                )
                raise
        return wrapper
    return decorator


def get_metrics_response():
    """获取Prometheus格式的指标数据"""
    # 更新系统指标
    metrics.update_system_metrics()
    
    return generate_latest()


# 便捷函数
def record_inference_metrics(model_type: str, latency_ms: float, 
                             confidence: float = None, error: bool = False):
    """记录推理指标"""
    metrics.record_prediction(
        model_type=model_type,
        latency=latency_ms / 1000,  # 转换为秒
        confidence=confidence,
        success=not error
    )


def record_training_metrics(task_type: str, duration_sec: float,
                           accuracy: float = None, error: bool = False):
    """记录训练指标"""
    metrics.record_training(
        task_type=task_type,
        duration=duration_sec,
        accuracy=accuracy,
        success=not error
    )
