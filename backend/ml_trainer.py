# -*- coding: utf-8 -*-
"""
通用ML训练器 - 支持分类/回归/异常检测
处理结构化数据（表格）和时序数据
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# 传统ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_auc_score
)

# 深度学习
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 时序
from sklearn.neighbors import LocalOutlierFactor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")


@dataclass
class MLTrainingConfig:
    """通用ML训练配置"""
    # 任务类型
    task_type: str = "classification"  # classification, regression, anomaly_detection
    
    # 模型选择
    model_type: str = "random_forest"  # random_forest, logistic, linear, lstm, transformer
    
    # 数据配置
    target_column: str = "target"
    feature_columns: List[str] = None
    time_column: Optional[str] = None  # 时序数据的时间列
    
    # 训练参数
    test_size: float = 0.2
    random_state: int = 42
    
    # 模型参数
    n_estimators: int = 100  # RF
    max_depth: Optional[int] = None  # RF
    learning_rate: float = 0.001  # DL
    epochs: int = 50  # DL
    batch_size: int = 32  # DL
    sequence_length: int = 10  # 时序
    
    # 异常检测参数
    contamination: float = 0.1  # 异常比例估计
    
    # 输出
    output_dir: str = "./outputs"


class TimeSeriesDataset(Dataset):
    """PyTorch时序数据集"""
    def __init__(self, data: np.ndarray, sequence_length: int):
        self.data = data
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx:idx+self.sequence_length])


class LSTMModel(nn.Module):
    """LSTM预测模型"""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class AnomalyDetector:
    """异常检测器 - 支持多种算法"""
    
    def __init__(self, method: str = "isolation_forest", contamination: float = 0.1):
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == "isolation_forest":
            self.model = IsolationForest(contamination=self.contamination, random_state=42)
        elif self.method == "lof":
            self.model = LocalOutlierFactor(contamination=self.contamination, novelty=True)
        
        self.model.fit(X_scaled)
        return self
    
    def predict(self, X: np.ndarray):
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # -1 是异常，1 是正常，转为 1/0
        return (predictions == -1).astype(int)
    
    def decision_function(self, X: np.ndarray):
        """返回异常分数（越大越正常）"""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X_scaled)
        elif hasattr(self.model, 'score_samples'):
            return self.model.score_samples(X_scaled)
        return None


class GeneralMLTrainer:
    """通用ML训练器"""
    
    SUPPORTED_MODELS = {
        "classification": ["random_forest", "logistic", "lstm"],
        "regression": ["random_forest", "linear", "lstm"],
        "anomaly_detection": ["isolation_forest", "lof"]
    }
    
    def __init__(self, config: MLTrainingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.feature_names = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据 - 支持文件或目录"""
        path = Path(file_path)
        
        # 如果是目录，查找目录下的CSV文件
        if path.is_dir():
            csv_files = list(path.glob('*.csv'))
            if not csv_files:
                raise ValueError(f"目录中没有CSV文件: {file_path}")
            # 使用第一个CSV文件
            path = csv_files[0]
            logger.info(f"从目录中选择数据文件: {path}")
        
        ext = path.suffix.lower()
        if ext == '.csv':
            return pd.read_csv(path)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """数据预处理"""
        # 选择特征列
        if self.config.feature_columns:
            feature_cols = self.config.feature_columns
        else:
            feature_cols = [c for c in df.columns if c != self.config.target_column]
            self.config.feature_columns = feature_cols
        
        self.feature_names = feature_cols
        
        # 提取特征
        X = df[feature_cols].copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 编码分类特征
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # 提取目标
        if self.config.task_type == "anomaly_detection":
            # 异常检测可能无标签
            y = None
        else:
            y = df[self.config.target_column].values
            
            # 分类任务编码标签
            if self.config.task_type == "classification":
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
        
        return X.values, y
    
    def prepare_sequence_data(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """准备时序数据"""
        seq_len = self.config.sequence_length
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            if y is not None:
                y_seq.append(y[i+seq_len])
        
        return np.array(X_seq), np.array(y_seq) if y_seq else None
    
    def train(self, train_path: str, val_path: Optional[str] = None, progress_callback=None) -> Dict:
        """训练入口"""
        logger.info(f"开始训练: {self.config.task_type} - {self.config.model_type}")
        
        # 加载数据
        df_train = self.load_data(train_path)
        
        # 预处理
        X, y = self.preprocess(df_train)
        
        # 划分训练/验证集
        if val_path:
            df_val = self.load_data(val_path)
            X_val, y_val = self.preprocess(df_val)
        else:
            if self.config.task_type == "anomaly_detection":
                X_val, y_val = X, None
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.config.test_size, random_state=self.config.random_state
                )
                X, y = X_train, y_train
        
        # 训练
        if self.config.task_type == "classification":
            result = self._train_classification(X, y, X_val, y_val, progress_callback)
        elif self.config.task_type == "regression":
            result = self._train_regression(X, y, X_val, y_val, progress_callback)
        elif self.config.task_type == "anomaly_detection":
            result = self._train_anomaly_detection(X, X_val, progress_callback)
        else:
            raise ValueError(f"不支持的任务类型: {self.config.task_type}")
        
        # 保存模型
        self._save_model()
        
        return result
    
    def _train_classification(self, X_train, y_train, X_val, y_val, progress_callback) -> Dict:
        """训练分类模型"""
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if self.config.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
        
        # 训练
        self.model.fit(X_train_scaled, y_train)
        
        # 评估
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        # 详细报告
        report = classification_report(y_val, val_pred, output_dict=True)
        cm = confusion_matrix(y_val, val_pred).tolist()
        
        # 特征重要性
        importance = None
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_.tolist()))
        
        return {
            "task": "classification",
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "feature_importance": importance
        }
    
    def _train_regression(self, X_train, y_train, X_val, y_val, progress_callback) -> Dict:
        """训练回归模型"""
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if self.config.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == "linear":
            self.model = LinearRegression()
        
        # 训练
        self.model.fit(X_train_scaled, y_train)
        
        # 评估
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        return {
            "task": "regression",
            "train_mse": mean_squared_error(y_train, train_pred),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "train_r2": r2_score(y_train, train_pred),
            "val_mse": mean_squared_error(y_val, val_pred),
            "val_mae": mean_absolute_error(y_val, val_pred),
            "val_r2": r2_score(y_val, val_pred),
            "predictions": {
                "actual": y_val[:100].tolist(),  # 只存前100个用于展示
                "predicted": val_pred[:100].tolist()
            }
        }
    
    def _train_anomaly_detection(self, X_train, X_val, progress_callback) -> Dict:
        """训练异常检测模型"""
        self.model = AnomalyDetector(
            method=self.config.model_type,
            contamination=self.config.contamination
        )
        
        # 训练
        self.model.fit(X_train)
        
        # 在训练集上检测
        train_pred = self.model.predict(X_train)
        train_scores = self.model.decision_function(X_train)
        
        # 在验证集上检测
        val_pred = self.model.predict(X_val)
        val_scores = self.model.decision_function(X_val)
        
        return {
            "task": "anomaly_detection",
            "train_anomaly_ratio": train_pred.mean(),
            "val_anomaly_ratio": val_pred.mean(),
            "threshold": np.percentile(train_scores, self.config.contamination * 100) if train_scores is not None else None,
            "sample_scores": val_scores[:100].tolist() if val_scores is not None else None
        }
    
    def _save_model(self):
        """保存模型"""
        import joblib
        
        model_path = self.output_dir / "model.joblib"
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "config": self.config
        }, model_path)
        
        # 保存配置
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "task_type": self.config.task_type,
                "model_type": self.config.model_type,
                "feature_columns": self.config.feature_columns,
                "target_column": self.config.target_column,
                "feature_names": self.feature_names
            }, f, indent=2)
    
    def predict(self, X: np.ndarray) -> Dict:
        """预测"""
        X_scaled = self.scaler.transform(X)
        
        if self.config.task_type == "classification":
            pred = self.model.predict(X_scaled)
            proba = self.model.predict_proba(X_scaled) if hasattr(self.model, 'predict_proba') else None
            
            results = []
            for i, p in enumerate(pred):
                label = self.label_encoder.inverse_transform([p])[0] if self.label_encoder else str(p)
                result = {"prediction": label}
                if proba is not None:
                    result["confidence"] = float(proba[i].max())
                    result["probabilities"] = {
                        self.label_encoder.inverse_transform([j])[0] if self.label_encoder else str(j): float(prob)
                        for j, prob in enumerate(proba[i])
                    }
                results.append(result)
            return results
        
        elif self.config.task_type == "regression":
            pred = self.model.predict(X_scaled)
            return [{"prediction": float(p)} for p in pred]
        
        elif self.config.task_type == "anomaly_detection":
            pred = self.model.predict(X_scaled)
            scores = self.model.decision_function(X_scaled)
            return [{
                "is_anomaly": bool(p),
                "anomaly_score": float(s) if s is not None else None
            } for p, s in zip(pred, scores)]
        
        return []


def run_ml_training(
    train_path: str,
    val_path: Optional[str],
    output_dir: str,
    config: Dict
) -> Dict:
    """
    通用ML训练入口
    
    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径（可选）
        output_dir: 输出目录
        config: 配置字典
        
    Returns:
        训练结果字典
    """
    # 自动检测目标列（如果未指定）
    target_column = config.get("target_column")
    if not target_column:
        # 尝试常见的目标列名
        import pandas as pd
        df = pd.read_csv(Path(train_path) / "data.csv" if Path(train_path).is_dir() else train_path, nrows=1)
        common_target_names = ['target', 'label', 'y', 'class', 'category', 'result']
        for col in df.columns:
            if col.lower() in common_target_names:
                target_column = col
                break
        # 如果没找到，使用最后一列
        if not target_column:
            target_column = df.columns[-1]
        logger.info(f"自动检测目标列: {target_column}")
    
    ml_config = MLTrainingConfig(
        task_type=config.get("task_type", "classification"),
        model_type=config.get("model_type", "random_forest"),
        target_column=target_column,
        feature_columns=config.get("feature_columns"),
        output_dir=output_dir,
        n_estimators=config.get("n_estimators", 100),
        max_depth=config.get("max_depth"),
        contamination=config.get("contamination", 0.1)
    )
    
    trainer = GeneralMLTrainer(ml_config)
    result = trainer.train(train_path, val_path)
    
    return {
        "success": True,
        "output_dir": output_dir,
        "result": result
    }


if __name__ == "__main__":
    # 测试分类
    test_df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "target": ["A", "A", "A", "B", "B", "B", "A", "A", "B", "B"]
    })
    test_df.to_csv("/tmp/test_ml.csv", index=False)
    
    result = run_ml_training(
        "/tmp/test_ml.csv",
        None,
        "/tmp/test_ml_output",
        {"task_type": "classification", "model_type": "random_forest", "target_column": "target"}
    )
    print(json.dumps(result, indent=2))
