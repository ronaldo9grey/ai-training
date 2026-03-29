# -*- coding: utf-8 -*-
"""
AI训练执行器 - 增强版
支持：早停机制、学习率自适应、训练可视化、自动保存最佳模型
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import numpy as np
import sqlite3
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置 HuggingFace 镜像（国内加速）
import os
if not os.environ.get('HF_ENDPOINT'):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    logger.info("使用 HuggingFace 镜像: https://hf-mirror.com")

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")


@dataclass
class TrainingConfig:
    """训练配置 - 增强版"""
    model_name: str = "bert-base-chinese"
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    max_length: int = 512
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # 自动调参配置
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    lr_scheduler_type: str = "cosine"  # linear, cosine, polynomial, constant
    lr_decay: bool = True
    
    # 检查点配置
    save_total_limit: int = 3  # 最多保留几个检查点
    load_best_model_at_end: bool = True
    
    # 数据配置
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # 数据库配置
    db_path: str = "/var/www/ai-training/training.db"
    job_id: Optional[str] = None


class MetricsCallback(TrainerCallback):
    """指标记录回调 - 保存到数据库用于可视化"""
    
    def __init__(self, job_id: str, db_path: str):
        self.job_id = job_id
        self.db_path = db_path
        self.metrics_history = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not self.job_id:
            return control
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 提取指标
            epoch = state.epoch if state.epoch else 0
            step = state.global_step
            train_loss = logs.get('loss', None)
            val_loss = logs.get('eval_loss', None)
            train_acc = logs.get('accuracy', None)
            val_acc = logs.get('eval_accuracy', None)
            lr = logs.get('learning_rate', None)
            
            # 插入数据库
            cursor.execute('''
                INSERT INTO training_metrics 
                (job_id, epoch, step, train_loss, val_loss, train_accuracy, val_accuracy, learning_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (self.job_id, epoch, step, train_loss, val_loss, train_acc, val_acc, lr))
            
            conn.commit()
            conn.close()
            
            # 更新任务最新指标
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = []
            values = []
            
            if val_acc is not None:
                update_fields.append("best_accuracy = ?")
                values.append(val_acc)
            if val_loss is not None:
                update_fields.append("best_val_loss = ?")
                values.append(val_loss)
            if lr is not None:
                update_fields.append("learning_rate = ?")
                values.append(lr)
            
            if update_fields:
                values.append(self.job_id)
                query = f"UPDATE training_jobs SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(query, values)
                conn.commit()
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"保存指标失败: {e}")
            
        return control


class ProgressCallback(TrainerCallback):
    """训练进度回调 - 支持外部回调函数"""
    
    def __init__(self, callback_func: Optional[Callable] = None):
        self.callback = callback_func
        self.current_epoch = 0
        self.best_metric = 0
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch) if state.epoch else 0
        self._notify({
            "type": "epoch_start",
            "epoch": self.current_epoch,
            "step": state.global_step,
        })
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self._notify({
            "type": "epoch_end",
            "epoch": self.current_epoch,
            "step": state.global_step,
        })
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.callback and logs:
            logs["type"] = "log"
            logs["epoch"] = self.current_epoch
            logs["step"] = state.global_step
            
            # 计算进度
            if state.max_steps:
                logs["progress"] = min(int((state.global_step / state.max_steps) * 100), 99)
            else:
                logs["progress"] = min(int((self.current_epoch / args.num_train_epochs) * 100), 99)
            
            self.callback(logs)
        return control
    
    def _notify(self, data):
        if self.callback:
            self.callback(data)


class TextClassifierTrainer:
    """文本分类训练器 - 增强版"""
    
    SUPPORTED_MODELS = {
        "bert-base-chinese": "bert-base-chinese",
        "distilbert-base-chinese": "distilbert-base-chinese",
        "chinese-roberta-wwm-ext": "hfl/chinese-roberta-wwm-ext",
        "macbert-base-chinese": "hfl/chinese-macbert-base",
        "tiny-bert": "prajjwal1/bert-tiny",  # 超小模型，适合测试
    }
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label2id = {}
        self.id2label = {}
        self.training_history = []
        
    def load_data(self, train_path: str, val_path: Optional[str] = None, 
                  text_column: str = "text", label_column: str = "label"):
        """加载数据集"""
        import pandas as pd
        
        train_df = pd.read_csv(train_path)
        
        labels = sorted(train_df[label_column].unique())
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for i, label in enumerate(labels)}
        num_labels = len(labels)
        
        logger.info(f"标签类别: {labels}")
        logger.info(f"训练样本数: {len(train_df)}")
        
        train_df["label_id"] = train_df[label_column].map(self.label2id)
        
        train_dataset = Dataset.from_pandas(train_df[[text_column, "label_id"]])
        train_dataset = train_dataset.rename_column("label_id", "labels")
        
        val_dataset = None
        if val_path and Path(val_path).exists():
            val_df = pd.read_csv(val_path)
            val_df["label_id"] = val_df[label_column].map(self.label2id)
            val_dataset = Dataset.from_pandas(val_df[[text_column, "label_id"]])
            val_dataset = val_dataset.rename_column("label_id", "labels")
            logger.info(f"验证样本数: {len(val_df)}")
        
        return train_dataset, val_dataset, num_labels
    
    def setup_model(self, num_labels: int):
        """初始化模型"""
        model_name = self.SUPPORTED_MODELS.get(
            self.config.model_name, 
            self.config.model_name
        )
        
        logger.info(f"加载模型: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        self.model.to(device)
        
    def preprocess_function(self, examples, text_column="text"):
        """预处理函数"""
        result = self.tokenizer(
            examples[text_column],
            truncation=True,
            padding=False,
            max_length=self.config.max_length
        )
        if "labels" in examples:
            result["labels"] = examples["labels"]
        return result
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # 计算每个类别的指标
        per_class = classification_report(
            labels, predictions, 
            target_names=[self.id2label[i] for i in range(len(self.id2label))],
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class': per_class
        }
    
    def train(self, train_dataset, val_dataset=None, progress_callback=None):
        """执行训练 - 增强版"""
        # 预处理
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["text"]
        )
        
        if val_dataset:
            val_dataset = val_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=["text"]
            )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # 计算总步数和评估策略
        total_steps = (len(train_dataset) // self.config.batch_size) * self.config.epochs
        eval_strategy = "steps" if val_dataset else "no"
        
        # 训练参数 - 增强版
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps if val_dataset else None,
            save_steps=self.config.save_steps,
            eval_strategy=eval_strategy,
            save_strategy="steps",
            logging_strategy="steps",
            load_best_model_at_end=self.config.load_best_model_at_end and val_dataset is not None,
            metric_for_best_model="accuracy" if val_dataset else None,
            greater_is_better=True,
            save_total_limit=self.config.save_total_limit,
            report_to=["none"],
            remove_unused_columns=False,
            dataloader_num_workers=0,
            fp16=torch.cuda.is_available(),
            max_grad_norm=self.config.max_grad_norm,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        
        # 创建回调列表
        callbacks = []
        
        # 进度回调
        if progress_callback:
            callbacks.append(ProgressCallback(progress_callback))
        
        # 指标记录回调
        if self.config.job_id:
            callbacks.append(MetricsCallback(self.config.job_id, self.config.db_path))
        
        # 早停回调
        if self.config.early_stopping and val_dataset:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold
            ))
        
        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        # 开始训练
        logger.info("开始训练...")
        logger.info(f"训练参数: epochs={self.config.epochs}, lr={self.config.learning_rate}, batch_size={self.config.batch_size}")
        logger.info(f"早停: {'开启' if self.config.early_stopping else '关闭'}, 耐心值={self.config.early_stopping_patience}")
        
        train_result = self.trainer.train()
        
        # 保存最终模型
        self.trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))
        
        # 保存标签映射
        with open(self.output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
            json.dump({
                "label2id": self.label2id,
                "id2label": self.id2label
            }, f, ensure_ascii=False, indent=2)
        
        # 评估
        eval_results = {}
        eval_report = {}
        
        if val_dataset:
            eval_results = self.trainer.evaluate()
            logger.info(f"评估结果: {eval_results}")
            
            # 生成详细评估报告
            predictions = self.trainer.predict(val_dataset)
            preds = np.argmax(predictions.predictions, axis=1)
            labels = predictions.label_ids
            
            # 混淆矩阵
            cm = confusion_matrix(labels, preds)
            
            # 详细分类报告
            report = classification_report(
                labels, preds,
                target_names=[self.id2label[i] for i in range(len(self.id2label))],
                output_dict=True,
                zero_division=0
            )
            
            eval_report = {
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "num_samples": len(labels),
                "num_classes": len(self.id2label)
            }
        
        # 训练历史
        training_history = {
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "total_steps": train_result.global_step,
            "final_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
        }
        
        return {
            "eval_results": eval_results,
            "eval_report": eval_report,
            "training_history": training_history,
            "early_stopped": self.trainer.state.global_step < total_steps if self.config.early_stopping else False
        }
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """预测"""
        if not self.model or not self.tokenizer:
            raise ValueError("模型未加载")
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        results = []
        for i, pred in enumerate(predictions):
            pred_id = torch.argmax(pred).item()
            confidence = pred[pred_id].item()
            results.append({
                "label": self.id2label[pred_id],
                "confidence": confidence,
                "all_probs": {self.id2label[j]: prob.item() for j, prob in enumerate(pred)}
            })
        
        return results


def run_training(
    train_path: str,
    val_path: Optional[str],
    output_dir: str,
    config: Dict,
    progress_callback=None,
    job_id: Optional[str] = None
) -> Dict:
    """
    执行训练的便捷函数 - 增强版
    """
    # CPU环境优化
    is_cpu = device.type == "cpu"
    batch_size = config.get("batch_size", 16)
    if is_cpu:
        batch_size = min(batch_size, 8)
        logger.info(f"CPU环境，调整batch_size为: {batch_size}")
    
    # 构建配置
    training_config = TrainingConfig(
        model_name=config.get("model_name", "bert-base-chinese"),
        learning_rate=config.get("learning_rate", 2e-5),
        batch_size=batch_size,
        epochs=config.get("epochs", 3),
        max_length=config.get("max_length", 512),
        output_dir=output_dir,
        # 自动调参配置
        early_stopping=config.get("early_stopping", True),
        early_stopping_patience=config.get("early_stopping_patience", 3),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        save_total_limit=config.get("save_total_limit", 3),
        job_id=job_id,
        db_path=config.get("db_path", "/var/www/ai-training/training.db")
    )
    
    # 创建训练器
    trainer = TextClassifierTrainer(training_config)
    
    # 加载数据
    train_dataset, val_dataset, num_labels = trainer.load_data(
        train_path, val_path
    )
    
    # 设置模型
    trainer.setup_model(num_labels)
    
    # 训练
    result = trainer.train(train_dataset, val_dataset, progress_callback)
    
    return {
        "success": True,
        "output_dir": output_dir,
        "eval_results": result["eval_results"],
        "eval_report": result["eval_report"],
        "training_history": result["training_history"],
        "early_stopped": result["early_stopped"],
        "num_labels": num_labels,
        "device": str(device)
    }


if __name__ == "__main__":
    # 测试
    config = TrainingConfig(
        model_name="bert-base-chinese",
        epochs=1,
        batch_size=4
    )
    
    import pandas as pd
    test_data = pd.DataFrame({
        "text": ["这部电影很好看", "太失望了", "一般般", "非常精彩", "waste of time"],
        "label": ["正面", "负面", "中性", "正面", "负面"]
    })
    test_data.to_csv("/tmp/test_train.csv", index=False)
    
    result = run_training(
        "/tmp/test_train.csv",
        None,
        "/tmp/test_output",
        {"epochs": 1, "batch_size": 4}
    )
    print(result)
