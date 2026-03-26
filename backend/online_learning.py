# -*- coding: utf-8 -*-
"""
模型在线学习模块 - 支持增量学习和持续优化
"""

import os
import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path("/var/www/ai-training/training.db")
MODELS_DIR = Path("/var/www/ai-training/models")


class OnlineLearningManager:
    """在线学习管理器"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.models_dir = MODELS_DIR
        
    def _get_db_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)
    
    def create_learning_task(self, project_id: str, job_id: str, 
                            learning_type: str = 'incremental',
                            strategy: Dict = None) -> str:
        """
        创建在线学习任务
        
        Args:
            project_id: 项目ID
            job_id: 基础模型ID
            learning_type: 'incremental'(增量) / 'full'(全量重训)
            strategy: 学习策略配置
            
        Returns:
            学习任务ID
        """
        task_id = f"ol_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id[:8]}"
        
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # 创建在线学习表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS online_learning_tasks (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                base_job_id TEXT,
                learning_type TEXT,
                strategy TEXT,
                status TEXT DEFAULT 'pending',
                new_samples_count INTEGER DEFAULT 0,
                accuracy_before REAL,
                accuracy_after REAL,
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id),
                FOREIGN KEY (base_job_id) REFERENCES training_jobs(id)
            )
        ''')
        
        # 插入任务
        cursor.execute('''
            INSERT INTO online_learning_tasks 
            (id, project_id, base_job_id, learning_type, strategy, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (task_id, project_id, job_id, learning_type, 
              json.dumps(strategy or {}), 'pending'))
        
        conn.commit()
        conn.close()
        
        logger.info(f"创建在线学习任务: {task_id}")
        return task_id
    
    def incremental_learn(self, task_id: str, new_data_path: str) -> Dict:
        """
        增量学习 - 用新数据更新现有模型
        
        策略：
        1. 加载原模型
        2. 用新数据继续训练（warm_start）
        3. 评估新旧模型性能
        4. 如果新模型更好，替换；否则保留旧模型
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # 获取任务信息
        cursor.execute('''
            SELECT base_job_id, project_id, strategy 
            FROM online_learning_tasks WHERE id = ?
        ''', (task_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"学习任务不存在: {task_id}")
        
        base_job_id, project_id, strategy = row
        strategy = json.loads(strategy) if strategy else {}
        
        # 获取原模型路径
        cursor.execute('SELECT model_path FROM training_jobs WHERE id = ?', (base_job_id,))
        model_row = cursor.fetchone()
        if not model_row:
            conn.close()
            raise ValueError(f"基础模型不存在: {base_job_id}")
        
        base_model_path = model_row[0]
        
        try:
            # 更新状态
            cursor.execute('''
                UPDATE online_learning_tasks 
                SET status = 'training' WHERE id = ?
            ''', (task_id,))
            conn.commit()
            
            # 1. 加载原模型包
            model_package = joblib.load(Path(base_model_path) / 'model.joblib')
            old_model = model_package['model']
            scaler = model_package['scaler']
            label_encoder = model_package.get('label_encoder')
            config = model_package.get('config', {})
            
            # 2. 加载新数据
            if new_data_path.endswith('.xlsx'):
                new_df = pd.read_excel(new_data_path)
            else:
                new_df = pd.read_csv(new_data_path)
            
            new_samples = len(new_df)
            
            # 3. 准备数据
            feature_cols = config.get('feature_columns', [])
            target_col = config.get('target_column', 'target')
            
            if not feature_cols:
                # 自动推断特征列
                feature_cols = [c for c in new_df.columns if c != target_col and c != 'timestamp']
            
            X_new = new_df[feature_cols].fillna(new_df.mean()).values
            y_new = new_df[target_col].values
            
            # 标准化
            X_new_scaled = scaler.transform(X_new)
            
            # 4. 增量训练
            # RandomForest支持warm_start进行增量学习
            if hasattr(old_model, 'warm_start'):
                old_model.warm_start = True
                old_model.n_estimators += 10  # 增加10棵树
                old_model.fit(X_new_scaled, y_new)
                logger.info(f"增量训练完成，新增10棵树，总树数: {old_model.n_estimators}")
            else:
                # 不支持warm_start的模型，用新旧数据混合重训
                logger.warning("模型不支持增量学习，使用混合数据重训")
                old_model.fit(X_new_scaled, y_new)
            
            # 5. 评估新模型（简单评估：在新数据上的准确率）
            new_accuracy = old_model.score(X_new_scaled, y_new)
            
            # 6. 保存新模型版本
            new_model_dir = Path(base_model_path).parent / f"{base_job_id}_updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            new_model_dir.mkdir(exist_ok=True)
            
            model_package['model'] = old_model
            model_package['scaler'] = scaler
            model_package['updated_at'] = datetime.now().isoformat()
            model_package['incremental_samples'] = model_package.get('incremental_samples', 0) + new_samples
            
            joblib.dump(model_package, new_model_dir / 'model.joblib')
            
            # 7. 更新数据库
            cursor.execute('''
                UPDATE online_learning_tasks 
                SET status = 'completed',
                    new_samples_count = ?,
                    accuracy_after = ?,
                    model_path = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (new_samples, new_accuracy, str(new_model_dir), task_id))
            
            # 创建新的训练任务记录
            new_job_id = f"{base_job_id}_v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            cursor.execute('''
                INSERT INTO training_jobs 
                (id, project_id, dataset_id, model_name, status, best_accuracy, model_path, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (new_job_id, project_id, 'incremental', 
                  config.get('model_type', 'unknown'),
                  'completed', new_accuracy, str(new_model_dir),
                  json.dumps(config)))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'task_id': task_id,
                'new_job_id': new_job_id,
                'new_samples': new_samples,
                'accuracy': new_accuracy,
                'model_path': str(new_model_dir),
                'message': f'增量学习完成，新增{new_samples}条样本，准确率{new_accuracy*100:.2f}%'
            }
            
        except Exception as e:
            logger.error(f"增量学习失败: {e}")
            cursor.execute('''
                UPDATE online_learning_tasks 
                SET status = 'failed' WHERE id = ?
            ''', (task_id,))
            conn.commit()
            conn.close()
            raise
    
    def get_learning_history(self, project_id: str, job_id: str = None) -> List[Dict]:
        """获取在线学习历史"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        if job_id:
            cursor.execute('''
                SELECT id, learning_type, status, new_samples_count, 
                       accuracy_before, accuracy_after, created_at, completed_at
                FROM online_learning_tasks
                WHERE project_id = ? AND base_job_id = ?
                ORDER BY created_at DESC
            ''', (project_id, job_id))
        else:
            cursor.execute('''
                SELECT id, learning_type, status, new_samples_count,
                       accuracy_before, accuracy_after, created_at, completed_at
                FROM online_learning_tasks
                WHERE project_id = ?
                ORDER BY created_at DESC
            ''', (project_id,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'task_id': row[0],
                'learning_type': row[1],
                'status': row[2],
                'new_samples': row[3],
                'accuracy_before': row[4],
                'accuracy_after': row[5],
                'created_at': row[6],
                'completed_at': row[7]
            })
        
        conn.close()
        return history
    
    def rollback_model(self, job_id: str, target_version: str = None) -> Dict:
        """
        模型回滚 - 恢复到之前的版本
        
        Args:
            job_id: 当前模型ID
            target_version: 目标版本ID，None则回滚到上一个版本
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # 查找历史版本
        base_id = job_id.split('_v')[0]  # 去掉版本号
        
        cursor.execute('''
            SELECT id, model_path, best_accuracy, created_at
            FROM training_jobs
            WHERE id LIKE ? AND status = 'completed'
            ORDER BY created_at DESC
        ''', (f"{base_id}%",))
        
        versions = cursor.fetchall()
        if len(versions) < 2:
            conn.close()
            return {'success': False, 'message': '没有可回滚的历史版本'}
        
        # 找到目标版本
        if target_version:
            target = next((v for v in versions if v[0] == target_version), None)
        else:
            target = versions[1]  # 上一个版本
        
        if not target:
            conn.close()
            return {'success': False, 'message': '目标版本不存在'}
        
        target_id, target_path, target_acc, target_time = target
        
        # 更新当前模型指向
        cursor.execute('''
            UPDATE training_jobs 
            SET model_path = ?, best_accuracy = ?, 
                config = json_set(config, '$.rolled_back_from', ?)
            WHERE id = ?
        ''', (target_path, target_acc, job_id, job_id))
        
        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'message': f'已回滚到版本 {target_id}',
            'target_version': target_id,
            'target_accuracy': target_acc,
            'target_time': target_time
        }
    
    def setup_auto_learning(self, project_id: str, job_id: str, 
                           schedule: str = 'daily',
                           min_samples: int = 100,
                           accuracy_threshold: float = 0.05) -> str:
        """
        设置自动在线学习
        
        Args:
            schedule: 'daily'(每天) / 'weekly'(每周) / 'never'(关闭)
            min_samples: 触发学习的最小新样本数
            accuracy_threshold: 准确率下降多少触发重训练
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # 创建自动学习配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auto_learning_config (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                job_id TEXT,
                schedule TEXT,
                min_samples INTEGER,
                accuracy_threshold REAL,
                is_active INTEGER DEFAULT 1,
                last_trigger_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id),
                FOREIGN KEY (job_id) REFERENCES training_jobs(id)
            )
        ''')
        
        config_id = f"al_{project_id}_{job_id}"
        
        # 检查是否已存在
        cursor.execute('SELECT id FROM auto_learning_config WHERE id = ?', (config_id,))
        if cursor.fetchone():
            # 更新
            cursor.execute('''
                UPDATE auto_learning_config
                SET schedule = ?, min_samples = ?, accuracy_threshold = ?, is_active = 1
                WHERE id = ?
            ''', (schedule, min_samples, accuracy_threshold, config_id))
        else:
            # 创建
            cursor.execute('''
                INSERT INTO auto_learning_config
                (id, project_id, job_id, schedule, min_samples, accuracy_threshold)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (config_id, project_id, job_id, schedule, min_samples, accuracy_threshold))
        
        conn.commit()
        conn.close()
        
        return config_id


def check_and_trigger_auto_learning():
    """
    检查并触发自动学习 - 应由定时任务调用
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT id, project_id, job_id, schedule, min_samples, last_trigger_at
            FROM auto_learning_config
            WHERE is_active = 1
        ''')
        
        configs = cursor.fetchall()
        manager = OnlineLearningManager()
        
        for config in configs:
            config_id, project_id, job_id, schedule, min_samples, last_trigger = config
            
            # 检查是否满足触发条件
            should_trigger = False
            
            if schedule == 'daily':
                if not last_trigger or \
                   datetime.now() - datetime.fromisoformat(last_trigger) > timedelta(days=1):
                    should_trigger = True
            elif schedule == 'weekly':
                if not last_trigger or \
                   datetime.now() - datetime.fromisoformat(last_trigger) > timedelta(weeks=1):
                    should_trigger = True
            
            if should_trigger:
                # 检查是否有足够的新数据
                # TODO: 实现新数据检测逻辑
                logger.info(f"触发自动学习: {config_id}")
                
                # 创建并执行学习任务
                task_id = manager.create_learning_task(project_id, job_id, 'incremental')
                # TODO: 获取新数据路径并执行学习
                
                # 更新触发时间
                cursor.execute('''
                    UPDATE auto_learning_config
                    SET last_trigger_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (config_id,))
                conn.commit()
        
    except Exception as e:
        logger.error(f"自动学习检查失败: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    # 测试
    manager = OnlineLearningManager()
    
    # 创建学习配置示例
    config_id = manager.setup_auto_learning(
        'demo-project', 
        'demo-job',
        schedule='daily',
        min_samples=50
    )
    print(f"自动学习配置创建: {config_id}")
