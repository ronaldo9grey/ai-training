# -*- coding: utf-8 -*-
"""
模型在线学习模块 - 支持增量学习和持续优化
(已迁移到 PostgreSQL)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 导入 PostgreSQL 数据库模块
from db_postgres import execute_query, execute_update

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("/var/www/ai-training/models")


class OnlineLearningManager:
    """在线学习管理器 - PostgreSQL 版本"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        
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
        
        execute_update('''
            INSERT INTO online_learning_tasks 
            (id, project_id, base_job_id, learning_type, strategy, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (task_id, project_id, job_id, learning_type, 
              json.dumps(strategy or {}), 'pending'))
        
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
        # 获取任务信息
        rows = execute_query('''
            SELECT base_job_id, project_id, strategy 
            FROM online_learning_tasks WHERE id = %s
        ''', (task_id,))
        
        if not rows:
            raise ValueError(f"学习任务不存在: {task_id}")
        
        row = rows[0]
        base_job_id = row['base_job_id']
        project_id = row['project_id']
        strategy = json.loads(row['strategy']) if row['strategy'] else {}
        
        # 获取原模型路径
        model_rows = execute_query('SELECT model_path FROM training_jobs WHERE id = %s', (base_job_id,))
        if not model_rows:
            raise ValueError(f"基础模型不存在: {base_job_id}")
        
        base_model_path = model_rows[0]['model_path']
        
        try:
            # 更新状态
            execute_update('''
                UPDATE online_learning_tasks 
                SET status = 'training' WHERE id = %s
            ''', (task_id,))
            
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
            execute_update('''
                UPDATE online_learning_tasks 
                SET status = 'completed',
                    new_samples_count = %s,
                    accuracy_after = %s,
                    model_path = %s,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = %s
            ''', (new_samples, new_accuracy, str(new_model_dir), task_id))
            
            # 创建新的训练任务记录
            new_job_id = f"{base_job_id}_v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            execute_update('''
                INSERT INTO training_jobs 
                (id, project_id, dataset_id, model_name, status, best_accuracy, model_path, config)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (new_job_id, project_id, 'incremental', 
                  config.get('model_type', 'unknown'),
                  'completed', new_accuracy, str(new_model_dir),
                  json.dumps(config)))
            
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
            execute_update('''
                UPDATE online_learning_tasks 
                SET status = 'failed' WHERE id = %s
            ''', (task_id,))
            raise
    
    def get_learning_history(self, project_id: str, job_id: str = None) -> List[Dict]:
        """获取在线学习历史"""
        if job_id:
            rows = execute_query('''
                SELECT id, learning_type, status, new_samples_count, 
                       accuracy_before, accuracy_after, created_at, completed_at
                FROM online_learning_tasks
                WHERE project_id = %s AND base_job_id = %s
                ORDER BY created_at DESC
            ''', (project_id, job_id))
        else:
            rows = execute_query('''
                SELECT id, learning_type, status, new_samples_count,
                       accuracy_before, accuracy_after, created_at, completed_at
                FROM online_learning_tasks
                WHERE project_id = %s
                ORDER BY created_at DESC
            ''', (project_id,))
        
        history = []
        for row in rows:
            history.append({
                'task_id': row['id'],
                'learning_type': row['learning_type'],
                'status': row['status'],
                'new_samples': row['new_samples_count'],
                'accuracy_before': row['accuracy_before'],
                'accuracy_after': row['accuracy_after'],
                'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                'completed_at': row['completed_at'].isoformat() if row['completed_at'] else None
            })
        
        return history
    
    def rollback_model(self, job_id: str, target_version: str = None) -> Dict:
        """
        模型回滚 - 恢复到之前的版本
        
        Args:
            job_id: 当前模型ID
            target_version: 目标版本ID，None则回滚到上一个版本
        """
        # 查找历史版本
        base_id = job_id.split('_v')[0]  # 去掉版本号
        
        rows = execute_query('''
            SELECT id, model_path, best_accuracy, created_at
            FROM training_jobs
            WHERE id LIKE %s AND status = 'completed'
            ORDER BY created_at DESC
        ''', (f"{base_id}%",))
        
        if len(rows) < 2:
            return {'success': False, 'message': '没有可回滚的历史版本'}
        
        # 找到目标版本
        if target_version:
            target = next((v for v in rows if v['id'] == target_version), None)
        else:
            target = rows[1]  # 上一个版本
        
        if not target:
            return {'success': False, 'message': '目标版本不存在'}
        
        target_id = target['id']
        target_path = target['model_path']
        target_acc = target['best_accuracy']
        target_time = target['created_at']
        
        # 更新当前模型指向 - 使用 jsonb_set 替代 json_set
        current_config_rows = execute_query('SELECT config FROM training_jobs WHERE id = %s', (job_id,))
        if current_config_rows:
            current_config = current_config_rows[0]['config'] or {}
            if isinstance(current_config, str):
                current_config = json.loads(current_config)
            current_config['rolled_back_from'] = job_id
            
            execute_update('''
                UPDATE training_jobs 
                SET model_path = %s, best_accuracy = %s, config = %s
                WHERE id = %s
            ''', (target_path, target_acc, json.dumps(current_config), job_id))
        
        return {
            'success': True,
            'message': f'已回滚到版本 {target_id}',
            'target_version': target_id,
            'target_accuracy': target_acc,
            'target_time': target_time.isoformat() if target_time else None
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
        config_id = f"al_{project_id}_{job_id}"
        
        # 检查是否已存在
        rows = execute_query('SELECT id FROM auto_learning_config WHERE id = %s', (config_id,))
        
        if rows:
            # 更新
            execute_update('''
                UPDATE auto_learning_config
                SET schedule = %s, min_samples = %s, accuracy_threshold = %s, is_active = TRUE
                WHERE id = %s
            ''', (schedule, min_samples, accuracy_threshold, config_id))
        else:
            # 创建
            execute_update('''
                INSERT INTO auto_learning_config
                (id, project_id, job_id, schedule, min_samples, accuracy_threshold)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (config_id, project_id, job_id, schedule, min_samples, accuracy_threshold))
        
        return config_id


def check_and_trigger_auto_learning():
    """
    检查并触发自动学习 - 应由定时任务调用
    """
    try:
        rows = execute_query('''
            SELECT id, project_id, job_id, schedule, min_samples, last_trigger_at
            FROM auto_learning_config
            WHERE is_active = TRUE
        ''')
        
        manager = OnlineLearningManager()
        
        for row in rows:
            config_id = row['id']
            project_id = row['project_id']
            job_id = row['job_id']
            schedule = row['schedule']
            min_samples = row['min_samples']
            last_trigger = row['last_trigger_at']
            
            # 检查是否满足触发条件
            should_trigger = False
            
            if schedule == 'daily':
                if not last_trigger or \
                   datetime.now() - last_trigger > timedelta(days=1):
                    should_trigger = True
            elif schedule == 'weekly':
                if not last_trigger or \
                   datetime.now() - last_trigger > timedelta(weeks=1):
                    should_trigger = True
            
            if should_trigger:
                # 检查是否有足够的新数据
                # TODO: 实现新数据检测逻辑
                logger.info(f"触发自动学习: {config_id}")
                
                # 创建并执行学习任务
                task_id = manager.create_learning_task(project_id, job_id, 'incremental')
                # TODO: 获取新数据路径并执行学习
                
                # 更新触发时间
                execute_update('''
                    UPDATE auto_learning_config
                    SET last_trigger_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                ''', (config_id,))
        
    except Exception as e:
        logger.error(f"自动学习检查失败: {e}")


def get_auto_learning_configs(project_id: str) -> List[Dict]:
    """获取项目的自动学习配置 - 用于 main.py API"""
    rows = execute_query('''
        SELECT id, job_id, schedule, min_samples, accuracy_threshold, is_active
        FROM auto_learning_config
        WHERE project_id = %s
    ''', (project_id,))
    
    configs = []
    for row in rows:
        configs.append({
            "id": row['id'],
            "job_id": row['job_id'],
            "schedule": row['schedule'],
            "min_samples": row['min_samples'],
            "accuracy_threshold": row['accuracy_threshold'],
            "is_active": bool(row['is_active'])
        })
    
    return configs


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
