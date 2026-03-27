# -*- coding: utf-8 -*-
"""
智能学习调度器 - 支持所有模型类型的统一学习管理
自动检测模型类型，选择增量学习或定时全量重训练
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import joblib

from db_postgres import execute_query, execute_update

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("/var/www/ai-training/models")
DATA_DIR = Path("/var/www/ai-training/data")


class ModelTypeDetector:
    """模型类型检测器"""
    
    @staticmethod
    def detect(job_id: str) -> str:
        """
        检测模型类型
        Returns: 'sklearn', 'pytorch', 'transformers', 'unknown'
        """
        rows = execute_query(
            'SELECT model_path, config FROM training_jobs WHERE id = %s',
            (job_id,)
        )
        if not rows:
            return 'unknown'
        
        model_path = Path(rows[0]['model_path']) if rows[0]['model_path'] else None
        config = rows[0]['config']
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except:
                config = {}
        
        if not model_path or not model_path.exists():
            # 从配置推断
            if config:
                if config.get('model_name') and 'bert' in config.get('model_name', '').lower():
                    return 'transformers'
                if config.get('task_type') == 'image_classification':
                    return 'pytorch'
            return 'unknown'
        
        # 检查文件类型
        if (model_path / 'model.joblib').exists() or (model_path / 'model.pkl').exists():
            return 'sklearn'
        
        if (model_path / 'final').exists():
            # 检查是否有 pytorch_model.bin 或 model.safetensors
            final_path = model_path / 'final'
            if any(final_path.glob('pytorch_model.bin')) or any(final_path.glob('model.safetensors')):
                return 'transformers'
        
        if (model_path / 'best_model.pth').exists() or (model_path / 'final_model.pth').exists():
            return 'pytorch'
        
        # 从配置文件推断
        if config:
            model_name = config.get('model_name', '')
            if any(x in model_name.lower() for x in ['bert', 'gpt', 't5', 'transformer']):
                return 'transformers'
            if config.get('task_type') == 'image_classification':
                return 'pytorch'
            if config.get('model_type') in ['random_forest', 'xgboost', 'lightgbm', 'svm']:
                return 'sklearn'
        
        return 'unknown'
    
    @staticmethod
    def get_learning_mode(model_type: str) -> str:
        """根据模型类型返回推荐的学习模式"""
        if model_type == 'sklearn':
            return 'incremental'
        elif model_type in ['pytorch', 'transformers']:
            return 'scheduled_retrain'
        return 'scheduled_retrain'  # 默认使用定时重训练


class DatasetVersionManager:
    """数据集版本管理 - 自动合并历史和新数据"""
    
    def prepare_training_dataset(self, project_id: str, 
                                 base_dataset_id: str = None,
                                 include_annotations: bool = True) -> Optional[str]:
        """
        准备训练数据集
        策略：
        1. 如果有基础数据集，加载历史数据
        2. 扫描项目下的所有新数据/标注
        3. 合并并去重
        4. 保存为新的训练集
        
        Returns: 新数据集路径 或 None
        """
        try:
            # 1. 获取历史数据
            historical_data = self._load_historical_data(project_id, base_dataset_id)
            logger.info(f"历史数据: {len(historical_data)} 条")
            
            # 2. 获取新数据
            new_data = self._load_new_data(project_id, include_annotations)
            logger.info(f"新数据: {len(new_data)} 条")
            
            if len(historical_data) == 0 and len(new_data) == 0:
                logger.warning("没有可用数据")
                return None
            
            # 3. 合并
            if len(historical_data) > 0 and len(new_data) > 0:
                combined = pd.concat([historical_data, new_data], ignore_index=True)
            elif len(historical_data) > 0:
                combined = historical_data
            else:
                combined = new_data
            
            # 去重（根据文本内容或图片路径）
            if 'text' in combined.columns:
                combined = combined.drop_duplicates(subset=['text'], keep='last')
            elif 'image_path' in combined.columns:
                combined = combined.drop_duplicates(subset=['image_path'], keep='last')
            
            # 4. 保存
            version_id = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir = DATA_DIR / project_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{version_id}.csv"
            
            combined.to_csv(output_path, index=False)
            logger.info(f"合并数据集已保存: {output_path} ({len(combined)} 条)")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"准备数据集失败: {e}")
            return None
    
    def _load_historical_data(self, project_id: str, dataset_id: str = None) -> pd.DataFrame:
        """加载历史数据"""
        if dataset_id:
            rows = execute_query(
                'SELECT file_path FROM datasets WHERE id = %s AND project_id = %s',
                (dataset_id, project_id)
            )
        else:
            # 使用项目最新的数据集
            rows = execute_query(
                '''SELECT file_path FROM datasets 
                   WHERE project_id = %s AND status = 'preprocessed'
                   ORDER BY created_at DESC LIMIT 1''',
                (project_id,)
            )
        
        if not rows:
            return pd.DataFrame()
        
        file_path = rows[0]['file_path']
        if not file_path or not Path(file_path).exists():
            return pd.DataFrame()
        
        try:
            if file_path.endswith('.xlsx'):
                return pd.read_excel(file_path)
            else:
                return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            return pd.DataFrame()
    
    def _load_new_data(self, project_id: str, include_annotations: bool) -> pd.DataFrame:
        """加载新数据（从标注任务等）"""
        new_data = []
        
        # 从标注任务加载
        if include_annotations:
            annotation_rows = execute_query(
                '''SELECT a.content, a.label 
                   FROM annotations a
                   JOIN annotation_tasks t ON a.task_id = t.id
                   WHERE t.project_id = %s AND a.status = 'completed'
                   AND a.updated_at > NOW() - INTERVAL '7 days'
                   ORDER BY a.updated_at DESC''',
                (project_id,)
            )
            
            if annotation_rows:
                df = pd.DataFrame(annotation_rows)
                df.columns = ['text', 'label']
                new_data.append(df)
                logger.info(f"从标注加载 {len(df)} 条")
        
        # TODO: 从其他来源加载新数据
        
        if new_data:
            return pd.concat(new_data, ignore_index=True)
        return pd.DataFrame()


class SmartLearningScheduler:
    """智能学习调度器"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.dataset_manager = DatasetVersionManager()
    
    def create_config(self, project_id: str, job_id: str,
                      trigger_schedule: str = 'weekly',
                      trigger_min_samples: int = 100,
                      trigger_performance_drop: float = 0.05,
                      auto_deploy: bool = False,
                      min_improvement: float = 0.0) -> Dict:
        """
        创建智能学习配置
        自动检测模型类型，选择合适的策略
        """
        # 检测模型类型
        model_type = ModelTypeDetector.detect(job_id)
        learning_mode = ModelTypeDetector.get_learning_mode(model_type)
        
        config_id = f"slc_{project_id[:8]}_{job_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        execute_update('''
            INSERT INTO smart_learning_config 
            (id, project_id, job_id, model_type, learning_mode, 
             trigger_schedule, trigger_min_samples, trigger_performance_drop,
             auto_deploy, min_improvement)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (config_id, project_id, job_id, model_type, learning_mode,
              trigger_schedule, trigger_min_samples, trigger_performance_drop,
              auto_deploy, min_improvement))
        
        mode_desc = '增量学习' if learning_mode == 'incremental' else '定时全量重训练'
        
        return {
            'success': True,
            'config_id': config_id,
            'model_type': model_type,
            'learning_mode': learning_mode,
            'message': f'已创建配置，检测到{model_type}模型，使用{mode_desc}'
        }
    
    def get_config(self, project_id: str, job_id: str = None) -> Optional[Dict]:
        """获取配置"""
        if job_id:
            rows = execute_query(
                '''SELECT * FROM smart_learning_config 
                   WHERE project_id = %s AND job_id = %s AND is_active = TRUE
                   ORDER BY created_at DESC LIMIT 1''',
                (project_id, job_id)
            )
        else:
            rows = execute_query(
                '''SELECT * FROM smart_learning_config 
                   WHERE project_id = %s AND is_active = TRUE
                   ORDER BY created_at DESC LIMIT 1''',
                (project_id,)
            )
        
        if not rows:
            return None
        
        row = rows[0]
        return {
            'id': row['id'],
            'project_id': row['project_id'],
            'job_id': row['job_id'],
            'model_type': row['model_type'],
            'learning_mode': row['learning_mode'],
            'trigger_schedule': row['trigger_schedule'],
            'trigger_min_samples': row['trigger_min_samples'],
            'trigger_performance_drop': row['trigger_performance_drop'],
            'auto_deploy': row['auto_deploy'],
            'min_improvement': row['min_improvement'],
            'is_active': row['is_active'],
            'trigger_count': row['trigger_count'],
            'success_count': row['success_count'],
            'last_trigger_at': row['last_trigger_at'].isoformat() if row['last_trigger_at'] else None,
            'created_at': row['created_at'].isoformat() if row['created_at'] else None
        }
    
    def should_trigger(self, config: Dict) -> Tuple[bool, str]:
        """
        检查是否应该触发学习
        Returns: (should_trigger, reason)
        """
        reasons = []
        
        # 1. 检查定时触发
        if self._check_schedule_trigger(config):
            reasons.append('schedule')
        
        # 2. 检查数据量触发
        if self._check_data_volume_trigger(config):
            reasons.append('data_volume')
        
        # 3. 检查性能触发
        if self._check_performance_trigger(config):
            reasons.append('performance_drop')
        
        return len(reasons) > 0, ','.join(reasons) if reasons else 'none'
    
    def _check_schedule_trigger(self, config: Dict) -> bool:
        """检查定时触发"""
        schedule = config.get('trigger_schedule')
        if not schedule or schedule == 'never':
            return False
        
        last_trigger = config.get('last_trigger_at')
        if not last_trigger:
            return True
        
        if isinstance(last_trigger, str):
            last_trigger = datetime.fromisoformat(last_trigger.replace('Z', '+00:00'))
        
        now = datetime.now()
        elapsed = now - last_trigger
        
        if schedule == 'daily':
            return elapsed >= timedelta(days=1)
        elif schedule == 'weekly':
            return elapsed >= timedelta(weeks=1)
        elif schedule == 'monthly':
            return elapsed >= timedelta(days=30)
        
        return False
    
    def _check_data_volume_trigger(self, config: Dict) -> bool:
        """检查数据量触发"""
        min_samples = config.get('trigger_min_samples', 100)
        project_id = config.get('project_id')
        
        # 统计最近7天的新标注数据
        rows = execute_query(
            '''SELECT COUNT(*) as cnt 
               FROM annotations a
               JOIN annotation_tasks t ON a.task_id = t.id
               WHERE t.project_id = %s AND a.status = 'completed'
               AND a.updated_at > NOW() - INTERVAL '7 days' ''',
            (project_id,)
        )
        
        new_samples = rows[0]['cnt'] if rows else 0
        return new_samples >= min_samples
    
    def _check_performance_trigger(self, config: Dict) -> bool:
        """检查性能触发（准确率下降）"""
        threshold = config.get('trigger_performance_drop', 0.05)
        project_id = config.get('project_id')
        job_id = config.get('job_id')
        
        # 获取最近的推理统计
        rows = execute_query(
            '''SELECT AVG(CASE WHEN success THEN 1 ELSE 0 END) as recent_acc
               FROM inference_logs
               WHERE project_id = %s AND job_id = %s
               AND created_at > NOW() - INTERVAL '1 day' ''',
            (project_id, job_id)
        )
        
        if not rows or rows[0]['recent_acc'] is None:
            return False
        
        recent_acc = rows[0]['recent_acc']
        
        # 获取训练时的基准准确率
        job_rows = execute_query(
            'SELECT best_accuracy FROM training_jobs WHERE id = %s',
            (job_id,)
        )
        
        if not job_rows or job_rows[0]['best_accuracy'] is None:
            return False
        
        base_acc = job_rows[0]['best_accuracy']
        
        # 准确率下降超过阈值
        return (base_acc - recent_acc) > threshold
    
    def trigger_learning(self, config_id: str, reason: str = 'manual',
                         dataset_id: str = None) -> Dict:
        """
        触发学习
        对于 sklearn 模型使用增量学习
        对于深度学习模型使用定时全量重训练
        """
        rows = execute_query(
            'SELECT * FROM smart_learning_config WHERE id = %s',
            (config_id,)
        )
        
        if not rows:
            return {'success': False, 'error': '配置不存在'}
        
        config = rows[0]
        learning_mode = config['learning_mode']
        
        # 创建学习作业记录
        job_id = f"slj_{config_id[:20]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        execute_update('''
            INSERT INTO scheduled_learning_jobs 
            (id, config_id, project_id, base_job_id, trigger_reason, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (job_id, config_id, config['project_id'], config['job_id'], reason, 'pending'))
        
        # 更新触发计数
        execute_update('''
            UPDATE smart_learning_config 
            SET trigger_count = trigger_count + 1, last_trigger_at = CURRENT_TIMESTAMP
            WHERE id = %s
        ''', (config_id,))
        
        if learning_mode == 'incremental':
            # 使用原有的增量学习
            return self._run_incremental_learning(job_id, config, dataset_id)
        else:
            # 使用定时全量重训练
            return self._run_scheduled_retrain(job_id, config, dataset_id)
    
    def _run_incremental_learning(self, job_id: str, config: Dict, dataset_id: str = None) -> Dict:
        """执行增量学习（sklearn模型）"""
        from online_learning import OnlineLearningManager
        
        try:
            manager = OnlineLearningManager()
            
            # 准备数据集
            dataset_path = self.dataset_manager.prepare_training_dataset(
                config['project_id'], dataset_id
            )
            
            if not dataset_path:
                execute_update(
                    "UPDATE scheduled_learning_jobs SET status = 'failed', error_message = %s WHERE id = %s",
                    ('没有可用数据', job_id)
                )
                return {'success': False, 'error': '没有可用数据'}
            
            # 创建在线学习任务并执行
            task_id = manager.create_learning_task(
                config['project_id'], config['job_id'], 'incremental'
            )
            
            result = manager.incremental_learn(task_id, dataset_path)
            
            # 更新学习作业状态
            execute_update('''
                UPDATE scheduled_learning_jobs 
                SET status = 'deployed', new_job_id = %s, deployed = TRUE, deployed_at = CURRENT_TIMESTAMP
                WHERE id = %s
            ''', (result.get('new_job_id'), job_id))
            
            # 更新成功计数
            execute_update(
                "UPDATE smart_learning_config SET success_count = success_count + 1 WHERE id = %s",
                (config['id'],)
            )
            
            return {
                'success': True,
                'learning_job_id': job_id,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"增量学习失败: {e}")
            execute_update(
                "UPDATE scheduled_learning_jobs SET status = 'failed', error_message = %s WHERE id = %s",
                (str(e), job_id)
            )
            return {'success': False, 'error': str(e)}
    
    def _run_scheduled_retrain(self, job_id: str, config: Dict, dataset_id: str = None) -> Dict:
        """
        执行定时全量重训练（深度学习模型）
        实际训练由 Celery 任务异步执行
        """
        try:
            # 更新状态为训练中
            execute_update(
                "UPDATE scheduled_learning_jobs SET status = 'training' WHERE id = %s",
                (job_id,)
            )
            
            # 准备数据集
            dataset_path = self.dataset_manager.prepare_training_dataset(
                config['project_id'], dataset_id, include_annotations=True
            )
            
            if not dataset_path:
                execute_update(
                    "UPDATE scheduled_learning_jobs SET status = 'failed', error_message = %s WHERE id = %s",
                    ('没有可用数据', job_id)
                )
                return {'success': False, 'error': '没有可用数据'}
            
            # 获取原模型配置
            job_rows = execute_query(
                'SELECT config, dataset_id FROM training_jobs WHERE id = %s',
                (config['job_id'],)
            )
            
            if not job_rows:
                return {'success': False, 'error': '基础模型不存在'}
            
            original_config = job_rows[0]['config']
            if isinstance(original_config, str):
                original_config = json.loads(original_config)
            
            # 创建新的训练任务
            from tasks import submit_training_task, submit_image_training_task, submit_ml_training_task
            
            new_job_id = f"retrain_{config['job_id'][:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 这里应该启动异步训练任务
            # 简化版：直接返回，训练由外部调用 /api/projects/{id}/train 完成
            
            return {
                'success': True,
                'learning_job_id': job_id,
                'message': '已准备重训练，请使用返回的数据集启动训练任务',
                'dataset_path': dataset_path,
                'training_config': original_config,
                'base_job_id': config['job_id']
            }
            
        except Exception as e:
            logger.error(f"定时重训练准备失败: {e}")
            execute_update(
                "UPDATE scheduled_learning_jobs SET status = 'failed', error_message = %s WHERE id = %s",
                (str(e), job_id)
            )
            return {'success': False, 'error': str(e)}
    
    def compare_and_deploy(self, learning_job_id: str, new_job_id: str,
                           auto_deploy: bool = False, min_improvement: float = 0.0) -> Dict:
        """
        对比新旧模型并决定是否部署
        """
        # 获取学习作业信息
        rows = execute_query(
            'SELECT * FROM scheduled_learning_jobs WHERE id = %s',
            (learning_job_id,)
        )
        
        if not rows:
            return {'success': False, 'error': '学习作业不存在'}
        
        job = rows[0]
        base_job_id = job['base_job_id']
        
        # 获取两个模型的性能
        base_info = execute_query(
            'SELECT best_accuracy FROM training_jobs WHERE id = %s',
            (base_job_id,)
        )
        new_info = execute_query(
            'SELECT best_accuracy FROM training_jobs WHERE id = %s',
            (new_job_id,)
        )
        
        if not base_info or not new_info:
            return {'success': False, 'error': '模型信息不完整'}
        
        base_acc = base_info[0]['best_accuracy'] or 0
        new_acc = new_info[0]['best_accuracy'] or 0
        improvement = new_acc - base_acc
        
        should_deploy = improvement >= min_improvement
        
        # 更新对比结果
        execute_update('''
            UPDATE scheduled_learning_jobs 
            SET status = %s, new_job_id = %s,
                base_accuracy = %s, new_accuracy = %s, accuracy_improvement = %s,
                deployed = %s, deployed_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE NULL END
            WHERE id = %s
        ''', ('deployed' if should_deploy else 'rejected', new_job_id,
              base_acc, new_acc, improvement,
              should_deploy, should_deploy, learning_job_id))
        
        if should_deploy:
            # 更新成功计数
            execute_update(
                "UPDATE smart_learning_config SET success_count = success_count + 1, last_success_at = CURRENT_TIMESTAMP WHERE id = %s",
                (job['config_id'],)
            )
        
        return {
            'success': True,
            'comparison': {
                'base_accuracy': base_acc,
                'new_accuracy': new_acc,
                'improvement': improvement,
                'should_deploy': should_deploy
            },
            'deployed': should_deploy,
            'reason': f'{"新模型更好" if should_deploy else "改进不足"} (改进: {improvement*100:.2f}%)'
        }
    
    def get_learning_history(self, project_id: str, job_id: str = None) -> List[Dict]:
        """获取统一的学习历史（包含增量和定时）"""
        history = []
        
        # 1. 获取增量学习历史（从 online_learning_tasks）
        from online_learning import OnlineLearningManager
        incremental_manager = OnlineLearningManager()
        incremental_history = incremental_manager.get_learning_history(project_id, job_id)
        
        for item in incremental_history:
            history.append({
                'type': 'incremental',
                'learning_type': item.get('learning_type'),
                'triggered_at': item.get('created_at'),
                'status': item.get('status'),
                'new_samples': item.get('new_samples'),
                'accuracy_before': item.get('accuracy_before'),
                'accuracy_after': item.get('accuracy_after'),
                'deployed': item.get('status') == 'completed'
            })
        
        # 2. 获取定时学习历史（从 scheduled_learning_jobs）
        if job_id:
            rows = execute_query(
                '''SELECT * FROM scheduled_learning_jobs 
                   WHERE project_id = %s AND base_job_id = %s
                   ORDER BY created_at DESC''',
                (project_id, job_id)
            )
        else:
            rows = execute_query(
                '''SELECT * FROM scheduled_learning_jobs 
                   WHERE project_id = %s
                   ORDER BY created_at DESC''',
                (project_id,)
            )
        
        for row in rows:
            history.append({
                'type': 'scheduled_retrain',
                'triggered_at': row['created_at'].isoformat() if row['created_at'] else None,
                'trigger_reason': row['trigger_reason'],
                'status': row['status'],
                'base_accuracy': row['base_accuracy'],
                'new_accuracy': row['new_accuracy'],
                'improvement': row['accuracy_improvement'],
                'deployed': row['deployed']
            })
        
        # 按时间排序
        history.sort(key=lambda x: x.get('triggered_at') or '', reverse=True)
        
        return history


# 全局调度器实例
scheduler = SmartLearningScheduler()


def check_all_scheduled_learning():
    """
    检查所有配置，触发需要执行的定时学习
    应由定时任务每分钟调用
    """
    rows = execute_query(
        'SELECT * FROM smart_learning_config WHERE is_active = TRUE'
    )
    
    triggered = []
    for row in rows:
        config = {
            'id': row['id'],
            'project_id': row['project_id'],
            'job_id': row['job_id'],
            'model_type': row['model_type'],
            'learning_mode': row['learning_mode'],
            'trigger_schedule': row['trigger_schedule'],
            'trigger_min_samples': row['trigger_min_samples'],
            'trigger_performance_drop': row['trigger_performance_drop'],
            'auto_deploy': row['auto_deploy'],
            'min_improvement': row['min_improvement'],
            'last_trigger_at': row['last_trigger_at']
        }
        
        should_trigger, reason = scheduler.should_trigger(config)
        
        if should_trigger:
            logger.info(f"触发定时学习: {config['id']}, 原因: {reason}")
            result = scheduler.trigger_learning(config['id'], reason)
            triggered.append({
                'config_id': config['id'],
                'reason': reason,
                'result': result
            })
    
    return triggered


if __name__ == "__main__":
    # 测试
    result = scheduler.create_config(
        'test-project',
        'test-job',
        trigger_schedule='weekly',
        auto_deploy=False
    )
    print(result)
