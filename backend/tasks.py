# -*- coding: utf-8 -*-
"""
训练任务管理 - Celery 版本
支持：异步训练、任务持久化、WebSocket推送、详细指标记录
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from celery import current_task
from celery.exceptions import SoftTimeLimitExceeded

from celeryconfig import celery_app
from db_postgres import execute_query, execute_update

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOGS_DIR = Path("/var/www/ai-training/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# WebSocket 连接管理器（将在main.py中注入）
websocket_manager = None

def set_websocket_manager(manager):
    """设置WebSocket管理器"""
    global websocket_manager
    websocket_manager = manager


def update_job_status(job_id: str, **kwargs):
    """更新任务状态到PostgreSQL"""
    fields = []
    values = []
    for key, value in kwargs.items():
        if value is not None:
            fields.append(f"{key} = %s")
            values.append(value)
    
    if fields:
        query = f"UPDATE training_jobs SET {', '.join(fields)} WHERE id = %s"
        values.append(job_id)
        execute_update(query, tuple(values))
        logger.debug(f"更新任务 {job_id} 状态: {kwargs}")


def send_feishu_notification(project_id: str, job_id: str, status: str, result: Dict = None):
    """发送飞书通知"""
    try:
        # 获取通知配置
        rows = execute_query('''
            SELECT webhook_url, notify_on_success, notify_on_failure 
            FROM feishu_notifications WHERE project_id = %s
        ''', (project_id,))
        
        if not rows:
            return
        
        config = rows[0]
        webhook_url = config['webhook_url']
        notify_success = config['notify_on_success']
        notify_failure = config['notify_on_failure']
        
        # 检查是否需要通知
        if status == 'completed' and not notify_success:
            return
        if status == 'failed' and not notify_failure:
            return
        
        # 获取任务信息
        rows = execute_query('''
            SELECT j.model_name, j.best_accuracy, j.best_val_loss, j.current_epoch, j.total_epochs,
                   p.name as project_name
            FROM training_jobs j
            JOIN projects p ON j.project_id = p.id
            WHERE j.id = %s
        ''', (job_id,))
        
        if not rows:
            return
        
        row = rows[0]
        model_name = row['model_name']
        best_acc = row['best_accuracy']
        best_loss = row['best_val_loss']
        current_epoch = row['current_epoch']
        total_epochs = row['total_epochs']
        project_name = row['project_name']
        
        # 构建飞书消息
        if status == 'completed':
            title = f"✅ 训练完成 - {project_name}"
            content = f"**模型**: {model_name}\n"
            content += f"**准确率**: {best_acc * 100:.2f}%\n" if best_acc else ""
            content += f"**Loss**: {best_loss:.4f}\n" if best_loss else ""
            content += f"**训练轮次**: {current_epoch}/{total_epochs}\n"
            content += f"**任务ID**: {job_id[:8]}..."
            color = "green"
        else:
            title = f"❌ 训练失败 - {project_name}"
            content = f"**模型**: {model_name}\n"
            content += f"**错误**: {result.get('error', '未知错误')[:200]}\n"
            content += f"**任务ID**: {job_id[:8]}..."
            color = "red"
        
        # 发送飞书消息
        import urllib.request
        
        message = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": color
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": content}
                    }
                ]
            }
        }
        
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(message).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            logger.info(f"飞书通知发送成功: {job_id}")
            
    except Exception as e:
        logger.error(f"飞书通知发送失败: {e}")


def broadcast_training_update_sync(job_id: str, data: Dict):
    """同步版本的WebSocket广播（用于Celery任务）"""
    if websocket_manager:
        try:
            # 创建新的事件循环来运行异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(websocket_manager.broadcast(job_id, data))
            loop.close()
        except Exception as e:
            logger.debug(f"WebSocket广播失败（可能无客户端连接）: {e}")


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_training_task(self, job_id: str, project_id: str, dataset_id: str, config: Dict) -> Dict:
    """
    执行NLP训练任务 - Celery版本
    
    Args:
        job_id: 任务ID
        project_id: 项目ID
        dataset_id: 数据集ID
        config: 训练配置
    
    Returns:
        训练结果
    """
    import sys
    sys.path.insert(0, '/var/www/ai-training/backend')
    from trainer import run_training
    
    logger.info(f"[Celery] 开始NLP训练任务: {job_id}")
    
    try:
        # 更新状态为训练中
        update_job_status(
            job_id,
            status='training',
            total_epochs=config.get('epochs', 3),
            started_at=datetime.now().isoformat()
        )
        
        # 更新Celery任务状态
        self.update_state(state='PROGRESS', meta={'progress': 0, 'message': '准备训练数据'})
        
        # 获取数据集路径
        rows = execute_query("SELECT file_path FROM datasets WHERE id = %s", (dataset_id,))
        if not rows:
            raise ValueError("数据集不存在")
        
        dataset_path = rows[0]['file_path']
        train_path = dataset_path.replace('.csv', '_train.csv')
        val_path = dataset_path.replace('.csv', '_val.csv')
        
        if not Path(train_path).exists():
            train_path = dataset_path
            val_path = None
        
        # 输出目录
        output_dir = f"/var/www/ai-training/models/{project_id}/{job_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 日志文件
        log_path = LOGS_DIR / f"{job_id}.log"
        
        # 训练指标历史
        last_broadcast_step = 0
        
        def progress_callback(logs):
            """进度回调"""
            nonlocal last_broadcast_step
            
            epoch = logs.get('epoch', 0)
            step = logs.get('step', 0)
            loss = logs.get('loss')
            val_loss = logs.get('eval_loss')
            lr = logs.get('learning_rate')
            
            # 计算进度
            total_epochs = config.get('epochs', 3)
            progress = min(int((epoch / total_epochs) * 100), 99)
            
            # 更新数据库
            update_data = {
                'current_epoch': int(epoch),
                'progress': progress
            }
            if loss is not None:
                update_data['current_loss'] = loss
            if lr is not None:
                update_data['learning_rate'] = lr
            
            update_job_status(job_id, **update_data)
            
            # 更新Celery任务状态（用于Flower监控）
            if step - last_broadcast_step >= 50 or logs.get('type') == 'epoch_end':
                last_broadcast_step = step
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress,
                        'epoch': epoch,
                        'loss': loss,
                        'message': f'Epoch {epoch:.1f}, Loss: {loss:.4f}'
                    }
                )
                
                # WebSocket实时推送
                broadcast_training_update_sync(job_id, {
                    "type": "training_update",
                    "epoch": epoch,
                    "progress": progress,
                    "loss": loss,
                    "val_loss": val_loss,
                    "learning_rate": lr,
                    "timestamp": datetime.now().isoformat()
                })
            
            # 记录日志
            with open(log_path, "a") as f:
                timestamp = datetime.now().isoformat()
                f.write(f"[{timestamp}] Epoch {epoch:.2f}, Step {step}, Loss: {loss}, LR: {lr}\n")
        
        # 执行训练
        result = run_training(
            train_path=train_path,
            val_path=val_path if val_path and Path(val_path).exists() else None,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            job_id=job_id
        )
        
        # 更新完成状态
        eval_report = result.get('eval_report', {})
        update_job_status(
            job_id,
            status='completed',
            progress=100,
            best_accuracy=result.get('eval_results', {}).get('eval_accuracy'),
            best_val_loss=result.get('eval_results', {}).get('eval_loss'),
            model_path=output_dir,
            eval_report=json.dumps(eval_report, ensure_ascii=False) if eval_report else None,
            early_stopped=result.get('early_stopped', False),
            stop_reason='early_stopping' if result.get('early_stopped') else 'completed',
            completed_at=datetime.now().isoformat()
        )
        
        # 发送飞书通知
        send_feishu_notification(project_id, job_id, 'completed', result)
        
        logger.info(f"[Celery] 训练完成: {job_id}")
        
        return {
            "success": True,
            "job_id": job_id,
            "result": result
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"[Celery] 训练超时: {job_id}")
        update_job_status(
            job_id,
            status='failed',
            stop_reason='timeout',
            completed_at=datetime.now().isoformat()
        )
        send_feishu_notification(project_id, job_id, 'failed', {'error': '训练超时'})
        raise
        
    except Exception as e:
        logger.error(f"[Celery] 训练失败: {e}")
        update_job_status(
            job_id,
            status='failed',
            stop_reason=str(e)[:200],
            completed_at=datetime.now().isoformat()
        )
        send_feishu_notification(project_id, job_id, 'failed', {'error': str(e)})
        
        # 记录错误日志
        log_path = LOGS_DIR / f"{job_id}.log"
        with open(log_path, "a") as f:
            f.write(f"\n[ERROR] {str(e)}\n")
        
        # 重试逻辑
        if self.request.retries < self.max_retries:
            logger.info(f"[Celery] 任务将在60秒后重试: {job_id}")
            raise self.retry(exc=e, countdown=60)
        
        raise


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_ml_training_task(self, job_id: str, project_id: str, dataset_id: str, config: Dict) -> Dict:
    """执行ML训练任务 - Celery版本"""
    import sys
    sys.path.insert(0, '/var/www/ai-training/backend')
    from ml_trainer import run_ml_training
    
    logger.info(f"[Celery] 开始ML训练任务: {job_id}, 类型: {config.get('task_type')}")
    
    try:
        # 更新状态
        update_job_status(
            job_id,
            status='training',
            total_epochs=config.get('epochs', config.get('n_estimators', 100)),
            started_at=datetime.now().isoformat()
        )
        
        self.update_state(state='PROGRESS', meta={'progress': 0, 'message': '准备训练数据'})
        
        # 获取数据集
        rows = execute_query("SELECT file_path FROM datasets WHERE id = %s", (dataset_id,))
        if not rows:
            raise ValueError("数据集不存在")
        
        dataset_path = rows[0]['file_path']
        
        # 输出目录
        output_dir = f"/var/www/ai-training/models/{project_id}/{job_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 日志
        log_path = LOGS_DIR / f"{job_id}.log"
        with open(log_path, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] ML训练开始: {config.get('task_type')} - {config.get('model_type')}\n")
        
        # 执行训练
        result = run_ml_training(
            train_path=dataset_path,
            val_path=None,
            output_dir=output_dir,
            config=config
        )
        
        # 提取关键指标
        task_type = config.get('task_type', 'classification')
        best_acc = None
        best_loss = None
        
        if task_type == 'classification':
            best_acc = result.get('result', {}).get('val_accuracy')
        elif task_type == 'regression':
            best_loss = result.get('result', {}).get('val_mse')
        elif task_type == 'anomaly_detection':
            best_acc = 1 - result.get('result', {}).get('val_anomaly_ratio', 0)
        
        # 更新完成状态
        update_job_status(
            job_id,
            status='completed',
            progress=100,
            best_accuracy=best_acc,
            best_val_loss=best_loss,
            model_path=output_dir,
            eval_report=json.dumps(result.get('result', {}), ensure_ascii=False, default=str),
            stop_reason='completed',
            completed_at=datetime.now().isoformat()
        )
        
        # 记录完成
        with open(log_path, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] ML训练完成\n")
        
        # 发送飞书通知
        send_feishu_notification(project_id, job_id, 'completed', result)
        
        logger.info(f"[Celery] ML训练完成: {job_id}")
        
        return {
            "success": True,
            "job_id": job_id,
            "result": result
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"[Celery] ML训练超时: {job_id}")
        update_job_status(
            job_id,
            status='failed',
            stop_reason='timeout',
            completed_at=datetime.now().isoformat()
        )
        send_feishu_notification(project_id, job_id, 'failed', {'error': '训练超时'})
        raise
        
    except Exception as e:
        logger.error(f"[Celery] ML训练失败: {e}")
        update_job_status(
            job_id,
            status='failed',
            stop_reason=str(e)[:200],
            completed_at=datetime.now().isoformat()
        )
        send_feishu_notification(project_id, job_id, 'failed', {'error': str(e)})
        
        # 记录错误日志
        log_path = LOGS_DIR / f"{job_id}.log"
        with open(log_path, "a") as f:
            f.write(f"\n[ERROR] {str(e)}\n")
        
        # 重试
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)
        
        raise


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_image_training_task(self, job_id: str, project_id: str, dataset_id: str, config: Dict) -> Dict:
    """
    执行图片分类训练任务 - Celery版本
    """
    import sys
    sys.path.insert(0, '/var/www/ai-training/backend')
    from image_trainer import run_image_classification_training, quantize_model
    
    logger.info(f"[Celery] 开始图片分类训练任务: {job_id}")
    
    try:
        # 更新状态
        update_job_status(
            job_id,
            status='training',
            total_epochs=config.get('epochs', 10),
            started_at=datetime.now().isoformat()
        )
        
        self.update_state(state='PROGRESS', meta={'progress': 0, 'message': '准备图片数据'})
        
        # 获取数据集路径
        rows = execute_query("SELECT file_path FROM datasets WHERE id = %s", (dataset_id,))
        if not rows:
            raise ValueError("数据集不存在")
        
        dataset_path = rows[0]['file_path']
        
        # 输出目录
        output_dir = f"/var/www/ai-training/models/{project_id}/{job_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 日志
        log_path = LOGS_DIR / f"{job_id}.log"
        with open(log_path, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] 图片分类训练开始: {config.get('model_name')} - {config.get('image_size', 224)}px\n")
        
        # 进度回调
        def progress_callback(logs):
            epoch = logs.get('epoch', 0)
            total_epochs = config.get('epochs', 10)
            progress = min(int((epoch / total_epochs) * 100), 99)
            val_acc = logs.get('val_acc')
            
            update_job_status(
                job_id,
                current_epoch=epoch,
                progress=progress,
                best_accuracy=logs.get('best_acc')
            )
            
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'message': f'Epoch {epoch}/{total_epochs}, Val Acc: {val_acc or 0:.4f}'
                }
            )
            
            # WebSocket实时推送
            broadcast_training_update_sync(job_id, {
                "type": "training_update",
                "epoch": epoch,
                "progress": progress,
                "val_acc": val_acc,
                "best_acc": logs.get('best_acc'),
                "timestamp": datetime.now().isoformat()
            })
        
        # 执行训练
        result = run_image_classification_training(
            train_dir=dataset_path,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            job_id=job_id
        )
        
        # 模型量化（如果配置启用）
        if config.get('quantize', False):
            self.update_state(state='PROGRESS', meta={'message': '正在进行模型量化...'})
            quantize_result = quantize_model(
                f"{output_dir}/best_model.pth",
                f"{output_dir}/model_quantized.pth"
            )
            if quantize_result['success']:
                result['quantized_path'] = f"{output_dir}/model_quantized.pth"
                result['quantized_info'] = quantize_result
        
        # 更新完成状态
        update_job_status(
            job_id,
            status='completed',
            progress=100,
            best_accuracy=result['best_accuracy'],
            model_path=output_dir,
            eval_report=json.dumps(result.get('history', {}), ensure_ascii=False),
            stop_reason='completed',
            completed_at=datetime.now().isoformat()
        )
        
        # 记录完成
        with open(log_path, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] 图片分类训练完成，准确率: {result['best_accuracy']:.4f}\n")
        
        # 发送飞书通知
        send_feishu_notification(project_id, job_id, 'completed', result)
        
        logger.info(f"[Celery] 图片分类训练完成: {job_id}")
        
        return {
            "success": True,
            "job_id": job_id,
            "result": result
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"[Celery] 图片训练超时: {job_id}")
        update_job_status(
            job_id,
            status='failed',
            stop_reason='timeout',
            completed_at=datetime.now().isoformat()
        )
        send_feishu_notification(project_id, job_id, 'failed', {'error': '训练超时'})
        raise
        
    except Exception as e:
        logger.error(f"[Celery] 图片训练失败: {e}")
        update_job_status(
            job_id,
            status='failed',
            stop_reason=str(e)[:200],
            completed_at=datetime.now().isoformat()
        )
        send_feishu_notification(project_id, job_id, 'failed', {'error': str(e)})
        
        log_path = LOGS_DIR / f"{job_id}.log"
        with open(log_path, "a") as f:
            f.write(f"\n[ERROR] {str(e)}\n")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)
        
        raise


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_object_detection_task(self, job_id: str, project_id: str, dataset_id: str, config: Dict) -> Dict:
    """执行目标检测训练任务 - YOLOv8"""
    import sys
    sys.path.insert(0, '/var/www/ai-training/backend')
    from detection_trainer import run_object_detection_training
    
    logger.info(f"[Celery] 开始目标检测训练任务: {job_id}")
    
    try:
        # 更新状态
        update_job_status(
            job_id,
            status='training',
            total_epochs=config.get('epochs', 100),
            started_at=datetime.now().isoformat()
        )
        
        self.update_state(state='PROGRESS', meta={'progress': 0, 'message': '准备目标检测数据集'})
        
        # 获取数据集
        rows = execute_query("SELECT file_path FROM datasets WHERE id = %s", (dataset_id,))
        if not rows:
            raise ValueError("数据集不存在")
        
        dataset_path = rows[0]['file_path']
        
        # 输出目录
        output_dir = f"/var/www/ai-training/models/{project_id}/{job_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 日志
        log_path = LOGS_DIR / f"{job_id}.log"
        with open(log_path, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] 目标检测训练开始: {config.get('model', 'yolov8s')}\n")
        
        # 进度回调
        def progress_callback(logs):
            epoch = logs.get('epoch', 0)
            total_epochs = config.get('epochs', 100)
            progress = min(int((epoch / total_epochs) * 100), 99)
            box_map = logs.get('box_map', 0)
            
            update_job_status(
                job_id,
                current_epoch=epoch,
                progress=progress,
                best_accuracy=box_map
            )
            
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'epoch': epoch,
                    'box_map': box_map,
                    'message': f'Epoch {epoch}/{total_epochs}, mAP: {box_map:.4f}'
                }
            )
            
            # WebSocket实时推送
            broadcast_training_update_sync(job_id, {
                "type": "training_update",
                "epoch": epoch,
                "progress": progress,
                "box_map": box_map,
                "timestamp": datetime.now().isoformat()
            })
        
        # 执行训练
        result = run_object_detection_training(
            train_dir=dataset_path,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            job_id=job_id
        )
        
        # 更新完成状态
        metrics = result.get('metrics', {})
        update_job_status(
            job_id,
            status='completed',
            progress=100,
            best_accuracy=metrics.get('box_map'),
            model_path=output_dir,
            eval_report=json.dumps(metrics, ensure_ascii=False),
            stop_reason='completed',
            completed_at=datetime.now().isoformat()
        )
        
        # 记录完成
        with open(log_path, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] 目标检测训练完成, mAP: {metrics.get('box_map', 0):.4f}\n")
        
        # 发送飞书通知
        send_feishu_notification(project_id, job_id, 'completed', result)
        
        logger.info(f"[Celery] 目标检测训练完成: {job_id}")
        
        return {
            "success": True,
            "job_id": job_id,
            "result": result
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"[Celery] 目标检测训练超时: {job_id}")
        update_job_status(
            job_id,
            status='failed',
            stop_reason='timeout',
            completed_at=datetime.now().isoformat()
        )
        send_feishu_notification(project_id, job_id, 'failed', {'error': '训练超时'})
        raise
        
    except Exception as e:
        logger.error(f"[Celery] 目标检测训练失败: {e}")
        update_job_status(
            job_id,
            status='failed',
            stop_reason=str(e)[:200],
            completed_at=datetime.now().isoformat()
        )
        send_feishu_notification(project_id, job_id, 'failed', {'error': str(e)})
        
        log_path = LOGS_DIR / f"{job_id}.log"
        with open(log_path, "a") as f:
            f.write(f"\n[ERROR] {str(e)}\n")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)
        
        raise


# 兼容性函数（保持API不变）
def submit_training_task(job_id: str, project_id: str, dataset_id: str, config: Dict):
    """提交NLP训练任务（兼容旧代码）"""
    return run_training_task.delay(job_id, project_id, dataset_id, config)

def submit_ml_training_task(job_id: str, project_id: str, dataset_id: str, config: Dict):
    """提交ML训练任务（兼容旧代码）"""
    return run_ml_training_task.delay(job_id, project_id, dataset_id, config)

def submit_image_training_task(job_id: str, project_id: str, dataset_id: str, config: Dict):
    """提交图片分类训练任务"""
    return run_image_training_task.delay(job_id, project_id, dataset_id, config)

def submit_object_detection_task(job_id: str, project_id: str, dataset_id: str, config: Dict):
    """提交目标检测训练任务"""
    return run_object_detection_task.delay(job_id, project_id, dataset_id, config)


if __name__ == "__main__":
    # 测试
    celery_app.start()
