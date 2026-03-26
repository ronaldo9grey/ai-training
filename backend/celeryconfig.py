# -*- coding: utf-8 -*-
"""
Celery 配置文件
"""

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 Celery 应用
celery_app = Celery(
    'ai_training',
    broker='redis://localhost:6379/0',  # 任务队列
    backend='redis://localhost:6379/1',  # 结果存储
    include=['tasks']  # 包含的任务模块
)

# Celery 配置
celery_app.conf.update(
    # 任务序列化
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
    
    # Worker 配置
    worker_prefetch_multiplier=1,  # 每个worker只预取1个任务
    task_acks_late=True,  # 任务完成后才确认（防止worker崩溃丢失任务）
    task_reject_on_worker_lost=True,  # worker丢失时重新投递任务
    
    # 任务超时配置
    task_time_limit=3600,  # 硬超时1小时
    task_soft_time_limit=3300,  # 软超时55分钟（提前警告）
    
    # 结果存储配置
    result_expires=86400,  # 结果保留24小时
    result_backend_max_sleep_between_retries_ms=10000,
    
    # 重试配置
    task_default_retry_delay=60,  # 默认重试间隔60秒
    task_max_retries=3,  # 最大重试3次
    
    # 队列配置
    task_default_queue='default',
    task_routes={
        'tasks.run_training_task': {'queue': 'training'},
        'tasks.run_ml_training_task': {'queue': 'training'},
    },
    
    # Worker 并发配置（根据CPU核数调整）
    worker_concurrency=2,  # 4核CPU，用2个并发
    
    # 监控配置
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# 任务信号处理
@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **extras):
    """任务开始前的处理"""
    logger.info(f"任务开始: {task.name}[{task_id}]")

@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, state, **extras):
    """任务完成后的处理"""
    logger.info(f"任务完成: {task.name}[{task_id}] 状态: {state}")

@task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, einfo, **extras):
    """任务失败的处理"""
    logger.error(f"任务失败: {task_id} 错误: {exception}")

# 启动时打印配置
logger.info("Celery 配置加载完成")
logger.info(f"Broker: {celery_app.conf.broker_url}")
logger.info(f"Backend: {celery_app.conf.result_backend}")
logger.info(f"Worker并发: {celery_app.conf.worker_concurrency}")
