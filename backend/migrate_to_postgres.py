#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite → PostgreSQL 数据迁移脚本
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 路径配置
SQLITE_DB_PATH = Path("/var/www/ai-training/training.db")


def migrate_data():
    """执行数据迁移"""
    from db_postgres import (
        init_connection_pool, init_database_tables, 
        execute_update, execute_many, test_connection
    )
    
    # 1. 初始化PostgreSQL
    logger.info("正在初始化PostgreSQL...")
    init_connection_pool()
    init_database_tables()
    
    if not test_connection():
        logger.error("PostgreSQL连接失败")
        return False
    
    # 2. 连接SQLite
    logger.info(f"正在连接SQLite: {SQLITE_DB_PATH}")
    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()
    
    try:
        # 3. 迁移项目表
        logger.info("迁移项目数据...")
        sqlite_cursor.execute("SELECT * FROM projects")
        projects = sqlite_cursor.fetchall()
        
        if projects:
            project_sql = """
                INSERT INTO projects (id, name, description, task_type, status, created_at, updated_at, config)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """
            project_data = []
            for p in projects:
                project_data.append((
                    p['id'],
                    p['name'],
                    p['description'],
                    p['task_type'],
                    p['status'],
                    p['created_at'],
                    p['updated_at'],
                    json.dumps(json.loads(p['config'])) if p['config'] else None
                ))
            execute_many(project_sql, project_data)
            logger.info(f"  迁移了 {len(project_data)} 个项目")
        
        # 4. 迁移数据集表
        logger.info("迁移数据集...")
        sqlite_cursor.execute("SELECT * FROM datasets")
        datasets = sqlite_cursor.fetchall()
        
        if datasets:
            dataset_sql = """
                INSERT INTO datasets (id, project_id, name, file_path, file_type, total_samples, labels, status, created_at, meta_info)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """
            dataset_data = []
            for d in datasets:
                dataset_data.append((
                    d['id'],
                    d['project_id'],
                    d['name'],
                    d['file_path'],
                    d['file_type'],
                    d['total_samples'],
                    json.dumps(json.loads(d['labels'])) if d['labels'] else None,
                    d['status'],
                    d['created_at'],
                    json.dumps(json.loads(d['meta_info'])) if 'meta_info' in d.keys() and d['meta_info'] else None
                ))
            execute_many(dataset_sql, dataset_data)
            logger.info(f"  迁移了 {len(dataset_data)} 个数据集")
        
        # 5. 迁移训练任务表
        logger.info("迁移训练任务...")
        sqlite_cursor.execute("SELECT * FROM training_jobs")
        jobs = sqlite_cursor.fetchall()
        
        if jobs:
            job_sql = """
                INSERT INTO training_jobs (
                    id, project_id, dataset_id, model_name, status, progress,
                    current_epoch, total_epochs, current_loss, best_accuracy, best_val_loss,
                    learning_rate, log_path, model_path, config, eval_report,
                    training_metrics, early_stopped, stop_reason,
                    created_at, started_at, completed_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """
            job_data = []
            for j in jobs:
                job_data.append((
                    j['id'],
                    j['project_id'],
                    j['dataset_id'],
                    j['model_name'],
                    j['status'],
                    j['progress'],
                    j['current_epoch'],
                    j['total_epochs'],
                    j['current_loss'],
                    j['best_accuracy'],
                    j['best_val_loss'],
                    j['learning_rate'],
                    j['log_path'],
                    j['model_path'],
                    json.dumps(json.loads(j['config'])) if j['config'] else None,
                    json.dumps(json.loads(j['eval_report'])) if 'eval_report' in j.keys() and j['eval_report'] else None,
                    json.dumps(json.loads(j['training_metrics'])) if 'training_metrics' in j.keys() and j['training_metrics'] else None,
                    bool(j['early_stopped']) if 'early_stopped' in j.keys() else False,
                    j['stop_reason'],
                    j['created_at'],
                    j['started_at'],
                    j['completed_at']
                ))
            execute_many(job_sql, job_data)
            logger.info(f"  迁移了 {len(job_data)} 个训练任务")
        
        # 6. 迁移训练指标
        logger.info("迁移训练指标...")
        sqlite_cursor.execute("SELECT * FROM training_metrics")
        metrics = sqlite_cursor.fetchall()
        
        if metrics:
            metric_sql = """
                INSERT INTO training_metrics (job_id, epoch, step, train_loss, val_loss, train_accuracy, val_accuracy, learning_rate, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            metric_data = []
            for m in metrics:
                metric_data.append((
                    m['job_id'],
                    m['epoch'],
                    m['step'],
                    m['train_loss'],
                    m['val_loss'],
                    m['train_accuracy'],
                    m['val_accuracy'],
                    m['learning_rate'],
                    m['created_at']
                ))
            execute_many(metric_sql, metric_data)
            logger.info(f"  迁移了 {len(metric_data)} 条训练指标")
        
        logger.info("✅ 数据迁移完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 迁移失败: {e}")
        return False
    finally:
        sqlite_conn.close()


def verify_migration():
    """验证迁移结果"""
    from db_postgres import execute_query
    
    logger.info("\n验证迁移结果:")
    
    # 统计各表数据
    tables = ['projects', 'datasets', 'training_jobs', 'training_metrics']
    for table in tables:
        count = execute_query(f"SELECT COUNT(*) as cnt FROM {table}")[0]['cnt']
        logger.info(f"  {table}: {count} 条记录")
    
    # 对比SQLite和PostgreSQL
    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
    sqlite_cursor = sqlite_conn.cursor()
    
    logger.info("\n对比验证:")
    for table in tables:
        sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table}")
        sqlite_count = sqlite_cursor.fetchone()[0]
        pg_count = execute_query(f"SELECT COUNT(*) as cnt FROM {table}")[0]['cnt']
        
        status = "✅" if sqlite_count == pg_count else "❌"
        logger.info(f"  {table}: SQLite={sqlite_count}, PostgreSQL={pg_count} {status}")
    
    sqlite_conn.close()


if __name__ == "__main__":
    print("="*60)
    print("SQLite → PostgreSQL 数据迁移")
    print("="*60)
    
    if migrate_data():
        verify_migration()
        print("\n✅ 迁移成功！请检查数据完整性。")
    else:
        print("\n❌ 迁移失败，请检查错误日志。")
