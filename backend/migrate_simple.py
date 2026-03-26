#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite → PostgreSQL 数据迁移脚本（简化版）
"""

import sqlite3
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SQLITE_DB_PATH = Path("/var/www/ai-training/training.db")


def safe_json_loads(data):
    """安全解析JSON"""
    if not data:
        return None
    try:
        return json.dumps(json.loads(data))
    except:
        return None


def safe_timestamp(ts):
    """处理时间戳，将CURRENT_TIMESTAMP等字符串转为None"""
    if not ts or ts == 'CURRENT_TIMESTAMP' or ts == 'None':
        return None
    return ts


def migrate_data():
    """执行数据迁移"""
    from db_postgres import init_connection_pool, init_database_tables, execute_update, test_connection
    
    logger.info("正在初始化PostgreSQL...")
    init_connection_pool()
    init_database_tables()
    
    if not test_connection():
        logger.error("PostgreSQL连接失败")
        return False
    
    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()
    
    try:
        # 1. 迁移项目
        logger.info("迁移项目数据...")
        sqlite_cursor.execute("SELECT * FROM projects")
        projects = sqlite_cursor.fetchall()
        
        for p in projects:
            try:
                execute_update("""
                    INSERT INTO projects (id, name, description, task_type, status, created_at, updated_at, config)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (p['id'], p['name'], p['description'], p['task_type'], p['status'], 
                     p['created_at'], p['updated_at'], safe_json_loads(p['config'])))
            except Exception as e:
                logger.warning(f"项目 {p['id']} 迁移失败: {e}")
        
        logger.info(f"  迁移了 {len(projects)} 个项目")
        
        # 2. 迁移数据集
        logger.info("迁移数据集...")
        sqlite_cursor.execute("SELECT * FROM datasets")
        datasets = sqlite_cursor.fetchall()
        
        for d in datasets:
            try:
                execute_update("""
                    INSERT INTO datasets (id, project_id, name, file_path, file_type, total_samples, labels, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (d['id'], d['project_id'], d['name'], d['file_path'], d['file_type'],
                     d['total_samples'], safe_json_loads(d['labels']), d['status'], d['created_at']))
            except Exception as e:
                logger.warning(f"数据集 {d['id']} 迁移失败: {e}")
        
        logger.info(f"  迁移了 {len(datasets)} 个数据集")
        
        # 3. 迁移训练任务
        logger.info("迁移训练任务...")
        sqlite_cursor.execute("SELECT * FROM training_jobs")
        jobs = sqlite_cursor.fetchall()
        
        for j in jobs:
            try:
                execute_update("""
                    INSERT INTO training_jobs (id, project_id, dataset_id, model_name, status, progress,
                        current_epoch, total_epochs, current_loss, best_accuracy, log_path, model_path, config,
                        created_at, started_at, completed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (j['id'], j['project_id'], j['dataset_id'], j['model_name'], j['status'],
                     j['progress'], j['current_epoch'], j['total_epochs'], j['current_loss'],
                     j['best_accuracy'], j['log_path'], j['model_path'],
                     safe_json_loads(j['config']), safe_timestamp(j['created_at']), 
                     safe_timestamp(j['started_at']), safe_timestamp(j['completed_at'])))
            except Exception as e:
                logger.warning(f"任务 {j['id']} 迁移失败: {e}")
        
        logger.info(f"  迁移了 {len(jobs)} 个训练任务")
        
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
    tables = ['projects', 'datasets', 'training_jobs']
    for table in tables:
        try:
            count = execute_query(f"SELECT COUNT(*) as cnt FROM {table}")[0]['cnt']
            logger.info(f"  {table}: {count} 条记录 ✅")
        except Exception as e:
            logger.error(f"  {table}: 查询失败 ❌ - {e}")


if __name__ == "__main__":
    print("="*60)
    print("SQLite → PostgreSQL 数据迁移")
    print("="*60)
    
    if migrate_data():
        verify_migration()
        print("\n✅ 迁移成功！")
    else:
        print("\n❌ 迁移失败")
