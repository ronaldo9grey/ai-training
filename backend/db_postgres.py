# -*- coding: utf-8 -*-
"""
PostgreSQL 数据库连接管理
支持连接池、读写分离
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'ai_training'),
    'user': os.getenv('DB_USER', 'ai_training'),
    'password': os.getenv('DB_PASSWORD', 'ai_training_2024')
}

# 全局连接池
_connection_pool: Optional[ThreadedConnectionPool] = None


def init_connection_pool(min_conn: int = 2, max_conn: int = 10):
    """初始化连接池"""
    global _connection_pool
    
    if _connection_pool is not None:
        return
    
    try:
        _connection_pool = ThreadedConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        logger.info(f"PostgreSQL连接池初始化成功: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    except Exception as e:
        logger.error(f"连接池初始化失败: {e}")
        raise


@contextmanager
def get_db_connection():
    """获取数据库连接（上下文管理器）"""
    global _connection_pool
    
    if _connection_pool is None:
        init_connection_pool()
    
    conn = None
    try:
        conn = _connection_pool.getconn()
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            _connection_pool.putconn(conn)


@contextmanager
def get_db_cursor(commit: bool = False):
    """获取数据库游标"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            cursor.close()


def execute_query(sql: str, params: tuple = None, fetch: bool = True) -> List[Dict]:
    """执行查询语句"""
    with get_db_cursor(commit=False) as cursor:
        cursor.execute(sql, params)
        if fetch:
            return cursor.fetchall()
        return []


def execute_update(sql: str, params: tuple = None) -> int:
    """执行更新语句（INSERT/UPDATE/DELETE）"""
    with get_db_cursor(commit=True) as cursor:
        cursor.execute(sql, params)
        return cursor.rowcount


def execute_many(sql: str, params_list: List[tuple]) -> int:
    """批量执行"""
    with get_db_cursor(commit=True) as cursor:
        cursor.executemany(sql, params_list)
        return cursor.rowcount


# 初始化数据库表结构
def init_database_tables():
    """创建PostgreSQL表结构"""
    
    tables_sql = """
    -- 项目表
    CREATE TABLE IF NOT EXISTS projects (
        id VARCHAR(64) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        task_type VARCHAR(50) DEFAULT 'text_classification',
        status VARCHAR(50) DEFAULT 'created',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        config JSONB
    );
    
    -- 数据集表
    CREATE TABLE IF NOT EXISTS datasets (
        id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) REFERENCES projects(id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL,
        file_path TEXT,
        file_type VARCHAR(50),
        total_samples INTEGER DEFAULT 0,
        labels JSONB,
        status VARCHAR(50) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        meta_info JSONB
    );
    
    -- 训练任务表
    CREATE TABLE IF NOT EXISTS training_jobs (
        id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) REFERENCES projects(id) ON DELETE CASCADE,
        dataset_id VARCHAR(64) REFERENCES datasets(id) ON DELETE SET NULL,
        model_name VARCHAR(255),
        status VARCHAR(50) DEFAULT 'pending',
        progress INTEGER DEFAULT 0,
        current_epoch INTEGER DEFAULT 0,
        total_epochs INTEGER,
        current_loss FLOAT,
        best_accuracy FLOAT,
        best_val_loss FLOAT,
        learning_rate FLOAT,
        log_path TEXT,
        model_path TEXT,
        config JSONB,
        eval_report JSONB,
        training_metrics JSONB,
        early_stopped BOOLEAN DEFAULT FALSE,
        stop_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP
    );
    
    -- 训练指标历史表
    CREATE TABLE IF NOT EXISTS training_metrics (
        id SERIAL PRIMARY KEY,
        job_id VARCHAR(64) REFERENCES training_jobs(id) ON DELETE CASCADE,
        epoch FLOAT,
        step INTEGER,
        train_loss FLOAT,
        val_loss FLOAT,
        train_accuracy FLOAT,
        val_accuracy FLOAT,
        learning_rate FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- 模型检查点表
    CREATE TABLE IF NOT EXISTS model_checkpoints (
        id VARCHAR(64) PRIMARY KEY,
        job_id VARCHAR(64) REFERENCES training_jobs(id) ON DELETE CASCADE,
        epoch INTEGER,
        step INTEGER,
        metric_name VARCHAR(100),
        metric_value FLOAT,
        file_path TEXT,
        is_best BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- 定时任务表
    CREATE TABLE IF NOT EXISTS training_schedules (
        id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) REFERENCES projects(id) ON DELETE CASCADE,
        name VARCHAR(255),
        dataset_id VARCHAR(64),
        config JSONB,
        cron_expression VARCHAR(100),
        is_active BOOLEAN DEFAULT TRUE,
        last_run_at TIMESTAMP,
        next_run_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- 飞书通知配置表
    CREATE TABLE IF NOT EXISTS feishu_notifications (
        id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) REFERENCES projects(id) ON DELETE CASCADE,
        webhook_url TEXT NOT NULL,
        notify_on_success BOOLEAN DEFAULT TRUE,
        notify_on_failure BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- 在线学习任务表
    CREATE TABLE IF NOT EXISTS online_learning_tasks (
        id VARCHAR(64) PRIMARY KEY,
        project_id VARCHAR(64) REFERENCES projects(id) ON DELETE CASCADE,
        base_job_id VARCHAR(64) REFERENCES training_jobs(id) ON DELETE SET NULL,
        learning_type VARCHAR(50),
        strategy JSONB,
        status VARCHAR(50) DEFAULT 'pending',
        new_samples_count INTEGER DEFAULT 0,
        accuracy_before FLOAT,
        accuracy_after FLOAT,
        model_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );
    
    -- 创建索引优化查询
    CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at);
    CREATE INDEX IF NOT EXISTS idx_datasets_project_id ON datasets(project_id);
    CREATE INDEX IF NOT EXISTS idx_training_jobs_project_id ON training_jobs(project_id);
    CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
    CREATE INDEX IF NOT EXISTS idx_training_metrics_job_id ON training_metrics(job_id);
    CREATE INDEX IF NOT EXISTS idx_training_metrics_created_at ON training_metrics(created_at);
    """
    
    with get_db_cursor(commit=True) as cursor:
        cursor.execute(tables_sql)
    
    logger.info("PostgreSQL表结构初始化完成")


# 测试连接
def test_connection() -> bool:
    """测试数据库连接"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()
            cursor.close()
            logger.info(f"PostgreSQL连接成功: {version[0]}")
            return True
    except Exception as e:
        logger.error(f"PostgreSQL连接失败: {e}")
        return False


if __name__ == "__main__":
    # 测试
    if test_connection():
        init_database_tables()
    else:
        print("连接失败，请检查配置")
