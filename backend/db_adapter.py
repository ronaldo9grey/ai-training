# -*- coding: utf-8 -*-
"""
数据库适配层 - 兼容SQLite和PostgreSQL
"""

import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# 导入PostgreSQL模块
try:
    from db_postgres import execute_query as pg_execute_query, execute_update as pg_execute_update
    POSTGRES_AVAILABLE = True
except:
    POSTGRES_AVAILABLE = False
    logging.warning("PostgreSQL模块未加载")

logger = logging.getLogger(__name__)

# 使用PostgreSQL
USE_POSTGRES = True


def get_db_connection():
    """获取数据库连接（适配层）"""
    if USE_POSTGRES and POSTGRES_AVAILABLE:
        from db_postgres import get_db_connection as pg_conn
        return pg_conn()
    else:
        import sqlite3
        from pathlib import Path
        DB_PATH = Path("/var/www/ai-training/training.db")
        return sqlite3.connect(DB_PATH)


def execute_query(sql: str, params: tuple = None) -> List[Dict]:
    """执行查询（适配层）"""
    if USE_POSTGRES and POSTGRES_AVAILABLE:
        # 转换SQL语法
        sql = convert_sql(sql)
        return pg_execute_query(sql, params)
    else:
        import sqlite3
        from pathlib import Path
        DB_PATH = Path("/var/www/ai-training/training.db")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql, params)
        result = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return result


def execute_update(sql: str, params: tuple = None) -> int:
    """执行更新（适配层）"""
    if USE_POSTGRES and POSTGRES_AVAILABLE:
        # 转换SQL语法
        sql = convert_sql(sql)
        return pg_execute_update(sql, params)
    else:
        import sqlite3
        from pathlib import Path
        DB_PATH = Path("/var/www/ai-training/training.db")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
        rowcount = cursor.rowcount
        conn.close()
        return rowcount


def convert_sql(sql: str) -> str:
    """转换SQL语法从SQLite到PostgreSQL"""
    # 替换占位符 ? 为 %s
    sql = sql.replace('?', '%s')
    
    # 替换日期时间函数
    sql = sql.replace('CURRENT_TIMESTAMP', 'NOW()')
    
    # 替换JSON操作
    # SQLite: json_extract(column, '$.key')
    # PostgreSQL: column->>'key' (对于text) 或 column->'key' (对于json)
    
    return sql


# 简化的连接上下文管理器（兼容旧代码）
class MockConnection:
    """模拟SQLite连接对象，兼容旧代码"""
    def __init__(self):
        self._cursor = None
    
    def cursor(self):
        from db_postgres import get_db_connection
        self._conn = get_db_connection().__enter__()
        self._cursor = self._conn.cursor()
        return self._cursor
    
    def commit(self):
        if hasattr(self, '_conn'):
            self._conn.commit()
    
    def close(self):
        if hasattr(self, '_conn'):
            if self._cursor:
                self._cursor.close()
            from db_postgres import get_db_connection
            get_db_connection().__exit__(None, None, None)


def sqlite3_connect(db_path):
    """替代sqlite3.connect的函数"""
    if USE_POSTGRES and POSTGRES_AVAILABLE:
        return MockConnection()
    else:
        import sqlite3
        return sqlite3.connect(db_path)
