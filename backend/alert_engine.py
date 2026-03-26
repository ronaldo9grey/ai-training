# -*- coding: utf-8 -*-
"""
告警引擎模块
支持：规则管理、条件检查、告警触发、通知发送
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

from db_postgres import execute_query, execute_update
from tasks import send_feishu_notification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertRuleType(Enum):
    """告警规则类型"""
    TRAINING = "training"  # 训练相关
    INFERENCE = "inference"  # 推理相关
    DATA_DRIFT = "data_drift"  # 数据漂移
    RUL = "rul"  # 剩余寿命
    SYSTEM = "system"  # 系统资源


class AlertEngine:
    """告警引擎"""
    
    # 预置告警规则模板
    RULE_TEMPLATES = {
        "training_accuracy_low": {
            "name": "训练准确率过低",
            "description": "当模型训练准确率低于阈值时触发",
            "rule_type": "training",
            "condition_field": "best_accuracy",
            "condition_operator": "<",
            "condition_value": 0.8,
            "severity": "warning",
            "message_template": "⚠️ 训练准确率偏低\n模型: {model_name}\n准确率: {actual_value:.2%}\n阈值: {threshold_value:.2%}\n建议: 检查数据质量或增加训练轮数"
        },
        "training_failed": {
            "name": "训练任务失败",
            "description": "训练任务失败时触发",
            "rule_type": "training",
            "condition_field": "status",
            "condition_operator": "==",
            "condition_value": None,  # 字符串比较
            "severity": "error",
            "message_template": "❌ 训练任务失败\n模型: {model_name}\n错误: {error_message}\n请及时排查问题"
        },
        "rul_critical": {
            "name": "设备RUL严重不足",
            "description": "设备剩余使用寿命低于阈值",
            "rule_type": "rul",
            "condition_field": "rul_hours",
            "condition_operator": "<",
            "condition_value": 24,
            "severity": "critical",
            "message_template": "🔴 紧急：设备即将故障\n设备: {equipment_name}\n预计剩余时间: {actual_value:.1f}小时\n建议: 立即安排维护停机"
        },
        "rul_warning": {
            "name": "设备RUL预警",
            "description": "设备剩余使用寿命不足，提前规划维护",
            "rule_type": "rul",
            "condition_field": "rul_hours",
            "condition_operator": "<",
            "condition_value": 72,
            "severity": "warning",
            "message_template": "⚠️ 设备维护预警\n设备: {equipment_name}\n预计剩余时间: {actual_value:.1f}小时\n建议: 安排下次停机维护"
        },
        "data_drift_detected": {
            "name": "数据漂移检测",
            "description": "检测到输入数据分布发生变化",
            "rule_type": "data_drift",
            "condition_field": "drift_score",
            "condition_operator": ">",
            "condition_value": 0.3,
            "severity": "error",
            "message_template": "📊 数据漂移警报\n项目: {project_name}\n漂移分数: {actual_value:.3f}\n建议: 模型可能需要重新训练"
        },
        "system_resource_high": {
            "name": "系统资源使用率过高",
            "description": "CPU/内存/GPU使用率超过阈值",
            "rule_type": "system",
            "condition_field": "cpu_percent",
            "condition_operator": ">",
            "condition_value": 90,
            "severity": "warning",
            "message_template": "💻 系统资源告警\nCPU使用率: {actual_value}%\n建议: 检查是否有异常任务或扩容资源"
        }
    }
    
    @classmethod
    def create_rule_from_template(cls, project_id: str, template_key: str, 
                                   custom_values: Dict = None) -> str:
        """从模板创建告警规则"""
        import uuid
        
        if template_key not in cls.RULE_TEMPLATES:
            raise ValueError(f"未知模板: {template_key}")
        
        template = cls.RULE_TEMPLATES[template_key].copy()
        if custom_values:
            template.update(custom_values)
        
        rule_id = str(uuid.uuid4())
        
        execute_update("""
            INSERT INTO alert_rules (
                id, project_id, name, description, rule_type,
                condition_field, condition_operator, condition_value,
                severity, enabled, notify_channels
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            rule_id,
            project_id,
            template["name"],
            template["description"],
            template["rule_type"],
            template["condition_field"],
            template["condition_operator"],
            template.get("condition_value", 0),
            template["severity"],
            True,
            json.dumps(["feishu"])
        ))
        
        logger.info(f"从模板创建告警规则: {template['name']} ({rule_id})")
        return rule_id
    
    @staticmethod
    def create_rule(project_id: str, name: str, description: str,
                    rule_type: str, condition_field: str,
                    condition_operator: str, condition_value: float,
                    severity: str = "warning",
                    notify_channels: List[str] = None) -> str:
        """创建自定义告警规则"""
        import uuid
        
        rule_id = str(uuid.uuid4())
        
        execute_update("""
            INSERT INTO alert_rules (
                id, project_id, name, description, rule_type,
                condition_field, condition_operator, condition_value,
                severity, enabled, notify_channels
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            rule_id, project_id, name, description, rule_type,
            condition_field, condition_operator, condition_value,
            severity, True,
            json.dumps(notify_channels or ["feishu"])
        ))
        
        logger.info(f"创建告警规则: {name} ({rule_id})")
        return rule_id
    
    @staticmethod
    def check_rule(rule: Dict, context: Dict) -> Optional[Dict]:
        """
        检查规则条件是否满足
        
        Args:
            rule: 规则配置
            context: 当前上下文数据
        
        Returns:
            如果触发告警返回告警信息，否则None
        """
        field = rule["condition_field"]
        operator = rule["condition_operator"]
        threshold = rule["condition_value"]
        
        # 获取实际值
        actual_value = context.get(field)
        if actual_value is None:
            return None
        
        # 类型转换
        try:
            actual_value = float(actual_value)
            threshold = float(threshold)
        except (ValueError, TypeError):
            # 字符串比较
            actual_value = str(actual_value)
            threshold = str(threshold)
        
        # 条件判断
        triggered = False
        if operator == "<":
            triggered = actual_value < threshold
        elif operator == ">":
            triggered = actual_value > threshold
        elif operator == "<=":
            triggered = actual_value <= threshold
        elif operator == ">=":
            triggered = actual_value >= threshold
        elif operator == "==":
            triggered = actual_value == threshold
        elif operator == "!=":
            triggered = actual_value != threshold
        
        if triggered:
            return {
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "severity": rule["severity"],
                "actual_value": actual_value,
                "threshold_value": threshold,
                "message": rule.get("message_template", "告警触发").format(
                    **context,
                    actual_value=actual_value,
                    threshold_value=threshold
                )
            }
        
        return None
    
    @staticmethod
    def trigger_alert(rule_id: str, project_id: str, severity: str,
                     title: str, message: str, actual_value: float = None,
                     threshold_value: float = None, context: Dict = None) -> str:
        """触发告警"""
        import uuid
        
        alert_id = str(uuid.uuid4())
        
        # 检查冷却时间
        rows = execute_query("""
            SELECT last_triggered_at, cooldown_minutes 
            FROM alert_rules 
            WHERE id = %s
        """, (rule_id,))
        
        if rows:
            last_triggered = rows[0]["last_triggered_at"]
            cooldown = rows[0]["cooldown_minutes"]
            
            if last_triggered:
                cooldown_end = last_triggered + timedelta(minutes=cooldown)
                if datetime.now() < cooldown_end:
                    logger.info(f"告警规则 {rule_id} 在冷却期内，跳过")
                    return None
        
        # 记录告警历史
        execute_update("""
            INSERT INTO alert_history (
                id, rule_id, project_id, severity, title, message,
                actual_value, threshold_value, context
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            alert_id, rule_id, project_id, severity, title, message,
            actual_value, threshold_value,
            json.dumps(context) if context else None
        ))
        
        # 更新规则触发时间
        execute_update("""
            UPDATE alert_rules 
            SET last_triggered_at = CURRENT_TIMESTAMP,
                trigger_count = trigger_count + 1
            WHERE id = %s
        """, (rule_id,))
        
        logger.info(f"触发告警: {title} ({alert_id})")
        return alert_id
    
    @staticmethod
    def send_notification(project_id: str, alert_id: str, severity: str,
                         title: str, message: str):
        """发送告警通知"""
        # 获取项目信息和飞书配置
        rows = execute_query("""
            SELECT p.name as project_name, f.webhook_url
            FROM projects p
            LEFT JOIN feishu_notifications f ON p.id = f.project_id
            WHERE p.id = %s
        """, (project_id,))
        
        if not rows or not rows[0]["webhook_url"]:
            logger.warning(f"项目 {project_id} 未配置飞书通知")
            return
        
        project_name = rows[0]["project_name"]
        webhook_url = rows[0]["webhook_url"]
        
        # 构建飞书消息
        severity_colors = {
            "info": "blue",
            "warning": "orange",
            "error": "red",
            "critical": "red"
        }
        
        severity_emojis = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🔴"
        }
        
        feishu_message = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"{severity_emojis.get(severity, '⚠️')} {title}"
                    },
                    "template": severity_colors.get(severity, "orange")
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": message
                        }
                    },
                    {
                        "tag": "action",
                        "actions": [
                            {
                                "tag": "button",
                                "text": {"tag": "plain_text", "content": "查看详情"},
                                "type": "primary",
                                "url": f"https://your-domain/ai-training/projects/{project_id}/alerts"
                            }
                        ]
                    }
                ]
            }
        }
        
        # 发送飞书通知
        try:
            import urllib.request
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(feishu_message).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                logger.info(f"飞书告警通知发送成功: {alert_id}")
        except Exception as e:
            logger.error(f"飞书告警通知发送失败: {e}")
    
    @classmethod
    def check_training_alerts(cls, job_id: str, project_id: str):
        """检查训练相关告警"""
        # 获取任务信息
        rows = execute_query("""
            SELECT * FROM training_jobs WHERE id = %s
        """, (job_id,))
        
        if not rows:
            return
        
        job = rows[0]
        
        # 获取该项目的所有训练相关告警规则
        rules = execute_query("""
            SELECT * FROM alert_rules 
            WHERE project_id = %s 
            AND rule_type = 'training'
            AND enabled = TRUE
        """, (project_id,))
        
        context = {
            "job_id": job_id,
            "model_name": job["model_name"],
            "best_accuracy": job["best_accuracy"],
            "status": job["status"],
            "current_epoch": job["current_epoch"],
            "total_epochs": job["total_epochs"],
            "error_message": job.get("stop_reason", ""),
            "project_id": project_id
        }
        
        for rule in rules:
            result = cls.check_rule(rule, context)
            if result:
                alert_id = cls.trigger_alert(
                    rule_id=rule["id"],
                    project_id=project_id,
                    severity=result["severity"],
                    title=result["rule_name"],
                    message=result["message"],
                    actual_value=result["actual_value"],
                    threshold_value=result["threshold_value"],
                    context=context
                )
                
                if alert_id:
                    cls.send_notification(
                        project_id=project_id,
                        alert_id=alert_id,
                        severity=result["severity"],
                        title=result["rule_name"],
                        message=result["message"]
                    )
    
    @classmethod
    def check_rul_alerts(cls, project_id: str, equipment_data: Dict):
        """检查RUL相关告警"""
        rules = execute_query("""
            SELECT * FROM alert_rules 
            WHERE project_id = %s 
            AND rule_type = 'rul'
            AND enabled = TRUE
        """, (project_id,))
        
        for rule in rules:
            result = cls.check_rule(rule, equipment_data)
            if result:
                alert_id = cls.trigger_alert(
                    rule_id=rule["id"],
                    project_id=project_id,
                    severity=result["severity"],
                    title=result["rule_name"],
                    message=result["message"],
                    actual_value=result["actual_value"],
                    threshold_value=result["threshold_value"],
                    context=equipment_data
                )
                
                if alert_id:
                    cls.send_notification(
                        project_id=project_id,
                        alert_id=alert_id,
                        severity=result["severity"],
                        title=result["rule_name"],
                        message=result["message"]
                    )


def get_project_alert_rules(project_id: str) -> List[Dict]:
    """获取项目的所有告警规则"""
    rows = execute_query("""
        SELECT * FROM alert_rules 
        WHERE project_id = %s 
        ORDER BY created_at DESC
    """, (project_id,))
    
    return [dict(row) for row in rows]


def get_alert_history(project_id: str, status: str = None, limit: int = 50) -> List[Dict]:
    """获取告警历史"""
    if status:
        rows = execute_query("""
            SELECT * FROM alert_history 
            WHERE project_id = %s AND status = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (project_id, status, limit))
    else:
        rows = execute_query("""
            SELECT * FROM alert_history 
            WHERE project_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (project_id, limit))
    
    return [dict(row) for row in rows]


if __name__ == "__main__":
    print("告警引擎模块加载成功")
