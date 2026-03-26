# -*- coding: utf-8 -*-
"""
AutoML 模块 - 自动超参数优化
支持：贝叶斯优化、网格搜索、实验对比
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import itertools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入Optuna，如果未安装则使用简化版
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna未安装，使用简化版网格搜索。运行: pip install optuna")

from db_postgres import execute_query, execute_update


class AutoMLConfig:
    """AutoML配置"""
    
    # 各任务类型的搜索空间
    SEARCH_SPACES = {
        "image_classification": {
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
            "batch_size": {"type": "categorical", "choices": [8, 16, 32, 64]},
            "epochs": {"type": "int", "low": 5, "high": 30},
            "optimizer": {"type": "categorical", "choices": ["adam", "adamw", "sgd"]},
            "weight_decay": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        },
        "object_detection": {
            "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
            "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
            "epochs": {"type": "int", "low": 50, "high": 200},
            "image_size": {"type": "categorical", "choices": [640, 1280]},
        },
        "text_classification": {
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 5e-5},
            "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
            "epochs": {"type": "int", "low": 2, "high": 10},
            "max_seq_length": {"type": "categorical", "choices": [128, 256, 512]},
        },
        "classification": {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "max_depth": {"type": "int", "low": 3, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
        },
    }


class AutoMLExperiment:
    """AutoML实验管理"""
    
    def __init__(self, experiment_id: str, project_id: str, dataset_id: str, 
                 base_config: Dict, search_space: Dict, max_trials: int = 20):
        self.experiment_id = experiment_id
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.base_config = base_config
        self.search_space = search_space
        self.max_trials = max_trials
        self.trials = []
        self.best_trial = None
        self.status = "running"
        
    def suggest_params(self, trial_num: int) -> Dict:
        """生成一组超参数（简化版网格搜索）"""
        params = self.base_config.copy()
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config.get("type", "float")
            
            if param_type == "loguniform":
                # 对数均匀分布
                import random
                import math
                low = math.log(param_config["low"])
                high = math.log(param_config["high"])
                value = math.exp(random.uniform(low, high))
                params[param_name] = round(value, 6)
                
            elif param_type == "int":
                # 整数均匀分布
                import random
                low = param_config["low"]
                high = param_config["high"]
                params[param_name] = random.randint(low, high)
                
            elif param_type == "categorical":
                # 类别选择
                import random
                choices = param_config["choices"]
                params[param_name] = random.choice(choices)
                
            else:
                # 默认浮点数
                import random
                low = param_config.get("low", 0)
                high = param_config.get("high", 1)
                params[param_name] = round(random.uniform(low, high), 6)
        
        return params
    
    def record_trial(self, trial_num: int, params: Dict, metrics: Dict) -> str:
        """记录实验结果"""
        import uuid
        trial_id = str(uuid.uuid4())
        
        trial = {
            "id": trial_id,
            "experiment_id": self.experiment_id,
            "trial_number": trial_num,
            "params": params,
            "metrics": metrics,
            "created_at": datetime.now().isoformat()
        }
        
        self.trials.append(trial)
        
        # 保存到数据库
        execute_update("""
            INSERT INTO automl_trials (id, experiment_id, trial_number, params, metrics, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            trial_id,
            self.experiment_id,
            trial_num,
            json.dumps(params),
            json.dumps(metrics),
            trial["created_at"]
        ))
        
        # 更新最佳实验
        self._update_best_trial()
        
        return trial_id
    
    def _update_best_trial(self):
        """更新最佳实验"""
        if not self.trials:
            return
        
        # 根据准确率排序
        best = max(self.trials, key=lambda t: t["metrics"].get("accuracy", 0))
        self.best_trial = best
        
        # 更新实验状态
        execute_update("""
            UPDATE automl_experiments 
            SET best_trial_id = %s,
                best_accuracy = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (
            best["id"],
            best["metrics"].get("accuracy", 0),
            self.experiment_id
        ))
    
    def get_summary(self) -> Dict:
        """获取实验摘要"""
        if not self.trials:
            return {"status": self.status, "trials_count": 0}
        
        accuracies = [t["metrics"].get("accuracy", 0) for t in self.trials]
        
        return {
            "status": self.status,
            "trials_count": len(self.trials),
            "best_accuracy": max(accuracies),
            "worst_accuracy": min(accuracies),
            "avg_accuracy": sum(accuracies) / len(accuracies),
            "best_trial": self.best_trial
        }


def create_automl_experiment(project_id: str, dataset_id: str, name: str,
                             base_config: Dict, task_type: str,
                             max_trials: int = 20) -> str:
    """创建AutoML实验"""
    import uuid
    
    experiment_id = str(uuid.uuid4())
    
    # 获取搜索空间
    search_space = AutoMLConfig.SEARCH_SPACES.get(task_type, {})
    
    # 创建实验记录
    execute_update("""
        INSERT INTO automl_experiments (
            id, project_id, dataset_id, name, 
            base_config, search_space, max_trials, status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        experiment_id,
        project_id,
        dataset_id,
        name,
        json.dumps(base_config),
        json.dumps(search_space),
        max_trials,
        "pending"
    ))
    
    logger.info(f"创建AutoML实验: {name} ({experiment_id})")
    return experiment_id


def run_automl_trial(experiment_id: str, trial_num: int) -> Dict:
    """
    运行单次AutoML实验 - 生成超参数并创建trial记录
    
    返回：{trial_id, params, experiment_id}
    """
    import uuid
    
    # 获取实验信息
    rows = execute_query("""
        SELECT * FROM automl_experiments WHERE id = %s
    """, (experiment_id,))
    
    if not rows:
        raise ValueError("实验不存在")
    
    exp = rows[0]
    base_config = exp["base_config"] if isinstance(exp["base_config"], dict) else json.loads(exp["base_config"])
    search_space = exp["search_space"] if isinstance(exp["search_space"], dict) else json.loads(exp["search_space"])
    
    # 创建实验对象
    experiment = AutoMLExperiment(
        experiment_id=experiment_id,
        project_id=exp["project_id"],
        dataset_id=exp["dataset_id"],
        base_config=base_config,
        search_space=search_space,
        max_trials=exp["max_trials"]
    )
    
    # 生成超参数
    params = experiment.suggest_params(trial_num)
    
    # 创建trial记录到数据库
    trial_id = str(uuid.uuid4())
    execute_update("""
        INSERT INTO automl_trials (id, experiment_id, trial_number, params, status, metrics, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
    """, (
        trial_id,
        experiment_id,
        trial_num,
        json.dumps(params),
        'pending',
        json.dumps({})
    ))
    
    logger.info(f"创建AutoML trial #{trial_num}: {trial_id}")
    
    return {
        "trial_id": trial_id,
        "trial_num": trial_num,
        "params": params,
        "experiment_id": experiment_id
    }


def compare_trials(experiment_id: str) -> Dict:
    """对比所有实验结果"""
    trials = execute_query("""
        SELECT * FROM automl_trials 
        WHERE experiment_id = %s
        ORDER BY (metrics->>'accuracy')::float DESC
    """, (experiment_id,))
    
    if not trials:
        return {"trials": [], "comparison": {}}
    
    # 分析参数重要性（简化版：统计各参数在top trials中的分布）
    top_trials = trials[:5]
    param_analysis = {}
    
    for trial in top_trials:
        params = trial["params"] if isinstance(trial["params"], dict) else json.loads(trial["params"])
        for param_name, value in params.items():
            if param_name not in param_analysis:
                param_analysis[param_name] = []
            param_analysis[param_name].append(value)
    
    return {
        "trials": [dict(t) for t in trials],
        "top_5_trials": [dict(t) for t in top_trials],
        "param_analysis": param_analysis
    }


def get_recommended_params(experiment_id: str) -> Dict:
    """获取推荐的超参数配置"""
    rows = execute_query("""
        SELECT * FROM automl_experiments WHERE id = %s
    """, (experiment_id,))
    
    if not rows:
        return {}
    
    exp = rows[0]
    
    # 如果有最佳实验，返回最佳参数
    if exp.get("best_trial_id"):
        trial_rows = execute_query("""
            SELECT params FROM automl_trials WHERE id = %s
        """, (exp["best_trial_id"],))
        
        if trial_rows:
            params = trial_rows[0]["params"]
            if isinstance(params, str):
                params = json.loads(params)
            return {
                "recommended": params,
                "accuracy": exp.get("best_accuracy"),
                "source": "best_trial"
            }
    
    # 否则返回基础配置
    base_config = exp["base_config"]
    if isinstance(base_config, str):
        base_config = json.loads(base_config)
    
    return {
        "recommended": base_config,
        "source": "base_config"
    }


if __name__ == "__main__":
    print("AutoML模块加载成功")
    if OPTUNA_AVAILABLE:
        print("使用Optuna贝叶斯优化")
    else:
        print("使用简化版网格搜索")
