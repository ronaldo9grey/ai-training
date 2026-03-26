#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序分析 + 在线学习 功能演示
"""

import json
import sys
sys.path.insert(0, '/var/www/ai-training/backend')

print("="*70)
print("🎬 AI训练智能体 - 时序分析 & 在线学习 Demo")
print("="*70)

# ============ 第一部分：时序分析演示 ============
print("\n" + "="*70)
print("📊 第一部分：设备传感器时序分析")
print("="*70)

from time_series_analyzer import analyze_equipment_trends, TimeSeriesAnalyzer
import pandas as pd

print("\n【场景】分析过去7天的设备传感器数据，预测未来24小时趋势...")
print("-"*70)

# 加载数据
df = pd.read_excel('/tmp/equipment_failure_prediction.xlsx')
print(f"✅ 加载数据完成：{len(df)}条记录")
print(f"   时间范围：{df['timestamp'].min()} 至 {df['timestamp'].max()}")

# 执行时序分析
result = analyze_equipment_trends('/tmp/equipment_failure_prediction.xlsx', forecast_hours=24)

print("\n" + "🔍"*35)
print("📈 趋势预测结果")
print("🔍"*35)

for metric, trend in result['trend_analysis'].items():
    print(f"\n📊 {metric.upper()}")
    print(f"   当前值: {trend['current_value']}")
    print(f"   平均值: {trend['mean_value']}")
    print(f"   趋势方向: {trend['trend_direction']}")
    print(f"   风险等级: {trend['risk_level'].upper()}")
    
    if trend['will_exceed_threshold']:
        print(f"   ⚠️  警告: 预计将在 {trend['exceed_time']} 超出正常范围!")
    
    # 显示未来预测
    print(f"   未来12小时预测:")
    for pred in trend['future_predictions'][:6]:
        print(f"      {pred['timestamp']}: {pred['predicted_value']}")

print("\n" + "🔍"*35)
print("🔎 异常模式检测")
print("🔍"*35)

patterns = result['anomaly_patterns']
print(f"\n发现 {patterns['total_patterns']} 个异常模式")
print(f"总结: {patterns['summary']}")

for i, pattern in enumerate(patterns['patterns'][:5], 1):
    severity_emoji = "🔴" if pattern['severity'] == 'high' else ("🟡" if pattern['severity'] == 'medium' else "🔵")
    print(f"\n{severity_emoji} {pattern['pattern_type']}")
    print(f"   描述: {pattern['description']}")
    print(f"   建议: {pattern['recommendation']}")

# RUL预测演示
print("\n" + "🔍"*35)
print("⏱️  剩余使用寿命(RUL)预测")
print("🔍"*35)

analyzer = TimeSeriesAnalyzer()

# 假设温度超过95度为故障阈值
rul_result = analyzer.predict_rul(df, 'timestamp', 'temperature', threshold=95)

print(f"\n当前温度: {rul_result['current_value']}°C")
print(f"故障阈值: {rul_result['threshold']}°C")
print(f"趋势斜率: {rul_result['trend_slope']:.4f} (每小时变化)")

if rul_result['status'] == 'critical':
    print(f"🔴 紧急: {rul_result['message']}")
elif rul_result['status'] == 'warning':
    print(f"🟡 警告: {rul_result['message']}")
else:
    print(f"🟢 正常: {rul_result['message']}")

if rul_result['rul_hours']:
    print(f"\n预计剩余寿命: {rul_result['rul_hours']}小时 ({rul_result['rul_days']:.1f}天)")
    print(f"预计故障时间: {rul_result['estimated_failure_time']}")

# ============ 第二部分：在线学习演示 ============
print("\n\n" + "="*70)
print("🔄 第二部分：模型在线学习")
print("="*70)

from online_learning import OnlineLearningManager
import shutil
from pathlib import Path

manager = OnlineLearningManager()

print("\n【场景】今天收集了50条新设备数据，用增量学习更新模型...")
print("-"*70)

# 准备新数据（模拟）
print("\n1️⃣  准备新数据")
new_data = df.sample(n=50, random_state=42)
new_data_path = '/tmp/equipment_new_data.xlsx'
new_data.to_excel(new_data_path, index=False)
print(f"   ✅ 生成50条新样本: {new_data_path}")

# 先创建一个基础模型
print("\n2️⃣  检查基础模型")
base_model_path = '/var/www/ai-training/models/demo-equipment-failure/final-model'
if Path(base_model_path).exists():
    print(f"   ✅ 找到基础模型: {base_model_path}")
    
    # 查看原模型信息
    import joblib
    model_package = joblib.load(Path(base_model_path) / 'model.joblib')
    print(f"   📊 原模型树数量: {model_package['model'].n_estimators}")
else:
    print(f"   ⚠️  基础模型不存在，跳过演示")
    sys.exit(0)

# 创建在线学习任务
print("\n3️⃣  创建在线学习任务")
task_id = manager.create_learning_task(
    project_id='demo-equipment-failure',
    job_id='final-model',
    learning_type='incremental',
    strategy={'description': '每日增量更新'}
)
print(f"   ✅ 任务ID: {task_id}")

# 执行增量学习
print("\n4️⃣  执行增量学习")
print("   🔄 正在训练... (使用warm_start增量更新)")

try:
    learn_result = manager.incremental_learn(task_id, new_data_path)
    
    print(f"\n   ✅ 增量学习完成!")
    print(f"   📊 新增样本数: {learn_result['new_samples']}")
    print(f"   📈 模型准确率: {learn_result['accuracy']*100:.2f}%")
    print(f"   💾 新模型路径: {learn_result['model_path']}")
    print(f"   🆔 新任务ID: {learn_result['new_job_id']}")
    
    # 查看新模型信息
    new_model_package = joblib.load(Path(learn_result['model_path']) / 'model.joblib')
    print(f"   🌲 新模型树数量: {new_model_package['model'].n_estimators}")
    print(f"   📥 累计学习样本: {new_model_package.get('incremental_samples', 0)}")
    
except Exception as e:
    print(f"   ❌ 学习失败: {e}")

# 查看学习历史
print("\n5️⃣  查看学习历史")
history = manager.get_learning_history('demo-equipment-failure', 'final-model')
print(f"   📚 历史学习次数: {len(history)}")

for i, h in enumerate(history[:3], 1):
    print(f"\n   记录{i}:")
    print(f"      类型: {h['learning_type']}")
    print(f"      状态: {h['status']}")
    print(f"      新样本: {h['new_samples']}")
    if h['accuracy_after']:
        print(f"      准确率: {h['accuracy_after']*100:.2f}%")

# 设置自动学习
print("\n6️⃣  设置自动学习")
config_id = manager.setup_auto_learning(
    project_id='demo-equipment-failure',
    job_id='final-model',
    schedule='daily',
    min_samples=100,
    accuracy_threshold=0.05
)
print(f"   ✅ 自动学习配置ID: {config_id}")
print(f"   📅 频率: 每天")
print(f"   📊 最小触发样本: 100")

# ============ 总结 ============
print("\n\n" + "="*70)
print("🎉 Demo演示完成！")
print("="*70)

print("""
📋 演示内容总结:

┌─────────────────────────────────────────────────────────────────┐
│ 📊 时序分析功能                                                  │
├─────────────────────────────────────────────────────────────────┤
│ ✅ 趋势预测：预测未来24小时各传感器走势                          │
│ ✅ 异常检测：发现渐变劣化/波动性增加/突变点等模式                │
│ ✅ RUL预测：估算设备剩余使用寿命                                 │
│ ✅ 相关性分析：找出影响故障的关键因素                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 🔄 在线学习功能                                                  │
├─────────────────────────────────────────────────────────────────┤
│ ✅ 增量学习：用50条新数据更新模型，保留已有知识                  │
│ ✅ 模型版本：自动创建新版本，支持历史追溯                        │
│ ✅ 效果评估：对比学习前后准确率变化                              │
│ ✅ 自动配置：设置每日自动学习，积累100样本触发                   │
└─────────────────────────────────────────────────────────────────┘

🎯 核心价值:
   • 从"单点预测"升级到"趋势预测 + 寿命预测"
   • 从"静态模型"升级到"持续进化"的自适应系统
   • 实现真正的"预测性维护"而不是"事后维修"

🔗 访问系统:
   https://你的域名/ai-training/
   
   导航: 项目 → 时序分析 / 在线学习
""")
