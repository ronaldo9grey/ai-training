#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备故障预测 - 全链路模拟
包括: 数据上传 → 创建任务 → 训练模型 → 部署 → 测试
"""

import requests
import json
import sqlite3
from pathlib import Path
import shutil
import time
import sys

# API配置
API_BASE = "http://localhost:8004/api"
PROJECT_ID = "demo-equipment-failure"
DATASET_ID = "demo-dataset-001"
JOB_ID = "demo-job-001"

def print_step(step_num, title):
    print(f"\n{'='*60}")
    print(f"步骤 {step_num}: {title}")
    print('='*60)

def step1_create_project():
    """步骤1: 创建项目"""
    print_step(1, "创建项目 - 设备故障预测")
    
    url = f"{API_BASE}/projects"
    data = {
        "name": "设备故障预测-Demo",
        "description": "模拟设备传感器数据，预测故障状态",
        "task_type": "classification"
    }
    
    try:
        res = requests.post(url, json=data)
        if res.status_code == 200:
            result = res.json()
            print(f"✅ 项目创建成功: {result['project_id']}")
            return result['project_id']
    except Exception as e:
        print(f"⚠️  项目可能已存在: {e}")
    
    return None

def step2_upload_data():
    """步骤2: 上传数据"""
    print_step(2, "上传训练数据")
    
    # 复制数据到项目目录
    data_dir = Path(f"/var/www/ai-training/data/{PROJECT_ID}")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    source_file = "/tmp/equipment_failure_prediction.xlsx"
    target_file = data_dir / f"{DATASET_ID}.xlsx"
    shutil.copy(source_file, target_file)
    
    # 更新数据库
    conn = sqlite3.connect("/var/www/ai-training/training.db")
    cursor = conn.cursor()
    
    # 确保项目存在
    cursor.execute('INSERT OR IGNORE INTO projects (id, name, description, task_type) VALUES (?, ?, ?, ?)',
                   (PROJECT_ID, "设备故障预测-Demo", "模拟设备传感器数据", "classification"))
    
    # 添加数据集记录
    cursor.execute('''
        INSERT OR REPLACE INTO datasets (id, project_id, name, file_path, file_type, total_samples, labels, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (DATASET_ID, PROJECT_ID, "设备传感器数据", str(target_file), "xlsx", 1000, 
          '["正常", "故障"]', "uploaded"))
    
    conn.commit()
    conn.close()
    
    print(f"✅ 数据上传完成: {target_file}")
    print(f"   样本数: 1000 (正常904, 故障96)")
    return str(target_file)

def step3_train_model():
    """步骤3: 训练模型"""
    print_step(3, "训练故障预测模型")
    
    url = f"{API_BASE}/projects/{PROJECT_ID}/train"
    
    config = {
        "task_type": "classification",
        "model_type": "random_forest",
        "target_column": "failure",
        "feature_columns": ["temperature", "pressure", "vibration", "rpm", "voltage", "oil_pressure", "runtime_hours"],
        "n_estimators": 100
    }
    
    data = {
        "dataset_id": DATASET_ID,
        "config": json.dumps(config),
        "template": "balanced"
    }
    
    print("启动训练任务...")
    res = requests.post(url, data=data)
    
    if res.status_code == 200:
        result = res.json()
        job_id = result['job_id']
        print(f"✅ 训练任务启动: {job_id}")
        return job_id
    else:
        print(f"❌ 训练启动失败: {res.text}")
        return None

def wait_for_training(job_id, timeout=300):
    """等待训练完成"""
    print_step("3.1", "等待训练完成")
    
    url = f"{API_BASE}/projects/{PROJECT_ID}/jobs/{job_id}/status"
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        res = requests.get(url)
        if res.status_code == 200:
            status = res.json()
            print(f"   状态: {status['status']} | 进度: {status['progress']}% | 准确率: {status.get('best_accuracy', '-')} ")
            
            if status['status'] == 'completed':
                print(f"✅ 训练完成!")
                print(f"   最佳准确率: {status.get('best_accuracy', '-')}")
                print(f"   训练轮次: {status.get('current_epoch', '-')}")
                return True
            elif status['status'] == 'failed':
                print(f"❌ 训练失败!")
                return False
        
        time.sleep(2)
    
    print("⚠️  训练超时")
    return False

def step4_deploy_model(job_id):
    """步骤4: 部署模型"""
    print_step(4, "部署模型")
    
    url = f"{API_BASE}/projects/{PROJECT_ID}/models/{job_id}/deploy"
    
    res = requests.post(url, data={"deploy_type": "api"})
    
    if res.status_code == 200:
        result = res.json()
        print(f"✅ 模型部署成功!")
        print(f"   模型ID: {result['model_id']}")
        print(f"   任务类型: {result['task_type']}")
        return result['model_id']
    else:
        print(f"❌ 部署失败: {res.text}")
        return None

def step5_test_prediction(model_id):
    """步骤5: 测试预测效果"""
    print_step(5, "测试模型预测效果")
    
    # 测试用例
    test_cases = [
        {
            "name": "正常状态",
            "features": {
                "temperature": 70,
                "pressure": 12.5,
                "vibration": 1.0,
                "rpm": 3000,
                "voltage": 225,
                "oil_pressure": 3.0,
                "runtime_hours": 150
            },
            "expected": "正常"
        },
        {
            "name": "过热故障",
            "features": {
                "temperature": 98,
                "pressure": 12.0,
                "vibration": 1.2,
                "rpm": 2900,
                "voltage": 220,
                "oil_pressure": 2.8,
                "runtime_hours": 160
            },
            "expected": "故障"
        },
        {
            "name": "高压故障",
            "features": {
                "temperature": 75,
                "pressure": 20.5,
                "vibration": 1.1,
                "rpm": 2950,
                "voltage": 225,
                "oil_pressure": 3.2,
                "runtime_hours": 155
            },
            "expected": "故障"
        },
        {
            "name": "振动异常",
            "features": {
                "temperature": 72,
                "pressure": 13.0,
                "vibration": 3.5,
                "rpm": 2800,
                "voltage": 225,
                "oil_pressure": 2.5,
                "runtime_hours": 170
            },
            "expected": "故障"
        },
        {
            "name": "油压不足",
            "features": {
                "temperature": 68,
                "pressure": 11.5,
                "vibration": 0.8,
                "rpm": 3050,
                "voltage": 228,
                "oil_pressure": 0.8,
                "runtime_hours": 145
            },
            "expected": "故障"
        }
    ]
    
    print("\n执行预测测试:\n")
    print(f"{'测试项':<12} {'预测结果':<8} {'置信度':<10} {'预期':<8} {'是否正确'}")
    print("-" * 60)
    
    correct = 0
    for test in test_cases:
        # 构造特征数组
        features = [
            test['features']['temperature'],
            test['features']['pressure'],
            test['features']['vibration'],
            test['features']['rpm'],
            test['features']['voltage'],
            test['features']['oil_pressure'],
            test['features']['runtime_hours']
        ]
        
        # 调用预测API
        url = f"{API_BASE}/inference/{PROJECT_ID}/{model_id.split('/')[-1]}"
        # 使用features参数
        res = requests.post(url, data={"features": json.dumps(features)})
        
        if res.status_code == 200:
            result = res.json()
            pred = result['result'].get('prediction', '-')
            conf = result['result'].get('confidence', 0)
            
            is_correct = (pred == test['expected'])
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{test['name']:<12} {pred:<8} {conf*100:>6.1f}%     {test['expected']:<8} {status}")
        else:
            print(f"{test['name']:<12} {'ERROR':<8} {'-':<10} {test['expected']:<8} ❌")
    
    print("-" * 60)
    print(f"\n测试准确率: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
    
    return correct == len(test_cases)

def step6_batch_predict(model_id):
    """步骤6: 批量预测测试"""
    print_step(6, "批量预测测试")
    
    url = f"{API_BASE}/projects/{PROJECT_ID}/jobs/{model_id.split('/')[-1]}/batch-predict"
    
    # 上传测试文件
    test_file = "/tmp/equipment_test_data.xlsx"
    
    with open(test_file, 'rb') as f:
        files = {'file': ('test_data.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        res = requests.post(url, files=files)
    
    if res.status_code == 200:
        # 保存结果
        output_file = "/tmp/equipment_predictions.csv"
        with open(output_file, 'wb') as f:
            f.write(res.content)
        
        print(f"✅ 批量预测完成!")
        print(f"   结果文件: {output_file}")
        
        # 读取并显示统计
        import pandas as pd
        df = pd.read_csv(output_file)
        failure_count = df['prediction'].value_counts().get('1', 0) + df['prediction'].value_counts().get(1, 0)
        normal_count = len(df) - failure_count
        
        print(f"\n预测统计:")
        print(f"   总样本: {len(df)}")
        print(f"   预测正常: {normal_count}")
        print(f"   预测故障: {failure_count}")
        print(f"\n前5条预测结果:")
        print(df.head().to_string())
        
        return True
    else:
        print(f"❌ 批量预测失败: {res.text}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║        设备故障预测 - 全链路模拟演示                          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 步骤1: 创建项目
    project_id = step1_create_project()
    
    # 步骤2: 上传数据
    dataset_path = step2_upload_data()
    
    # 步骤3: 训练模型
    job_id = step3_train_model()
    if not job_id:
        print("❌ 训练启动失败，退出")
        sys.exit(1)
    
    # 等待训练完成
    if not wait_for_training(job_id):
        print("❌ 训练失败，退出")
        sys.exit(1)
    
    # 步骤4: 部署模型
    model_id = step4_deploy_model(job_id)
    if not model_id:
        print("❌ 部署失败，退出")
        sys.exit(1)
    
    # 步骤5: 测试预测
    step5_test_prediction(model_id)
    
    # 步骤6: 批量预测
    step6_batch_predict(model_id)
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  全链路模拟完成! ✅                            ║
╠══════════════════════════════════════════════════════════════╣
║  访问 https://你的域名/ai-training/ 查看项目详情             ║
╚══════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
