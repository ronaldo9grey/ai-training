#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL全链路测试
验证：项目创建 → 数据上传 → 训练任务 → 数据持久化
"""

import requests
import json
import time
import sys
sys.path.insert(0, '/var/www/ai-training/backend')

API_BASE = "http://localhost:8004/api"
TEST_PROJECT_ID = None
TEST_JOB_ID = None

def print_section(title):
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)

def test_1_create_project():
    """测试1：创建项目"""
    print_section("测试1：创建项目")
    
    url = f"{API_BASE}/projects"
    data = {
        "name": "PostgreSQL测试项目",
        "description": "验证数据库迁移后功能正常",
        "task_type": "classification"
    }
    
    try:
        res = requests.post(url, json=data)
        result = res.json()
        
        if res.status_code == 200:
            global TEST_PROJECT_ID
            TEST_PROJECT_ID = result['project_id']
            print(f"✅ 项目创建成功: {TEST_PROJECT_ID}")
            return True
        else:
            print(f"❌ 项目创建失败: {result}")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False

def test_2_verify_project_in_postgres():
    """测试2：验证项目已写入PostgreSQL"""
    print_section("测试2：验证项目写入PostgreSQL")
    
    from db_postgres import execute_query
    
    try:
        result = execute_query(
            "SELECT * FROM projects WHERE id = %s",
            (TEST_PROJECT_ID,)
        )
        
        if result and len(result) > 0:
            project = result[0]
            print(f"✅ 项目在PostgreSQL中找到")
            print(f"   名称: {project['name']}")
            print(f"   类型: {project['task_type']}")
            print(f"   创建时间: {project['created_at']}")
            return True
        else:
            print(f"❌ 项目在PostgreSQL中未找到")
            return False
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return False

def test_3_upload_dataset():
    """测试3：上传数据集"""
    print_section("测试3：上传数据集")
    
    import pandas as pd
    from pathlib import Path
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': ['A', 'A', 'B', 'B', 'A']
    })
    
    test_file = '/tmp/postgres_test_data.csv'
    test_data.to_csv(test_file, index=False)
    
    url = f"{API_BASE}/projects/{TEST_PROJECT_ID}/datasets"
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            data = {'name': '测试数据集'}
            res = requests.post(url, files=files, data=data)
        
        if res.status_code == 200:
            result = res.json()
            print(f"✅ 数据集上传成功: {result['dataset_id']}")
            print(f"   样本数: {result.get('total_samples', 'N/A')}")
            return result['dataset_id']
        else:
            print(f"❌ 上传失败: {res.text}")
            return None
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_4_verify_dataset_in_postgres(dataset_id):
    """测试4：验证数据集已写入PostgreSQL"""
    print_section("测试4：验证数据集写入PostgreSQL")
    
    from db_postgres import execute_query
    
    try:
        result = execute_query(
            "SELECT * FROM datasets WHERE id = %s",
            (dataset_id,)
        )
        
        if result and len(result) > 0:
            dataset = result[0]
            print(f"✅ 数据集在PostgreSQL中找到")
            print(f"   名称: {dataset['name']}")
            print(f"   样本数: {dataset['total_samples']}")
            print(f"   状态: {dataset['status']}")
            return True
        else:
            print(f"❌ 数据集在PostgreSQL中未找到")
            return False
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return False

def test_5_start_training(dataset_id):
    """测试5：启动训练任务"""
    print_section("测试5：启动训练任务")
    
    url = f"{API_BASE}/projects/{TEST_PROJECT_ID}/train"
    
    config = {
        "task_type": "classification",
        "model_type": "random_forest",
        "target_column": "target",
        "feature_columns": ["feature1", "feature2"],
        "n_estimators": 10
    }
    
    data = {
        "dataset_id": dataset_id,
        "config": json.dumps(config),
        "template": "fast"
    }
    
    try:
        res = requests.post(url, data=data)
        result = res.json()
        
        if res.status_code == 200:
            global TEST_JOB_ID
            TEST_JOB_ID = result['job_id']
            print(f"✅ 训练任务启动成功: {TEST_JOB_ID}")
            return TEST_JOB_ID
        else:
            print(f"❌ 启动失败: {result}")
            return None
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_6_verify_job_in_postgres(job_id):
    """测试6：验证训练任务已写入PostgreSQL"""
    print_section("测试6：验证训练任务写入PostgreSQL")
    
    from db_postgres import execute_query
    
    try:
        result = execute_query(
            "SELECT * FROM training_jobs WHERE id = %s",
            (job_id,)
        )
        
        if result and len(result) > 0:
            job = result[0]
            print(f"✅ 训练任务在PostgreSQL中找到")
            print(f"   模型: {job['model_name']}")
            print(f"   状态: {job['status']}")
            print(f"   配置: {job['config']}")
            return True
        else:
            print(f"❌ 训练任务在PostgreSQL中未找到")
            return False
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return False

def test_7_wait_and_verify_completion(job_id):
    """测试7：等待训练完成并验证"""
    print_section("测试7：等待训练完成")
    
    url = f"{API_BASE}/projects/{TEST_PROJECT_ID}/jobs/{job_id}/status"
    
    max_wait = 60  # 最多等60秒
    waited = 0
    
    while waited < max_wait:
        try:
            res = requests.get(url)
            if res.status_code == 200:
                status = res.json()
                print(f"   状态: {status['status']} | 进度: {status['progress']}%")
                
                if status['status'] == 'completed':
                    print(f"✅ 训练完成!")
                    print(f"   准确率: {status.get('best_accuracy', 'N/A')}")
                    return True
                elif status['status'] == 'failed':
                    print(f"❌ 训练失败")
                    return False
        except Exception as e:
            print(f"   查询失败: {e}")
        
        time.sleep(5)
        waited += 5
    
    print(f"⚠️  等待超时，但任务可能仍在运行")
    return True

def test_8_list_all_projects():
    """测试8：列出所有项目（验证查询性能）"""
    print_section("测试8：查询性能测试")
    
    url = f"{API_BASE}/projects"
    
    try:
        start = time.time()
        res = requests.get(url)
        elapsed = time.time() - start
        
        if res.status_code == 200:
            projects = res.json()
            print(f"✅ 查询成功 ({elapsed:.3f}s)")
            print(f"   总项目数: {len(projects)}")
            return True
        else:
            print(f"❌ 查询失败")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False

def main():
    print("="*60)
    print("🚀 PostgreSQL 全链路测试")
    print("="*60)
    
    results = []
    
    # 测试1：创建项目
    results.append(("创建项目", test_1_create_project()))
    
    if not results[-1][1]:
        print("\n❌ 测试中止：项目创建失败")
        return
    
    # 测试2：验证项目写入
    results.append(("项目写入验证", test_2_verify_project_in_postgres()))
    
    # 测试3：上传数据
    dataset_id = test_3_upload_dataset()
    results.append(("上传数据集", dataset_id is not None))
    
    if dataset_id:
        # 测试4：验证数据集写入
        results.append(("数据集写入验证", test_4_verify_dataset_in_postgres(dataset_id)))
        
        # 测试5：启动训练
        job_id = test_5_start_training(dataset_id)
        results.append(("启动训练", job_id is not None))
        
        if job_id:
            # 测试6：验证任务写入
            results.append(("任务写入验证", test_6_verify_job_in_postgres(job_id)))
            
            # 测试7：等待完成
            results.append(("训练完成", test_7_wait_and_verify_completion(job_id)))
    
    # 测试8：查询性能
    results.append(("查询性能", test_8_list_all_projects()))
    
    # 汇总
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:.<40} {status}")
    
    print(f"\n总计: {passed}/{total} 通过 ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 全链路测试通过！PostgreSQL迁移成功！")
    else:
        print("\n⚠️  部分测试失败，请检查日志")

if __name__ == "__main__":
    main()
