#!/usr/bin/env python3
"""
监控测试脚本 - 生成模拟数据验证监控
"""

import requests
import time
import random
import sys

API_BASE = "http://127.0.0.1:8004"

def test_metrics_endpoint():
    """测试指标端点"""
    print("=== 测试 /metrics 端点 ===")
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=5)
        if r.status_code == 200:
            content = r.text
            # 检查关键指标是否存在
            metrics_to_check = [
                "ai_prediction_total",
                "ai_system_memory_percent",
                "ai_loaded_models_count"
            ]
            for metric in metrics_to_check:
                if metric in content:
                    print(f"✅ {metric} - 存在")
                else:
                    print(f"⚠️ {metric} - 不存在（可能暂无数据）")
            return True
        else:
            print(f"❌ 状态码错误: {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False

def test_prometheus_connection():
    """测试 Prometheus 连接"""
    print("\n=== 测试 Prometheus ===")
    try:
        r = requests.get("http://localhost:9090/-/healthy", timeout=3)
        if r.status_code == 200:
            print("✅ Prometheus 运行正常")
            return True
        else:
            print(f"⚠️ Prometheus 返回: {r.status_code}")
            return False
    except:
        print("⚠️ Prometheus 未启动（可运行 ./monitoring/start.sh）")
        return False

def generate_prediction_traffic():
    """生成预测请求产生监控数据"""
    print("\n=== 生成预测流量 ===")
    
    # 使用 Demo 项目进行测试
    project_id = "d6e7d716-030a-4143-a9fc-5ed0824394f8"
    job_id = "f41ab9c9-5718-4663-843a-8dd81e9ca501"
    
    test_texts = [
        "这个产品非常好用，强烈推荐",
        "质量一般，性价比不高",
        "物流太慢了，等了一个星期",
        "服务态度很好，很满意",
        "包装破损，产品损坏了",
        "价格实惠，质量不错",
        "功能齐全，操作简单"
    ]
    
    success_count = 0
    for i in range(10):
        try:
            text = random.choice(test_texts)
            r = requests.post(
                f"{API_BASE}/api/inference/{project_id}/{job_id}",
                data={"text": text},
                timeout=10
            )
            if r.status_code == 200:
                success_count += 1
                print(f"  请求 {i+1}: ✅ 成功")
            else:
                print(f"  请求 {i+1}: ❌ 失败 ({r.status_code})")
            time.sleep(0.5)
        except Exception as e:
            print(f"  请求 {i+1}: ❌ 异常: {e}")
    
    print(f"\n完成: {success_count}/10 成功")
    return success_count > 0

def query_prometheus_metrics():
    """查询 Prometheus 中的指标"""
    print("\n=== 查询 Prometheus 指标 ===")
    
    queries = [
        ("预测总次数", "ai_prediction_total"),
        ("系统内存使用", "ai_system_memory_percent"),
        ("加载模型数", "ai_loaded_models_count"),
    ]
    
    for name, query in queries:
        try:
            r = requests.get(
                "http://localhost:9090/api/v1/query",
                params={"query": query},
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("data", {}).get("result"):
                    print(f"✅ {name}: {data['data']['result']}")
                else:
                    print(f"⚠️ {name}: 暂无数据")
            else:
                print(f"❌ {name}: 查询失败")
        except Exception as e:
            print(f"❌ {name}: {e}")

def main():
    print("AI训练平台监控测试\n")
    print("=" * 50)
    
    # 1. 测试指标端点
    if not test_metrics_endpoint():
        print("\n❌ 指标端点测试失败，请检查 API 是否启动")
        sys.exit(1)
    
    # 2. 测试 Prometheus
    prometheus_ok = test_prometheus_connection()
    
    # 3. 生成流量
    if generate_prediction_traffic():
        time.sleep(2)  # 等待数据上报
        
        # 4. 再次检查指标
        print("\n=== 验证指标更新 ===")
        test_metrics_endpoint()
        
        # 5. 查询 Prometheus
        if prometheus_ok:
            query_prometheus_metrics()
    
    print("\n" + "=" * 50)
    print("监控测试完成")
    print("\n建议:")
    print("1. 访问 Prometheus: http://localhost:9090")
    print("2. 访问 Grafana: http://localhost:3000")
    print("3. 查看仪表盘验证可视化效果")

if __name__ == "__main__":
    main()
