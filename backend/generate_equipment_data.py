# -*- coding: utf-8 -*-
"""
设备故障预测模拟数据生成器
生成包含正常/故障状态的设备传感器数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_equipment_data(n_samples=1000, failure_rate=0.1):
    """
    生成设备传感器数据
    
    字段说明:
    - timestamp: 时间戳
    - temperature: 温度(°C) - 正常60-80, 故障时>90
    - pressure: 压力(bar) - 正常10-15, 故障时>18或<8
    - vibration: 振动(mm/s) - 正常0.5-1.5, 故障时>2.5
    - rpm: 转速(rpm) - 正常2800-3200, 故障时偏离范围
    - voltage: 电压(V) - 正常220±10%, 故障时波动大
    - oil_pressure: 油压(bar) - 正常2-4, 故障时<1.5
    - runtime_hours: 运行时长(小时)
    - failure: 是否故障(0=正常, 1=故障)
    """
    
    data = []
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    
    for i in range(n_samples):
        # 基础时间（每5分钟一条记录）
        timestamp = base_time + timedelta(minutes=5*i)
        
        # 随机决定是否为故障样本
        is_failure = random.random() < failure_rate
        
        if is_failure:
            # 故障状态 - 某些指标异常
            failure_type = random.choice(['overheat', 'pressure_high', 'vibration', 'oil_low', 'combined'])
            
            if failure_type == 'overheat':
                temperature = random.uniform(90, 105)
                pressure = random.uniform(10, 15)
                vibration = random.uniform(0.5, 1.5)
            elif failure_type == 'pressure_high':
                temperature = random.uniform(60, 85)
                pressure = random.uniform(18, 25)
                vibration = random.uniform(0.5, 1.5)
            elif failure_type == 'vibration':
                temperature = random.uniform(60, 85)
                pressure = random.uniform(10, 15)
                vibration = random.uniform(2.5, 5.0)
            elif failure_type == 'oil_low':
                temperature = random.uniform(60, 85)
                pressure = random.uniform(10, 15)
                vibration = random.uniform(0.5, 1.5)
                oil_pressure = random.uniform(0.5, 1.2)
            else:  # combined
                temperature = random.uniform(88, 100)
                pressure = random.uniform(17, 22)
                vibration = random.uniform(2.0, 4.0)
            
            rpm = random.uniform(2500, 2800)  # 转速偏低
            voltage = random.uniform(200, 240)  # 电压波动
            oil_pressure = random.uniform(0.5, 1.5) if failure_type != 'oil_low' else oil_pressure
            
        else:
            # 正常状态
            temperature = random.uniform(60, 82)
            pressure = random.uniform(10, 15)
            vibration = random.uniform(0.5, 1.5)
            rpm = random.uniform(2850, 3200)
            voltage = random.uniform(215, 235)
            oil_pressure = random.uniform(2.0, 4.0)
        
        # 运行时长累积
        runtime_hours = 100 + (i * 5 / 60)  # 从100小时开始累积
        
        # 添加一些随机噪声（模拟真实传感器误差）
        temperature += random.uniform(-1, 1)
        pressure += random.uniform(-0.3, 0.3)
        vibration += random.uniform(-0.05, 0.05)
        rpm += random.uniform(-50, 50)
        voltage += random.uniform(-3, 3)
        oil_pressure += random.uniform(-0.1, 0.1)
        
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': round(temperature, 1),
            'pressure': round(pressure, 2),
            'vibration': round(vibration, 2),
            'rpm': round(rpm, 0),
            'voltage': round(voltage, 1),
            'oil_pressure': round(oil_pressure, 2),
            'runtime_hours': round(runtime_hours, 1),
            'failure': 1 if is_failure else 0
        })
    
    df = pd.DataFrame(data)
    return df


def generate_test_data(n_samples=100):
    """生成测试数据（用于验证模型效果）"""
    return generate_equipment_data(n_samples=n_samples, failure_rate=0.15)


if __name__ == "__main__":
    # 生成训练数据
    print("生成设备故障预测训练数据...")
    train_data = generate_equipment_data(n_samples=1000, failure_rate=0.1)
    
    # 统计
    failure_count = train_data['failure'].sum()
    normal_count = len(train_data) - failure_count
    
    print(f"训练数据生成完成:")
    print(f"  总样本数: {len(train_data)}")
    print(f"  正常样本: {normal_count} ({normal_count/len(train_data)*100:.1f}%)")
    print(f"  故障样本: {failure_count} ({failure_count/len(train_data)*100:.1f}%)")
    print("\n数据预览:")
    print(train_data.head(10).to_string())
    print("\n数据统计:")
    print(train_data.describe().to_string())
    
    # 保存到Excel
    output_file = "/tmp/equipment_failure_prediction.xlsx"
    train_data.to_excel(output_file, index=False, sheet_name='training_data')
    print(f"\n数据已保存到: {output_file}")
    
    # 生成测试数据
    test_data = generate_test_data(n_samples=50)
    test_file = "/tmp/equipment_test_data.xlsx"
    test_data.to_excel(test_file, index=False, sheet_name='test_data')
    print(f"测试数据已保存到: {test_file}")
