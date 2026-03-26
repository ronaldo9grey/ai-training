# -*- coding: utf-8 -*-
"""
大模型API集成模块
支持: DeepSeek, GPT-4, 文心一言等
"""

import os
import json
import logging
from typing import Dict, List, Optional
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    """大语言模型服务集成"""
    
    # 支持的模型提供商
    PROVIDERS = {
        'deepseek': {
            'name': 'DeepSeek',
            'api_url': 'https://api.deepseek.com/v1/chat/completions',
            'env_key': 'DEEPSEEK_API_KEY',
            'model': 'deepseek-chat'
        },
        'openai': {
            'name': 'OpenAI GPT',
            'api_url': 'https://api.openai.com/v1/chat/completions',
            'env_key': 'OPENAI_API_KEY',
            'model': 'gpt-4'
        },
        'baidu': {
            'name': '文心一言',
            'api_url': 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions',
            'env_key': 'BAIDU_API_KEY',
            'model': 'ERNIE-Bot'
        }
    }
    
    def __init__(self, provider: str = 'deepseek'):
        self.provider = provider
        self.config = self.PROVIDERS.get(provider)
        if not self.config:
            raise ValueError(f"不支持的模型提供商: {provider}")
        
        self.api_key = os.getenv(self.config['env_key'])
        if not self.api_key:
            logger.warning(f"未设置{self.config['env_key']}环境变量，大模型服务不可用")
    
    def analyze_equipment_report(self, equipment_data: Dict, 
                                  prediction_result: Dict) -> Dict:
        """
        设备故障根因分析
        
        Args:
            equipment_data: 设备传感器数据
            prediction_result: 模型预测结果
            
        Returns:
            根因分析和建议
        """
        if not self.api_key:
            return {'error': '大模型API未配置'}
        
        # 构造提示词
        prompt = self._build_analysis_prompt(equipment_data, prediction_result)
        
        try:
            response = self._call_api(prompt)
            return {
                'success': True,
                'provider': self.config['name'],
                'analysis': response,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"大模型调用失败: {e}")
            return {'error': str(e)}
    
    def _build_analysis_prompt(self, data: Dict, prediction: Dict) -> str:
        """构造分析提示词"""
        
        prompt = f"""你是一位资深的设备维护工程师，拥有20年经验。

设备当前传感器数据：
- 温度: {data.get('temperature')}°C
- 压力: {data.get('pressure')} bar
- 振动: {data.get('vibration')} mm/s
- 转速: {data.get('rpm')} rpm
- 电压: {data.get('voltage')} V
- 油压: {data.get('oil_pressure')} bar
- 运行时长: {data.get('runtime_hours')} 小时

AI模型预测结果：
- 故障概率: {prediction.get('failure_probability', 0) * 100:.1f}%
- 主要风险指标: {prediction.get('key_risk_factors', 'N/A')}

请分析：
1. 可能的故障根因是什么？（列出3个最可能的原因）
2. 建议的维护措施？（按优先级排序）
3. 是否需要立即停机？
4. 预估的维修成本和时间？

用中文回答，要专业但易懂。"""
        
        return prompt
    
    def _call_api(self, prompt: str) -> str:
        """调用大模型API"""
        
        if self.provider == 'deepseek':
            return self._call_deepseek(prompt)
        elif self.provider == 'openai':
            return self._call_openai(prompt)
        elif self.provider == 'baidu':
            return self._call_baidu(prompt)
        else:
            raise ValueError(f"未实现的提供商: {self.provider}")
    
    def _call_deepseek(self, prompt: str) -> str:
        """调用DeepSeek API"""
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.config['model'],
            'messages': [
                {'role': 'system', 'content': '你是一位专业的设备维护工程师。'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.7,
            'max_tokens': 2000
        }
        
        response = requests.post(
            self.config['api_url'],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
    
    def _call_openai(self, prompt: str) -> str:
        """调用OpenAI API"""
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.config['model'],
            'messages': [
                {'role': 'system', 'content': 'You are a professional equipment maintenance engineer.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.7,
            'max_tokens': 2000
        }
        
        response = requests.post(
            self.config['api_url'],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
    
    def generate_maintenance_report(self, analysis_results: List[Dict]) -> Dict:
        """
        生成维护报告
        
        基于一段时间的分析结果，生成综合报告
        """
        if not self.api_key:
            return {'error': '大模型API未配置'}
        
        # 汇总数据
        summary = self._summarize_results(analysis_results)
        
        prompt = f"""基于以下设备分析数据，生成一份专业的维护建议报告：

数据摘要：
{summary}

请生成包含以下内容的报告：
1. 执行摘要（关键发现）
2. 设备健康度评估
3. 风险趋势分析
4. 具体维护建议（分短期/中期/长期）
5. 预算估算

格式为Markdown，专业但易读。"""
        
        try:
            response = self._call_api(prompt)
            return {
                'success': True,
                'report': response,
                'report_type': 'maintenance_plan',
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _summarize_results(self, results: List[Dict]) -> str:
        """汇总分析结果"""
        total = len(results)
        failure_count = sum(1 for r in results if r.get('is_failure', False))
        
        summary = f"""
- 分析样本数: {total}
- 故障预警次数: {failure_count}
- 平均故障概率: {sum(r.get('failure_probability', 0) for r in results) / total * 100:.1f}%
- 时间范围: {results[0].get('timestamp', 'N/A')} 至 {results[-1].get('timestamp', 'N/A')}
"""
        return summary


class HybridPredictor:
    """
    混合预测器
    本地模型 + 大模型 API 的组合
    """
    
    def __init__(self, local_model_path: str = None, 
                 llm_provider: str = 'deepseek'):
        self.local_model_path = local_model_path
        self.llm = LLMService(llm_provider) if llm_provider else None
        
    def predict(self, sensor_data: Dict, use_llm_analysis: bool = True) -> Dict:
        """
        混合预测
        
        步骤：
        1. 本地模型快速预测（<100ms）
        2. 如需要，调用大模型深度分析（~2s）
        """
        # 步骤1: 本地模型预测
        local_result = self._local_predict(sensor_data)
        
        result = {
            'local_prediction': local_result,
            'timestamp': datetime.now().isoformat()
        }
        
        # 步骤2: 高风险时调用大模型
        if use_llm_analysis and local_result.get('failure_probability', 0) > 0.3:
            logger.info("故障概率较高，调用大模型深度分析")
            llm_result = self.llm.analyze_equipment_report(
                sensor_data, 
                local_result
            )
            result['llm_analysis'] = llm_result
        
        return result
    
    def _local_predict(self, data: Dict) -> Dict:
        """本地模型预测（快速）"""
        # 调用本地Random Forest或LSTM模型
        # 这里简化处理，实际应加载模型
        
        # 模拟预测结果
        failure_prob = 0.0
        risk_factors = []
        
        if data.get('temperature', 0) > 90:
            failure_prob += 0.4
            risk_factors.append('温度过高')
        
        if data.get('oil_pressure', 10) < 1.5:
            failure_prob += 0.3
            risk_factors.append('油压不足')
        
        if data.get('vibration', 0) > 2.5:
            failure_prob += 0.2
            risk_factors.append('振动异常')
        
        return {
            'failure_probability': min(failure_prob, 0.99),
            'is_failure': failure_prob > 0.5,
            'key_risk_factors': ', '.join(risk_factors) if risk_factors else '正常',
            'model_type': 'local_rf'
        }


# 便捷函数
def analyze_with_llm(equipment_data: Dict, api_key: str = None) -> Dict:
    """
    便捷函数：用大模型分析设备数据
    """
    if api_key:
        os.environ['DEEPSEEK_API_KEY'] = api_key
    
    llm = LLMService('deepseek')
    
    # 模拟本地预测结果
    prediction = {
        'failure_probability': 0.75,
        'key_risk_factors': '温度过高, 油压不足'
    }
    
    return llm.analyze_equipment_report(equipment_data, prediction)


if __name__ == "__main__":
    # 测试
    test_data = {
        'temperature': 95,
        'pressure': 18,
        'vibration': 3.2,
        'rpm': 2800,
        'voltage': 220,
        'oil_pressure': 1.2,
        'runtime_hours': 500
    }
    
    # 混合预测
    predictor = HybridPredictor(llm_provider='deepseek')
    result = predictor.predict(test_data, use_llm_analysis=True)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
