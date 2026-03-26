# -*- coding: utf-8 -*-
"""
时序分析模块 - 设备传感器时序数据分析
支持：趋势预测、异常检测、相关性分析、RUL预测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """时序分析器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def analyze_trends(self, df: pd.DataFrame, time_col: str, value_col: str, 
                       forecast_hours: int = 24) -> Dict:
        """
        分析趋势并预测未来
        
        Args:
            df: 数据框
            time_col: 时间列名
            value_col: 值列名
            forecast_hours: 预测未来多少小时
            
        Returns:
            趋势分析结果
        """
        # 确保时间列是datetime类型
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        
        # 计算时间特征（小时数，相对于起始点）
        df['hours'] = (df[time_col] - df[time_col].min()).dt.total_seconds() / 3600
        
        # 线性趋势拟合
        X = df['hours'].values.reshape(-1, 1)
        y = df[value_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 趋势方向
        slope = model.coef_[0]  # 每小时变化量
        trend_direction = '上升' if slope > 0.01 else ('下降' if slope < -0.01 else '平稳')
        
        # 预测未来
        last_hour = df['hours'].max()
        future_hours = np.array([last_hour + i for i in range(1, forecast_hours + 1)]).reshape(-1, 1)
        future_predictions = model.predict(future_hours)
        
        # 计算统计指标
        current_value = y[-1]
        mean_value = np.mean(y)
        std_value = np.std(y)
        max_value = np.max(y)
        min_value = np.min(y)
        
        # 变化率
        value_change = ((y[-1] - y[0]) / y[0] * 100) if y[0] != 0 else 0
        
        # 预测未来是否超标（假设正常范围）
        threshold_high = mean_value + 2 * std_value
        threshold_low = mean_value - 2 * std_value
        
        will_exceed_high = any(future_predictions > threshold_high)
        will_exceed_low = any(future_predictions < threshold_low)
        
        # 预计超标时间
        exceed_time = None
        if will_exceed_high:
            exceed_idx = np.where(future_predictions > threshold_high)[0][0]
            exceed_time = (df[time_col].max() + timedelta(hours=exceed_idx+1)).strftime('%Y-%m-%d %H:%M')
        
        return {
            'metric': value_col,
            'current_value': round(current_value, 2),
            'mean_value': round(mean_value, 2),
            'trend_direction': trend_direction,
            'trend_slope': round(slope, 4),
            'value_change_percent': round(value_change, 2),
            'forecast_hours': forecast_hours,
            'future_predictions': [
                {
                    'hour': i+1,
                    'predicted_value': round(float(v), 2),
                    'timestamp': (df[time_col].max() + timedelta(hours=i+1)).strftime('%m-%d %H:%M')
                }
                for i, v in enumerate(future_predictions[:12])  # 只返回前12小时详细预测
            ],
            'will_exceed_threshold': will_exceed_high or will_exceed_low,
            'exceed_time': exceed_time,
            'threshold_high': round(threshold_high, 2),
            'threshold_low': round(threshold_low, 2),
            'risk_level': 'high' if will_exceed_high else ('medium' if abs(slope) > 0.5 else 'low')
        }
    
    def detect_anomaly_patterns(self, df: pd.DataFrame, time_col: str, 
                                value_cols: List[str]) -> Dict:
        """
        检测时序异常模式
        
        检测的异常类型：
        1. 周期性异常 - 固定周期出现的异常
        2. 渐变劣化 - 缓慢持续恶化的趋势
        3. 突变点 - 突然的变化
        4. 波动性异常 - 方差突然增大
        """
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        
        patterns = []
        
        for col in value_cols:
            values = df[col].values
            
            # 1. 检测渐变劣化（长时间趋势）
            window_size = min(len(values) // 4, 50)
            if window_size > 10:
                early_mean = np.mean(values[:window_size])
                late_mean = np.mean(values[-window_size:])
                degradation = (late_mean - early_mean) / early_mean if early_mean != 0 else 0
                
                if abs(degradation) > 0.1:  # 变化超过10%
                    patterns.append({
                        'metric': col,
                        'pattern_type': '渐变劣化' if degradation > 0 else '渐变改善',
                        'severity': 'high' if abs(degradation) > 0.2 else 'medium',
                        'description': f'{col}在过去{len(values)}个时间点内{"上升" if degradation > 0 else "下降"}了{abs(degradation)*100:.1f}%',
                        'recommendation': '建议安排检修' if degradation > 0.15 else '持续监控'
                    })
            
            # 2. 检测波动性异常
            rolling_std = pd.Series(values).rolling(window=10, min_periods=5).std()
            recent_std = np.nanmean(rolling_std[-10:])
            early_std = np.nanmean(rolling_std[:10])
            
            if recent_std > early_std * 1.5:  # 波动增加50%
                patterns.append({
                    'metric': col,
                    'pattern_type': '波动性增加',
                    'severity': 'medium',
                    'description': f'{col}近期波动性是早期的{recent_std/early_std:.1f}倍，可能存在不稳定因素',
                    'recommendation': '检查设备紧固件和润滑状态'
                })
            
            # 3. 检测突变点（简单版本：找最大变化）
            diff = np.diff(values)
            max_change_idx = np.argmax(np.abs(diff))
            max_change = diff[max_change_idx]
            
            if abs(max_change) > np.std(values) * 3:  # 超过3倍标准差
                change_time = df[time_col].iloc[max_change_idx + 1]
                patterns.append({
                    'metric': col,
                    'pattern_type': '突变点',
                    'severity': 'high',
                    'description': f'{col}在{change_time}发生突变，变化量{max_change:.2f}',
                    'recommendation': '检查该时间点前后的操作记录'
                })
        
        # 4. 检测多变量相关性变化
        if len(value_cols) >= 2:
            correlations = []
            for i, col1 in enumerate(value_cols):
                for col2 in value_cols[i+1:]:
                    corr = df[col1].corr(df[col2])
                    if abs(corr) > 0.7:  # 强相关
                        correlations.append({
                            'metric1': col1,
                            'metric2': col2,
                            'correlation': round(corr, 3)
                        })
            
            if correlations:
                patterns.append({
                    'pattern_type': '强相关性',
                    'severity': 'info',
                    'description': f"发现{len(correlations)}组强相关指标",
                    'details': correlations,
                    'recommendation': '可利用相关性进行交叉验证'
                })
        
        return {
            'total_patterns': len(patterns),
            'patterns': sorted(patterns, key=lambda x: {'high': 0, 'medium': 1, 'low': 2, 'info': 3}.get(x['severity'], 4)),
            'summary': self._generate_summary(patterns)
        }
    
    def _generate_summary(self, patterns: List[Dict]) -> str:
        """生成异常模式总结"""
        if not patterns:
            return "未发现明显异常模式"
        
        high_risk = [p for p in patterns if p.get('severity') == 'high']
        medium_risk = [p for p in patterns if p.get('severity') == 'medium']
        
        if high_risk:
            return f"发现{len(high_risk)}个高风险异常，建议立即处理"
        elif medium_risk:
            return f"发现{len(medium_risk)}个中等风险异常，建议近期关注"
        else:
            return "发现一些轻微异常，保持常规监控"
    
    def predict_rul(self, df: pd.DataFrame, time_col: str, 
                    degradation_col: str, threshold: float) -> Dict:
        """
        预测剩余使用寿命 (Remaining Useful Life)
        
        Args:
            df: 历史数据
            time_col: 时间列
            degradation_col: 劣化指标列（如磨损量）
            threshold: 故障阈值（达到此值视为故障）
            
        Returns:
            RUL预测结果
        """
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        
        values = df[degradation_col].values
        hours = (df[time_col] - df[time_col].min()).dt.total_seconds().values / 3600
        
        # 线性拟合劣化趋势
        model = LinearRegression()
        model.fit(hours.reshape(-1, 1), values)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        
        current_value = values[-1]
        
        # 计算达到阈值的时间
        if slope <= 0:
            return {
                'rul_hours': None,
                'rul_days': None,
                'status': 'stable',
                'message': '指标趋势平稳或改善，无法预测RUL'
            }
        
        hours_to_failure = (threshold - current_value) / slope
        
        if hours_to_failure < 0:
            return {
                'rul_hours': 0,
                'rul_days': 0,
                'status': 'critical',
                'message': '已超过阈值，建议立即检修！'
            }
        
        days_to_failure = hours_to_failure / 24
        
        # 确定状态
        if hours_to_failure < 24:
            status = 'critical'
            message = f'预计{hours_to_failure:.1f}小时后达到阈值，立即处理！'
        elif hours_to_failure < 72:
            status = 'warning'
            message = f'预计{days_to_failure:.1f}天后达到阈值，建议本周安排维护'
        else:
            status = 'normal'
            message = f'预计{days_to_failure:.1f}天后达到阈值，正常监控即可'
        
        return {
            'rul_hours': round(hours_to_failure, 1),
            'rul_days': round(days_to_failure, 1),
            'current_value': round(current_value, 2),
            'threshold': threshold,
            'trend_slope': round(slope, 4),
            'status': status,
            'message': message,
            'estimated_failure_time': (df[time_col].max() + timedelta(hours=hours_to_failure)).strftime('%Y-%m-%d %H:%M')
        }
    
    def analyze_multivariate_correlation(self, df: pd.DataFrame, 
                                         target_col: str, 
                                         feature_cols: List[str]) -> Dict:
        """
        多变量相关性分析 - 找出影响目标指标的关键因素
        """
        correlations = []
        
        for col in feature_cols:
            if col in df.columns:
                corr = df[target_col].corr(df[col])
                if not np.isnan(corr):
                    correlations.append({
                        'feature': col,
                        'correlation': round(corr, 3),
                        'abs_correlation': abs(corr),
                        'direction': '正相关' if corr > 0 else '负相关',
                        'strength': '强' if abs(corr) > 0.7 else ('中等' if abs(corr) > 0.4 else '弱')
                    })
        
        # 按相关性绝对值排序
        correlations = sorted(correlations, key=lambda x: x['abs_correlation'], reverse=True)
        
        # 找出关键驱动因素
        key_drivers = [c for c in correlations if c['abs_correlation'] > 0.5]
        
        return {
            'target_metric': target_col,
            'total_features': len(correlations),
            'correlations': correlations,
            'key_drivers': key_drivers,
            'summary': f"{target_col}主要受{', '.join([k['feature'] for k in key_drivers[:3]])}影响"
        }


# 便捷函数
def analyze_equipment_trends(data_path: str, forecast_hours: int = 24,
                             time_col: Optional[str] = None,
                             value_cols: Optional[List[str]] = None) -> Dict:
    """
    分析设备传感器趋势
    
    Args:
        data_path: 数据文件路径
        forecast_hours: 预测未来多少小时
        time_col: 时间列名，默认自动检测
        value_cols: 数值列名列表，默认自动检测所有数值列
    """
    try:
        # 读取完整数据文件
        if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"读取数据文件失败: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': f'读取数据文件失败: {str(e)}',
            'trend_analysis': {},
            'anomaly_patterns': {'total_patterns': 0, 'patterns': [], 'summary': '读取失败'},
            'multivariate_analysis': {}
        }
    
    # 检查数据是否为空
    if df.empty:
        return {
            'timestamp': datetime.now().isoformat(),
            'error': '数据文件为空',
            'trend_analysis': {},
            'anomaly_patterns': {'total_patterns': 0, 'patterns': [], 'summary': '无数据'},
            'multivariate_analysis': {}
        }
    
    logger.info(f"读取数据文件成功: {data_path}, 形状: {df.shape}, 列: {df.columns.tolist()}")
    
    analyzer = TimeSeriesAnalyzer()
    
    # 自动检测时间列
    if time_col is None:
        # 优先检测常见时间列名
        time_candidates = ['timestamp', 'time', 'datetime', '时间', '日期', 'ts', 'date']
        for col in df.columns:
            if col.lower() in time_candidates:
                time_col = col
                break
        # 如果没找到，尝试第一个能解析为时间的列
        if time_col is None:
            for col in df.columns:
                try:
                    pd.to_datetime(df[col].iloc[:5])  # 只测试前5行
                    time_col = col
                    break
                except:
                    continue
        # 还是没找到，用第一列
        if time_col is None:
            time_col = df.columns[0]
            logger.warning(f"未找到时间列，使用第一列作为时间: {time_col}")
    
    logger.info(f"使用时间列: {time_col}")
    
    # 自动检测数值列（如果未指定）
    if value_cols is None:
        # 优先检测常见传感器列名
        sensor_candidates = ['temperature', 'pressure', 'vibration', 'rpm', 'voltage', 'oil_pressure',
                            '温度', '压力', '振动', '转速', '电压', '油压', 'current', '电流',
                            'power', '功率', 'speed', '速度', 'flow', '流量', 'degradation_level',
                            'degradation', 'degradation', '劣化程度', '磨损']
        available_cols = [c for c in sensor_candidates if c in df.columns]
        
        # 如果没找到，自动选择所有数值列（排除时间列）
        if not available_cols:
            available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if time_col in available_cols:
                available_cols.remove(time_col)
    else:
        available_cols = [c for c in value_cols if c in df.columns]
    
    logger.info(f"检测到数值列: {available_cols}")
    
    if not available_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return {
            'timestamp': datetime.now().isoformat(),
            'error': '未找到可分析的数值列',
            'available_columns': df.columns.tolist(),
            'numeric_columns': numeric_cols,
            'trend_analysis': {},
            'anomaly_patterns': {'total_patterns': 0, 'patterns': [], 'summary': '无可用数值列'},
            'multivariate_analysis': {}
        }
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'time_column': time_col,
        'value_columns': available_cols,
        'data_shape': {'rows': len(df), 'cols': len(df.columns)},
        'trend_analysis': {},
        'anomaly_patterns': {},
        'multivariate_analysis': {}
    }
    
    # 异常检测
    try:
        results['anomaly_patterns'] = analyzer.detect_anomaly_patterns(df, time_col, available_cols)
    except Exception as e:
        logger.error(f"异常检测失败: {e}")
        results['anomaly_patterns'] = {'total_patterns': 0, 'patterns': [], 'summary': f'检测失败: {str(e)}'}
    
    # 每个指标的趋势分析
    for col in available_cols:
        try:
            trend = analyzer.analyze_trends(df, time_col, col, forecast_hours)
            results['trend_analysis'][col] = trend
        except Exception as e:
            logger.warning(f"分析{col}趋势失败: {e}")
            results['trend_analysis'][col] = {'error': str(e)}
    
    # 多变量相关性（以failure或degradation_level为目标，如果有的话）
    target_candidates = ['failure', 'label', 'target', '故障', '标签', 'degradation_level', 'degradation', '劣化']
    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col and available_cols:
        try:
            results['multivariate_analysis'] = analyzer.analyze_multivariate_correlation(
                df, target_col, available_cols
            )
        except Exception as e:
            logger.warning(f"多变量相关性分析失败: {e}")
    
    return results


if __name__ == "__main__":
    # 测试
    import json
    
    result = analyze_equipment_trends('/tmp/equipment_failure_prediction.xlsx')
    print(json.dumps(result, indent=2, ensure_ascii=False))
