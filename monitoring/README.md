# AI训练平台监控文档

## 架构概览

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  AI训练平台  │────▶│  Prometheus │────▶│   Grafana   │
│  (FastAPI)  │     │  (时序数据库) │     │  (可视化)   │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│   /metrics  │  ← 暴露指标端点
└─────────────┘
```

## 资源消耗

| 组件 | 内存 | 磁盘/天 | 说明 |
|------|------|---------|------|
| Prometheus | 200-300MB | 100-200MB | 保留7天数据 |
| Grafana | 50-100MB | 几乎不增 | 配置持久化 |
| **合计** | **~350MB** | **~200MB** | 符合轻量要求 |

## 监控指标

### 1. 系统资源
- `ai_system_memory_percent` - 内存使用率
- `ai_system_cpu_percent` - CPU使用率
- `ai_loaded_models_count` - 加载模型数
- `ai_active_projects` - 活跃项目数

### 2. 推理服务
- `ai_prediction_total` - 预测总次数（按类型和状态）
- `ai_prediction_latency_seconds` - 预测延迟分布
- `ai_prediction_confidence` - 预测置信度分布

### 3. 训练任务
- `ai_training_total` - 训练任务数（按类型和状态）
- `ai_training_duration_seconds` - 训练耗时分布
- `ai_model_accuracy` - 模型准确率

## 告警规则

| 规则 | 条件 | 级别 | 说明 |
|------|------|------|------|
| TrainingJobFailed | 5分钟内有失败任务 | warning | 训练异常 |
| HighPredictionLatency | P95延迟>2秒 | warning | 性能下降 |
| HighPredictionErrorRate | 错误率>10% | critical | 服务异常 |
| HighMemoryUsage | 内存>85% | warning | 资源告警 |
| HighCPUUsage | CPU>80%超过3分钟 | warning | 资源告警 |
| ModelAccuracyDrop | 准确率<50% | warning | 模型退化 |
| ServiceDown | 服务无响应>1分钟 | critical | 服务宕机 |

## 快速开始

### 1. 启动监控

```bash
cd /var/www/ai-training
./monitoring/start.sh
```

### 2. 访问监控界面

- **Prometheus**: http://localhost:9090
  - 查询指标、查看告警状态
  
- **Grafana**: http://localhost:3000
  - 默认账号: `admin` / `admin`
  - 导入仪表盘: 上传 `grafana-dashboard.json`

### 3. 验证指标采集

```bash
# 检查指标端点
curl http://localhost:8004/metrics

# 在 Prometheus 中查询
curl 'http://localhost:9090/api/v1/query?query=ai_prediction_total'
```

## 停止监控

```bash
# 停止 Prometheus
kill $(cat monitoring/prometheus.pid)

# 停止 Grafana
kill $(cat monitoring/grafana.pid)
```

## 配置说明

### 修改告警阈值

编辑 `monitoring/alerts.yml`，重启 Prometheus 生效。

### 调整数据保留期

编辑 `monitoring/prometheus.yml`:
```yaml
storage:
  tsdb:
    retention.time: 7d  # 改为需要的天数
    retention.size: 10GB
```

### 添加飞书通知

1. 创建飞书机器人，获取 Webhook URL
2. 部署 Alertmanager（可选，资源消耗+100MB）
3. 配置 `alertmanager.yml` 添加 webhook

## 问题排查

### Prometheus 无法连接 API

检查 API 是否暴露 /metrics:
```bash
curl http://127.0.0.1:8004/metrics | head
```

### 指标不更新

检查 API 是否有流量：
```bash
# 发送测试请求
curl -X POST http://127.0.0.1:8004/api/inference/xxx/yyy \
  -F "text=测试"
```

### 内存占用过高

减少 Prometheus 保留期或采样频率：
```yaml
global:
  scrape_interval: 30s  # 增大间隔
```
