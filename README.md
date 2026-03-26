# AI模型训练工坊

从数据到部署的完整MLOps平台

## 功能特性

- 📊 **数据管理** - 数据集上传、预览、标签管理
- 🏷️ **数据标注** - 文本分类标注、图片画框标注
- 🤖 **模型训练** - 支持NLP(BERT)、图像分类、目标检测、时序分析、AutoML超参优化
- 🚀 **模型部署** - API服务部署、Ollama导出
- 🧪 **模型测试** - 在线推理测试
- 📈 **训练监控** - 实时指标、内存管理
- 🔄 **在线学习** - 增量学习、自动学习
- 🔔 **告警系统** - 模型性能监控

## 技术栈

- **前端**: React 18 + Tailwind CSS
- **后端**: FastAPI + Uvicorn
- **数据库**: PostgreSQL
- **任务队列**: Celery + Redis
- **部署**: Nginx + Systemd

## 部署

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
systemctl start ai-training
systemctl start nginx
```

## 目录结构

```
/var/www/ai-training/
├── backend/          # FastAPI 后端
├── frontend/         # React 前端
├── data/            # 数据集存储
├── models/          # 模型存储
├── logs/            # 训练日志
└── monitoring/      # 监控配置
```

## 版本

v2.2.0 - 集成 AutoML 超参数优化到训练流程
