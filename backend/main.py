# AI模型训练工坊 —— 从数据到部署的完整MLOps平台
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import uuid
from datetime import datetime
import asyncio
from pathlib import Path
import logging
import io
import base64
import numpy as np

logger = logging.getLogger(__name__)

# PostgreSQL数据库
from db_postgres import (
    init_connection_pool, init_database_tables,
    get_db_connection, execute_query, execute_update,
    test_connection as test_db_connection
)

# Prometheus监控
from metrics_collector import metrics, get_metrics_response, CONTENT_TYPE_LATEST

# Celery 任务队列
from celeryconfig import celery_app
from tasks import (
    submit_training_task, submit_ml_training_task, submit_image_training_task, submit_object_detection_task,
    run_training_task, run_ml_training_task, run_image_training_task, run_object_detection_task
)

# 模型导出
from image_trainer import quantize_model
from alert_engine import AlertEngine, get_project_alert_rules, get_alert_history
from automl_engine import (
    AutoMLExperiment, AutoMLConfig, create_automl_experiment,
    run_automl_trial, compare_trials, get_recommended_params
)

app = FastAPI(title="AI模型训练工坊", description="从数据到部署的完整MLOps平台", version="2.2.0")

# 初始化PostgreSQL
init_connection_pool()
init_database_tables()

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置
DATA_DIR = Path("/var/www/ai-training/data")
MODELS_DIR = Path("/var/www/ai-training/models")
LOGS_DIR = Path("/var/www/ai-training/logs")



# 初始化数据库
def init_db():
    """初始化数据库，检查连接状态"""
    if not test_db_connection():
        raise Exception("PostgreSQL连接失败，请检查配置")
    logger.info("PostgreSQL数据库连接正常")

init_db()

# ============ 数据模型 ============

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    task_type: str = "text_classification"

class TrainingConfig(BaseModel):
    model_name: str = "bert-base-chinese"
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    max_length: int = 512
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    gpu_type: str = "local"  # local 或 cloud
    cloud_provider: Optional[str] = None

class AnnotationTask(BaseModel):
    content: str
    label: Optional[str] = None

# ============ API端点 ============

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "AI模型训练工坊", "version": "2.2.0"}


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus监控指标端点"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    
    # 更新系统指标
    try:
        metrics.update_system_metrics(
            loaded_models=len(inference_service.models)
        )
    except:
        pass
    
    # 更新活跃项目数
    try:
        rows = execute_query('SELECT COUNT(*) as cnt FROM projects')
        if rows:
            metrics.update_active_projects(rows[0]['cnt'])
    except:
        pass
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# 项目管理
@app.post("/api/projects")
async def create_project(project: ProjectCreate):
    project_id = str(uuid.uuid4())
    
    execute_update('''
        INSERT INTO projects (id, name, description, task_type, config)
        VALUES (%s, %s, %s, %s, %s)
    ''', (project_id, project.name, project.description, project.task_type, json.dumps({})))
    
    # 创建项目目录
    (DATA_DIR / project_id).mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / project_id).mkdir(parents=True, exist_ok=True)
    
    return {"success": True, "project_id": project_id}

@app.get("/api/projects")
async def list_projects():
    try:
        # 使用PostgreSQL查询
        from db_postgres import execute_query
        
        rows = execute_query('''
            SELECT id, name, description, task_type, status, created_at 
            FROM projects ORDER BY created_at DESC
        ''')
        
        projects = []
        for row in rows:
            # 根据训练任务状态动态计算项目状态
            job_rows = execute_query('''
                SELECT status FROM training_jobs 
                WHERE project_id = %s ORDER BY created_at DESC LIMIT 1
            ''', (row['id'],))
            
            display_status = row['status']
            if job_rows:
                job_status = job_rows[0]['status']
                if job_status == 'training':
                    display_status = 'training'
                elif job_status == 'completed':
                    display_status = 'completed'
                elif job_status == 'failed':
                    # 如果有失败的任务，检查是否有成功的
                    completed_jobs = execute_query('''
                        SELECT COUNT(*) as cnt FROM training_jobs 
                        WHERE project_id = %s AND status = 'completed'
                    ''', (row['id'],))
                    if completed_jobs and completed_jobs[0]['cnt'] > 0:
                        display_status = 'completed'
                    else:
                        display_status = 'created'  # 失败但还没成功的，显示未开始
            
            projects.append({
                "id": row['id'],
                "name": row['name'],
                "description": row['description'],
                "task_type": row['task_type'],
                "status": display_status,
                "created_at": row['created_at']
            })
        
        return {"projects": projects}
    except Exception as e:
        logger.error(f"查询项目列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    rows = execute_query('SELECT * FROM projects WHERE id = %s', (project_id,))
    
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    row = rows[0]
    return {
        "id": row['id'],
        "name": row['name'],
        "description": row['description'],
        "task_type": row['task_type'],
        "status": row['status'],
        "created_at": row['created_at'],
        "updated_at": row['updated_at'],
        "config": row['config'] if row['config'] else {}
}

# 创建Demo项目（一键体验）
@app.post("/api/demo/create")
async def create_demo_project():
    """创建示例文本分类项目，包含示例数据集"""
    import uuid
    from datetime import datetime
    
    project_id = str(uuid.uuid4())
    dataset_id = str(uuid.uuid4())
    
    # 创建项目
    execute_update('''
        INSERT INTO projects (id, name, description, task_type, config)
        VALUES (%s, %s, %s, %s, %s)
    ''', (project_id, "🎮 示例项目-新闻分类", "内置示例数据集，5分钟体验完整训练流程", "text_classification", json.dumps({})))
    
    # 创建项目目录
    project_data_dir = DATA_DIR / project_id
    project_data_dir.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / project_id).mkdir(parents=True, exist_ok=True)
    
    # 生成示例数据集（中文新闻分类：体育、科技、娱乐、财经）
    demo_data = [
        # 体育
        ("国足在世预赛中取得关键胜利，晋级形势大好", "体育"),
        ("NBA总决赛精彩对决，湖人队夺得总冠军", "体育"),
        ("梅西加盟迈阿密国际，美国足球迎来巨星时代", "体育"),
        ("中国乒乓球队包揽世乒赛全部金牌", "体育"),
        ("马拉松赛事在全国各地火热开展", "体育"),
        ("冬奥会花样滑冰比赛精彩瞬间回顾", "体育"),
        ("中超联赛新赛季开幕，球迷热情高涨", "体育"),
        ("网球大满贯赛事即将开幕，中国选手备受期待", "体育"),
        ("游泳世锦赛中国队再创佳绩", "体育"),
        ("电竞入亚，英雄联盟成为亚运会正式比赛项目", "体育"),
        ("羽毛球世锦赛中国队夺得三金", "体育"),
        ("F1赛车上海站比赛圆满结束", "体育"),
        ("中国女排在世界联赛中取得连胜", "体育"),
        ("高尔夫球大师赛产生新冠军", "体育"),
        ("滑雪运动在国内越来越受欢迎", "体育"),
        
        # 科技
        ("人工智能技术在医疗领域取得突破性进展", "科技"),
        ("新款智能手机发布，搭载最新处理器", "科技"),
        ("SpaceX星舰发射成功，商业航天迈出新步伐", "科技"),
        ("量子计算机计算能力再创新高", "科技"),
        ("5G网络覆盖全国主要城市，用户体验大幅提升", "科技"),
        ("电动汽车电池技术突破，续航里程大幅增加", "科技"),
        ("元宇宙概念持续升温，各大厂商布局虚拟现实", "科技"),
        ("区块链技术在供应链金融中的应用", "科技"),
        ("国产操作系统市场份额稳步提升", "科技"),
        ("机器人技术在制造业的广泛应用", "科技"),
        ("云计算服务降价，企业数字化转型加速", "科技"),
        ("脑机接口技术取得重要进展", "科技"),
        ("无人机配送服务开始试点运营", "科技"),
        ("智能家居生态系统日趋完善", "科技"),
        ("自动驾驶技术安全性持续提升", "科技"),
        
        # 娱乐
        ("春节档电影票房创历史新高", "娱乐"),
        ("知名歌手世界巡回演唱会启动", "娱乐"),
        ("热门电视剧迎来大结局，观众反响热烈", "娱乐"),
        ("综艺节目创新模式受到年轻观众喜爱", "娱乐"),
        ("国产动画电影在国际上获奖", "娱乐"),
        ("音乐节吸引数万名乐迷参加", "娱乐"),
        ("经典话剧复排，票房火爆", "娱乐"),
        ("网络剧品质提升，精品化趋势明显", "娱乐"),
        ("明星慈善活动引发社会关注", "娱乐"),
        ("相声小品晚会带来欢声笑语", "娱乐"),
        ("纪录片拍摄技术不断创新", "娱乐"),
        ("选秀节目挖掘新人，培养新生代艺人", "娱乐"),
        ("影视特效技术达到国际先进水平", "娱乐"),
        ("在线音乐平台版权合作取得进展", "娱乐"),
        ("舞台剧市场规模持续扩大", "娱乐"),
        
        # 财经
        ("央行宣布降准，释放流动性支持实体经济", "财经"),
        ("A股市场震荡调整，投资者情绪谨慎", "财经"),
        ("新能源汽车销量创新高，产业链受益", "财经"),
        ("房地产政策优化，市场逐步回暖", "财经"),
        ("跨境电商快速发展，出口贸易增长", "财经"),
        ("数字货币监管政策趋于完善", "财经"),
        ("银行业数字化转型加速推进", "财经"),
        ("消费复苏带动零售行业增长", "财经"),
        ("基建投资加大，拉动相关产业", "财经"),
        ("上市公司年报披露，业绩整体向好", "财经"),
        ("保险行业创新产品满足多样化需求", "财经"),
        ("股市注册制改革稳步推进", "财经"),
        ("外汇市场保持稳定，人民币汇率坚挺", "财经"),
        ("基金市场规模扩大，投资者结构优化", "财经"),
        ("绿色金融支持双碳目标实现", "财经"),
    ]
    
    # 创建DataFrame并保存
    import pandas as pd
    df = pd.DataFrame(demo_data, columns=["text", "label"])
    
    # 打乱顺序
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    dataset_path = project_data_dir / f"{dataset_id}.csv"
    df.to_csv(dataset_path, index=False)
    
    # 划分数据集
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    train_path = project_data_dir / f"{dataset_id}_train.csv"
    val_path = project_data_dir / f"{dataset_id}_val.csv"
    test_path = project_data_dir / f"{dataset_id}_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # 保存数据集记录
    labels = df['label'].unique().tolist()
    meta_info = json.dumps({
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'preprocessed': True
    })
    execute_update('''
        INSERT INTO datasets (id, project_id, name, file_path, file_type, total_samples, labels, status, meta_info)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (dataset_id, project_id, "📚 示例数据集-新闻分类", str(dataset_path), 'csv', len(df), json.dumps(labels), 'preprocessed', meta_info))
    
    return {
        "success": True,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "message": "示例项目创建成功！包含60条新闻数据（体育/科技/娱乐/财经各15条）",
        "stats": {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "labels": labels
        },
        "next_steps": [
            "1. 进入项目，选择数据集",
            "2. 点击'开始训练'，选择'快速训练'模板",
            "3. 5分钟后查看训练结果"
        ]
    }

# 数据集管理
@app.post("/api/projects/{project_id}/datasets")
async def upload_dataset(
    project_id: str,
    file: UploadFile = File(...),
    name: str = Form(...)
):
    # 验证项目存在
    rows = execute_query('SELECT id FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    # 保存文件
    dataset_id = str(uuid.uuid4())
    file_ext = file.filename.split('.')[-1]
    file_path = DATA_DIR / project_id / f"{dataset_id}.{file_ext}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 解析数据集
    try:
        if file_ext in ['csv', 'txt']:
            import pandas as pd
            df = pd.read_csv(file_path)
            total_samples = len(df)
            # 假设最后一列是标签
            labels = df.iloc[:, -1].unique().tolist() if len(df.columns) > 1 else []
        else:
            total_samples = 0
            labels = []
    except Exception as e:
        total_samples = 0
        labels = []
    
    execute_update('''
        INSERT INTO datasets (id, project_id, name, file_path, file_type, total_samples, labels, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ''', (dataset_id, project_id, name, str(file_path), file_ext, total_samples, json.dumps(labels), 'uploaded'))
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "total_samples": total_samples,
        "labels": labels
    }

@app.get("/api/projects/{project_id}/datasets")
async def list_datasets(project_id: str):
    rows = execute_query('''
        SELECT id, name, file_type, total_samples, labels, status, created_at
        FROM datasets WHERE project_id = %s ORDER BY created_at DESC
    ''', (project_id,))
    
    datasets = []
    for row in rows:
        datasets.append({
            "id": row['id'],
            "name": row['name'],
            "file_type": row['file_type'],
            "total_samples": row['total_samples'],
            "labels": row['labels'] if row['labels'] else [],
            "status": row['status'],
            "created_at": row['created_at']
        })
    
    return {"datasets": datasets}

# 数据预览（限制条数）
@app.get("/api/projects/{project_id}/datasets/{dataset_id}/preview")
async def preview_dataset(project_id: str, dataset_id: str, limit: int = 100):
    """预览数据集内容，默认最多返回100条"""
    rows = execute_query('''
        SELECT file_path, file_type FROM datasets WHERE id = %s AND project_id = %s
    ''', (dataset_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    file_path = rows[0]['file_path']
    file_type = rows[0]['file_type']
    
    try:
        import os
        import pandas as pd
        
        # 检查是否是目录
        if os.path.isdir(file_path):
            # 首先尝试查找图片文件
            image_files = []
            csv_files = []
            
            for root, dirs, files in os.walk(file_path):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                        rel_path = os.path.relpath(os.path.join(root, f), file_path)
                        image_files.append(rel_path)
                    elif f.lower().endswith('.csv'):
                        csv_files.append(os.path.join(root, f))
            
            # 如果是图片数据集
            if image_files and not csv_files:
                return {
                    "columns": ["图片路径", "类别文件夹"],
                    "preview": [{"图片路径": f, "类别文件夹": f.split('/')[0] if '/' in f else '根目录'} for f in image_files[:limit]],
                    "total_rows": len(image_files),
                    "preview_rows": min(len(image_files), limit),
                    "limit": limit,
                    "type": "image_folder"
                }
            
            # 如果是CSV数据集（目录下有CSV文件）
            if csv_files:
                # 使用第一个CSV文件
                csv_path = csv_files[0]
                df = pd.read_csv(csv_path)
                total_rows = len(df)
                preview_df = df.head(min(limit, 100))
                
                return {
                    "columns": df.columns.tolist(),
                    "preview": preview_df.to_dict(orient='records'),
                    "total_rows": total_rows,
                    "preview_rows": len(preview_df),
                    "limit": limit,
                    "type": "csv"
                }
            
            # 空目录
            return {
                "columns": [],
                "preview": [],
                "total_rows": 0,
                "preview_rows": 0,
                "limit": limit,
                "type": "empty"
            }
        else:
            # CSV/文本数据集（具体文件路径）
            df = pd.read_csv(file_path)
            total_rows = len(df)
            
            # 限制预览条数
            preview_df = df.head(min(limit, 100))
            
            return {
                "columns": df.columns.tolist(),
                "preview": preview_df.to_dict(orient='records'),
                "total_rows": total_rows,
                "preview_rows": len(preview_df),
                "limit": limit,
                "type": "csv"
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"数据解析失败: {str(e)}")


# 图片数据集查看单张图片
@app.get("/api/projects/{project_id}/datasets/{dataset_id}/image")
async def get_dataset_image(project_id: str, dataset_id: str, path: str = Query(..., description="图片相对路径")):
    """获取图片数据集中的单张图片"""
    rows = execute_query('''
        SELECT file_path, file_type FROM datasets WHERE id = %s AND project_id = %s
    ''', (dataset_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    file_path = rows[0]['file_path']
    
    import os
    from fastapi.responses import FileResponse
    
    # 安全检查：确保路径在数据集目录内
    full_path = os.path.normpath(os.path.join(file_path, path))
    if not full_path.startswith(os.path.normpath(file_path)):
        raise HTTPException(status_code=403, detail="非法路径")
    
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="图片不存在")
    
    # 根据文件扩展名返回正确的 content-type
    ext = os.path.splitext(full_path)[1].lower()
    media_type = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }.get(ext, 'application/octet-stream')
    
    return FileResponse(full_path, media_type=media_type)


# 数据集下载
@app.get("/api/projects/{project_id}/datasets/{dataset_id}/download")
async def download_dataset(project_id: str, dataset_id: str):
    """下载数据集文件"""
    from fastapi.responses import FileResponse
    
    rows = execute_query('''
        SELECT file_path, name, file_type FROM datasets WHERE id = %s AND project_id = %s
    ''', (dataset_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    file_path = rows[0]['file_path']
    name = rows[0]['name']
    file_type = rows[0]['file_type']
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    filename = f"{name}.{file_type}"
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/csv'
    )
    


# 数据预处理与划分
@app.post("/api/projects/{project_id}/datasets/{dataset_id}/preprocess")
async def preprocess_dataset(
    project_id: str,
    dataset_id: str,
    config: TrainingConfig
):
    rows = execute_query('''
        SELECT file_path FROM datasets WHERE id = %s AND project_id = %s
    ''', (dataset_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    file_path = rows[0]['file_path']
    
    # 数据预处理逻辑
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        df = pd.read_csv(file_path)
        total = len(df)
        
        # 划分数据集
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - config.train_ratio),
            random_state=42,
            stratify=df.iloc[:, -1] if len(df.columns) > 1 else None
        )
        
        val_ratio_adjusted = config.val_ratio / (config.val_ratio + config.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            random_state=42,
            stratify=temp_df.iloc[:, -1] if len(temp_df.columns) > 1 else None
        )
        
        # 保存划分后的数据
        project_data_dir = DATA_DIR / project_id
        train_df.to_csv(project_data_dir / f"{dataset_id}_train.csv", index=False)
        val_df.to_csv(project_data_dir / f"{dataset_id}_val.csv", index=False)
        test_df.to_csv(project_data_dir / f"{dataset_id}_test.csv", index=False)
        
        # 更新数据库
        execute_update('''
            UPDATE datasets 
            SET train_samples = %s, val_samples = %s, test_samples = %s, status = 'preprocessed'
            WHERE id = %s
        ''', (len(train_df), len(val_df), len(test_df), dataset_id))
        
        return {
            "success": True,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预处理失败: {str(e)}")

# 标注管理
@app.post("/api/projects/{project_id}/annotations")
async def create_annotation_task(project_id: str, task: AnnotationTask):
    annotation_id = str(uuid.uuid4())
    
    execute_update('''
        INSERT INTO annotations (id, project_id, data_id, content, label, status)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (annotation_id, project_id, annotation_id, task.content, task.label, 'pending'))
    
    return {"success": True, "annotation_id": annotation_id}

@app.get("/api/projects/{project_id}/annotations")
async def list_annotations(project_id: str, status: Optional[str] = None):
    if status:
        rows = execute_query('''
            SELECT id, content, label, status, annotated_at 
            FROM annotations WHERE project_id = %s AND status = %s
        ''', (project_id, status))
    else:
        rows = execute_query('''
            SELECT id, content, label, status, annotated_at 
            FROM annotations WHERE project_id = %s
        ''', (project_id,))
    
    annotations = []
    for row in rows:
        annotations.append({
            "id": row['id'],
            "content": row['content'],
            "label": row['label'],
            "status": row['status'],
            "annotated_at": row['annotated_at']
        })
    
    return {"annotations": annotations}

@app.put("/api/projects/{project_id}/annotations/{annotation_id}")
async def update_annotation(project_id: str, annotation_id: str, label: str):
    execute_update('''
        UPDATE annotations 
        SET label = %s, status = 'completed', annotated_at = CURRENT_TIMESTAMP
        WHERE id = %s AND project_id = %s
    ''', (label, annotation_id, project_id))
    
    return {"success": True}

# 训练任务管理
@app.post("/api/projects/{project_id}/train")
async def start_training_v2(
    project_id: str,
    dataset_id: str = Form(...),
    config: str = Form(...),
    template: Optional[str] = Form(None),
    automl_trial_id: Optional[str] = Form(None)
):
    """启动训练任务 - 增强版（支持多种任务类型）"""
    
    config_dict = json.loads(config)
    
    # 如果指定了模板，合并配置
    if template and template in TRAINING_TEMPLATES:
        template_config = TRAINING_TEMPLATES[template]["config"].copy()
        template_config.update(config_dict)  # 用户配置覆盖模板
        config_dict = template_config
    
    # 获取项目信息（判断任务类型）
    rows = execute_query('SELECT task_type FROM projects WHERE id = %s', (project_id,))
    
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    project_task_type = rows[0]['task_type'] or 'text_classification'
    
    job_id = str(uuid.uuid4())
    log_path = LOGS_DIR / f"{job_id}.log"
    model_path = MODELS_DIR / project_id / f"{job_id}"
    
    # 确定模型名称显示（加时间戳区分同名任务）
    from datetime import datetime
    base_name = config_dict.get("model_name", config_dict.get("model_type", "unknown"))
    timestamp = datetime.now().strftime("%m%d-%H:%M")
    model_display = f"{base_name} ({timestamp})"
    
    # 如果来自AutoML，在模型名称中标记
    if automl_trial_id:
        model_display = f"[AutoML] {model_display}"
    
    execute_update('''
        INSERT INTO training_jobs (
            id, project_id, dataset_id, model_name, total_epochs, 
            log_path, model_path, config, status, automl_trial_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (
        job_id, project_id, dataset_id, model_display,
        config_dict.get("epochs", config_dict.get("n_estimators", 100)), 
        str(log_path), str(model_path),
        json.dumps(config_dict), 'pending', automl_trial_id
    ))
    
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 根据任务类型选择训练器
    if project_task_type in ['classification', 'regression', 'anomaly_detection']:
        # 结构化数据任务 - 使用ML训练器
        from tasks import submit_ml_training_task
        future = submit_ml_training_task(job_id, project_id, dataset_id, config_dict)
    elif project_task_type == 'image_classification':
        # 图像分类任务 - 使用PyTorch训练器
        from tasks import submit_image_training_task
        future = submit_image_training_task(job_id, project_id, dataset_id, config_dict)
    elif project_task_type == 'object_detection':
        # 目标检测任务 - 使用YOLOv8
        from tasks import submit_object_detection_task
        future = submit_object_detection_task(job_id, project_id, dataset_id, config_dict)
    else:
        # NLP任务 - 使用transformers训练器
        from tasks import submit_training_task
        future = submit_training_task(job_id, project_id, dataset_id, config_dict)
    
    return {
        "success": True,
        "job_id": job_id,
        "status": "started",
        "config": config_dict,
        "websocket_url": f"/ws/training/{job_id}"
    }

@app.get("/api/projects/{project_id}/jobs")
async def list_training_jobs(project_id: str):
    rows = execute_query('''
        SELECT id, model_name, status, progress, current_epoch, total_epochs,
               current_loss, best_accuracy, best_val_loss, learning_rate,
               early_stopped, stop_reason, created_at, started_at, completed_at
        FROM training_jobs WHERE project_id = %s ORDER BY created_at DESC
    ''', (project_id,))
    
    jobs = []
    for row in rows:
        jobs.append({
            "id": row['id'],
            "model_name": row['model_name'],
            "status": row['status'],
            "progress": row['progress'],
            "current_epoch": row['current_epoch'],
            "total_epochs": row['total_epochs'],
            "current_loss": row['current_loss'],
            "best_accuracy": row['best_accuracy'],
            "best_val_loss": row['best_val_loss'],
            "learning_rate": row['learning_rate'],
            "early_stopped": bool(row['early_stopped']) if row['early_stopped'] is not None else False,
            "stop_reason": row['stop_reason'],
            "created_at": row['created_at'],
            "started_at": row['started_at'],
            "completed_at": row['completed_at']
        })
    
    return {"jobs": jobs}

@app.post("/api/projects/{project_id}/jobs/{job_id}/retry")
async def retry_training_job(project_id: str, job_id: str):
    """重试失败的任务"""
    rows = execute_query('''
        SELECT id, dataset_id, config, status FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    job = rows[0]
    if job['status'] not in ['failed', 'pending']:
        raise HTTPException(status_code=400, detail="只有失败或等待中的任务可以重试")
    
    # 重置任务状态
    execute_update('''
        UPDATE training_jobs 
        SET status = 'pending', progress = 0, current_epoch = 0,
            current_loss = NULL, best_accuracy = NULL, best_val_loss = NULL,
            started_at = NULL, completed_at = NULL, error_message = NULL
        WHERE id = %s
    ''', (job_id,))
    
    # 重新提交任务
    config = job['config']
    if isinstance(config, str):
        config = json.loads(config)
    
    # 根据任务类型选择训练器
    project_rows = execute_query('SELECT task_type FROM projects WHERE id = %s', (project_id,))
    if not project_rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    task_type = project_rows[0]['task_type']
    
    if task_type in ['classification', 'regression', 'anomaly_detection']:
        from tasks import submit_ml_training_task
        submit_ml_training_task.delay(job_id, project_id, job['dataset_id'], config)
    elif task_type == 'image_classification':
        from tasks import submit_image_training_task
        submit_image_training_task.delay(job_id, project_id, job['dataset_id'], config)
    elif task_type == 'object_detection':
        from tasks import submit_object_detection_task
        submit_object_detection_task.delay(job_id, project_id, job['dataset_id'], config)
    else:
        from tasks import submit_training_task
        submit_training_task.delay(job_id, project_id, job['dataset_id'], config)
    
    return {"success": True, "message": "任务已重新提交"}

@app.delete("/api/projects/{project_id}/jobs/{job_id}")
async def delete_training_job(project_id: str, job_id: str):
    """删除训练任务"""
    rows = execute_query('''
        SELECT id, status FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    job = rows[0]
    if job['status'] == 'training':
        raise HTTPException(status_code=400, detail="训练中的任务不能删除")
    
    # 删除相关数据
    execute_update('DELETE FROM training_metrics WHERE job_id = %s', (job_id,))
    execute_update('DELETE FROM model_checkpoints WHERE job_id = %s', (job_id,))
    execute_update('DELETE FROM training_jobs WHERE id = %s', (job_id,))
    
    return {"success": True, "message": "任务已删除"}

@app.get("/api/projects/{project_id}/jobs/{job_id}/logs")
async def get_training_logs(project_id: str, job_id: str, lines: int = 50):
    rows = execute_query('''
        SELECT log_path FROM training_jobs WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows or not rows[0]['log_path']:
        return {"logs": []}
    
    log_path = Path(rows[0]['log_path'])
    if not log_path.exists():
        return {"logs": []}
    
    # 读取最后N行
    with open(log_path, 'r') as f:
        all_lines = f.readlines()
        return {"logs": all_lines[-lines:]}

# 导入推理服务
from inference_service import inference_service

# 模型部署
@app.post("/api/projects/{project_id}/jobs/{job_id}/deploy")
async def deploy_model(project_id: str, job_id: str, deploy_type: str = Form("api")):
    """部署模型为API服务或导出到Ollama"""
    rows = execute_query('''
        SELECT model_path, model_name, status FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    model_path = rows[0]['model_path']
    model_name = rows[0]['model_name']
    status = rows[0]['status']
    
    if status != 'completed':
        raise HTTPException(status_code=400, detail="训练尚未完成，无法部署")
    
    if deploy_type == "api":
        # API模式：加载模型到内存
        try:
            model_id = f"{project_id}/{job_id}"
            inference_service.load_model(model_path, model_id)
            
            # 更新部署状态到数据库
            execute_update('''
                UPDATE training_jobs SET deployed = 1 WHERE id = %s
            ''', (job_id,))
            
            return {
                "success": True,
                "deploy_type": "api",
                "model_id": model_id,
                "endpoint": f"/api/inference/{model_id}",
                "status": "模型已加载到内存",
                "loaded_models": inference_service.list_loaded_models()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"部署失败: {str(e)}")
    
    elif deploy_type == "ollama":
        # Ollama模式：导出分类模型为ONNX并创建Modelfile
        try:
            from pathlib import Path
            import json
            
            # 检查模型文件
            model_dir = Path(model_path)
            final_dir = model_dir / "final"
            label_mapping_path = model_dir / "label_mapping.json"
            
            if not final_dir.exists():
                raise HTTPException(status_code=404, detail="模型文件不存在")
            
            # 读取标签映射
            labels = []
            if label_mapping_path.exists():
                with open(label_mapping_path) as f:
                    label_map = json.load(f)
                    labels = [label_map['id2label'][str(i)] for i in range(len(label_map['id2label']))]
            
            # 导出为ONNX
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            onnx_path = model_dir / "model.onnx"
            
            # 加载模型
            tokenizer = AutoTokenizer.from_pretrained(str(final_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(final_dir))
            model.eval()
            
            # 创建dummy输入
            dummy_text = "这是一个测试文本"
            inputs = tokenizer(dummy_text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
            
            # 导出ONNX
            torch.onnx.export(
                model,
                (inputs['input_ids'], inputs['attention_mask']),
                str(onnx_path),
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size'}
                },
                opset_version=14
            )
            
            # 创建Ollama Modelfile
            modelfile_content = f'''FROM {onnx_path}

SYSTEM """
你是一个文本分类模型。输入一段文本，输出分类结果。
可用类别: {', '.join(labels)}

输出格式（JSON）:
{{"label": "预测类别", "confidence": 0.95, "all_probabilities": {{...}}}}
"""

PARAMETER temperature 0.1
'''
            
            modelfile_path = model_dir / "Modelfile"
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
            
            # 创建Python包装脚本（因为Ollama不直接支持BERT分类模型）
            wrapper_script = '''#!/usr/bin/env python3
# Ollama API 包装器 - 将BERT分类模型包装为Ollama兼容格式
import os
import sys
import json
import http.server
import socketserver
from pathlib import Path

# 添加backend目录到路径
sys.path.insert(0, '/var/www/ai-training/backend')

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("''' + str(model_dir) + '''")

class OllamaHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request = json.loads(post_data)
            
            prompt = request.get('prompt', '')
            
            # 加载模型（首次请求时）
            if not hasattr(self, 'model'):
                final_dir = MODEL_DIR / "final"
                label_mapping_path = MODEL_DIR / "label_mapping.json"
                
                self.tokenizer = AutoTokenizer.from_pretrained(str(final_dir))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(final_dir))
                self.model.eval()
                
                with open(label_mapping_path) as f:
                    label_map = json.load(f)
                self.id2label = {int(k): v for k, v in label_map['id2label'].items()}
            
            # 预测
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_id].item()
                all_probs = {self.id2label[i]: probs[0][i].item() for i in range(len(self.id2label))}
            
            result = {
                "label": self.id2label[pred_id],
                "confidence": round(confidence, 4),
                "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()}
            }
            
            response = {
                "response": json.dumps(result, ensure_ascii=False),
                "done": True
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        
        elif self.path == '/api/tags':
            # Ollama模型列表
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"models": []}).encode())
    
    def log_message(self, format, *args):
        # 静默日志
        pass

if __name__ == '__main__':
    PORT = int(os.environ.get('OLLAMA_PORT', 11435))
    with socketserver.TCPServer(("", PORT), OllamaHandler) as httpd:
        print(f"Ollama兼容服务启动于端口 {PORT}")
        httpd.serve_forever()
'''
            
            wrapper_path = model_dir / "ollama_server.py"
            with open(wrapper_path, 'w') as f:
                f.write(wrapper_script)
            
            os.chmod(wrapper_path, 0o755)
            
            return {
                "success": True,
                "deploy_type": "ollama",
                "message": "模型已导出为Ollama格式",
                "files": {
                    "onnx_model": str(onnx_path),
                    "modelfile": str(modelfile_path),
                    "wrapper_script": str(wrapper_path)
                },
                "instructions": [
                    f"1. 启动Ollama包装服务: python3 {wrapper_path}",
                    "2. 或使用Ollama导入: ollama create my-classifier -f " + str(modelfile_path),
                    "3. 测试: curl http://localhost:11435/api/generate -d '{\"prompt\":\"测试文本\"}'"
                ],
                "labels": labels
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ollama导出失败: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail="不支持的部署类型")


# 已部署模型列表
@app.get("/api/inference/models")
async def list_deployed_models():
    """列出已部署（已加载到内存）的模型"""
    models = inference_service.list_loaded_models()
    return {
        "deployed_models": models,
        "count": len(models)
    }


# 常驻内存预测API
@app.post("/api/inference/{project_id}/{job_id}")
async def inference_predict(project_id: str, job_id: str, text: str = Form(...)):
    """
    使用已部署的模型进行实时预测（模型常驻内存）
    
    优势：
    - 毫秒级响应（50-100ms）
    - 无需重复加载模型
    - 支持高并发
    """
    import time
    start_time = time.time()
    
    model_id = f"{project_id}/{job_id}"
    
    # 检查模型是否已加载
    if model_id not in inference_service.models:
        # 尝试自动加载
        rows = execute_query('''
            SELECT model_path, status FROM training_jobs 
            WHERE id = %s AND project_id = %s
        ''', (job_id, project_id))
        
        if not rows:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        model_path = rows[0]['model_path']
        status = rows[0]['status']
        if status != 'completed':
            raise HTTPException(status_code=400, detail="训练尚未完成")
        
        # 自动加载
        try:
            inference_service.load_model(model_path, model_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")
    
    # 执行预测
    try:
        result = inference_service.predict([text], model_id)[0]
        latency_ms = int((time.time() - start_time) * 1000)
        
        # 记录推理日志
        execute_update('''
            INSERT INTO inference_logs (project_id, job_id, model_id, input_data, output_data, latency_ms, success)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (project_id, job_id, model_id, json.dumps({"text": text}), json.dumps(result), latency_ms, True))
        
        # 记录 Prometheus 指标
        try:
            confidence = result.get('confidence', 0)
            metrics.record_prediction(
                model_type='nlp',
                latency=latency_ms/1000.0,
                confidence=confidence,
                success=True
            )
        except:
            pass
        
        return {
            "success": True,
            "model_id": model_id,
            "result": result,
            "latency_ms": latency_ms,
            "device": str(inference_service.device)
        }
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        
        # 记录错误日志
        execute_update('''
            INSERT INTO inference_logs (project_id, job_id, model_id, input_data, error_message, latency_ms, success)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (project_id, job_id, model_id, json.dumps({"text": text}), str(e), latency_ms, False))
        
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


# 批量预测API
@app.post("/api/inference/{project_id}/{job_id}/batch")
async def inference_predict_batch(project_id: str, job_id: str, texts: List[str]):
    """批量预测（模型常驻内存）"""
    model_id = f"{project_id}/{job_id}"
    
    if model_id not in inference_service.models:
        raise HTTPException(status_code=400, detail="模型未部署，请先调用部署接口")
    
    try:
        results = inference_service.predict(texts, model_id)
        return {
            "success": True,
            "model_id": model_id,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")


# 卸载模型
@app.delete("/api/inference/{project_id}/{job_id}")
async def undeploy_model(project_id: str, job_id: str):
    """卸载模型释放内存"""
    model_id = f"{project_id}/{job_id}"
    inference_service.unload_model(model_id)
    
    return {
        "success": True,
        "message": f"模型 {model_id} 已卸载",
        "remaining_models": inference_service.list_loaded_models()
    }

# 模型预测
@app.post("/api/predict/{project_id}/{job_id}")
async def predict(project_id: str, job_id: str, text: str = Form(...)):
    """使用训练好的模型进行预测"""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import json
    
    # 获取模型路径
    rows = execute_query('''
        SELECT model_path, status FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    model_path = rows[0]['model_path']
    status = rows[0]['status']
    if status != 'completed':
        raise HTTPException(status_code=400, detail="训练尚未完成")
    
    final_model_path = Path(model_path) / "final"
    label_mapping_path = Path(model_path) / "label_mapping.json"
    
    if not final_model_path.exists():
        raise HTTPException(status_code=404, detail="模型文件不存在")
    
    # 加载标签映射
    with open(label_mapping_path, 'r') as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map['id2label'].items()}
    
    # 加载模型和tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(str(final_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(final_model_path))
    model.to(device)
    model.eval()
    
    # 预测
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()
    
    # 构建所有类别的概率
    all_probs = {id2label[i]: probs[0][i].item() for i in range(len(id2label))}
    
    return {
        "prediction": id2label[pred_id],
        "confidence": round(confidence, 4),
        "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()},
        "input_text": text
    }

# ============ 增强版API：训练可视化与评估 ============

@app.get("/api/projects/{project_id}/jobs/{job_id}/metrics")
async def get_training_metrics_api(project_id: str, job_id: str, limit: int = 1000):
    """获取训练指标历史（用于绘图）"""
    # 验证任务存在
    rows = execute_query('SELECT id FROM training_jobs WHERE id = %s AND project_id = %s', (job_id, project_id))
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    # 获取指标历史
    rows = execute_query('''
        SELECT epoch, step, train_loss, val_loss, train_accuracy, val_accuracy, learning_rate, created_at
        FROM training_metrics
        WHERE job_id = %s
        ORDER BY step ASC
        LIMIT %s
    ''', (job_id, limit))
    
    metrics = []
    for row in rows:
        metrics.append({
            "epoch": row['epoch'],
            "step": row['step'],
            "train_loss": row['train_loss'],
            "val_loss": row['val_loss'],
            "train_accuracy": row['train_accuracy'],
            "val_accuracy": row['val_accuracy'],
            "learning_rate": row['learning_rate'],
            "created_at": row['created_at']
        })
    
    return {"metrics": metrics, "count": len(metrics)}


@app.get("/api/projects/{project_id}/jobs/{job_id}/report")
async def get_evaluation_report(project_id: str, job_id: str):
    """获取详细评估报告（混淆矩阵、F1/P/R等）"""
    rows = execute_query('''
        SELECT eval_report, best_accuracy, best_val_loss, early_stopped, stop_reason,
               current_epoch, total_epochs, model_path, status
        FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    row = rows[0]
    eval_report_json = row['eval_report']
    best_acc = row['best_accuracy']
    best_loss = row['best_val_loss']
    early_stopped = row['early_stopped']
    stop_reason = row['stop_reason']
    current_epoch = row['current_epoch']
    total_epochs = row['total_epochs']
    model_path = row['model_path']
    status = row['status']
    
    # 解析评估报告
    eval_report = eval_report_json if eval_report_json else {}
    
    return {
        "job_id": job_id,
        "status": status,
        "summary": {
            "best_accuracy": best_acc,
            "best_val_loss": best_loss,
            "epochs_trained": current_epoch,
            "total_epochs": total_epochs,
            "early_stopped": bool(early_stopped),
            "stop_reason": stop_reason
        },
        "evaluation": eval_report,
        "model_path": model_path
    }


@app.get("/api/projects/{project_id}/jobs/{job_id}/status")
async def get_job_status_api(project_id: str, job_id: str):
    """获取训练任务实时状态"""
    rows = execute_query('''
        SELECT id, model_name, status, progress, current_epoch, total_epochs,
               current_loss, best_accuracy, best_val_loss, learning_rate,
               early_stopped, stop_reason, created_at, started_at, completed_at
        FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    row = rows[0]
    return {
        "id": row['id'],
        "model_name": row['model_name'],
        "status": row['status'],
        "progress": row['progress'],
        "current_epoch": row['current_epoch'],
        "total_epochs": row['total_epochs'],
        "current_loss": row['current_loss'],
        "best_accuracy": row['best_accuracy'],
        "best_val_loss": row['best_val_loss'],
        "learning_rate": row['learning_rate'],
        "early_stopped": bool(row['early_stopped']),
        "stop_reason": row['stop_reason'],
        "created_at": row['created_at'],
        "started_at": row['started_at'],
        "completed_at": row['completed_at']
    }


@app.get("/api/projects/{project_id}/jobs/{job_id}/checkpoints")
async def list_model_checkpoints(project_id: str, job_id: str):
    """列出模型检查点"""
    rows = execute_query('''
        SELECT id, epoch, step, metric_name, metric_value, file_path, is_best, created_at
        FROM model_checkpoints
        WHERE job_id = %s
        ORDER BY created_at DESC
    ''', (job_id,))
    
    checkpoints = []
    for row in rows:
        checkpoints.append({
            "id": row['id'],
            "epoch": row['epoch'],
            "step": row['step'],
            "metric_name": row['metric_name'],
            "metric_value": row['metric_value'],
            "file_path": row['file_path'],
            "is_best": bool(row['is_best']),
            "created_at": row['created_at']
        })
    
    return {"checkpoints": checkpoints}


@app.get("/api/projects/{project_id}/jobs/{job_id}/prediction-analysis")
async def analyze_predictions(project_id: str, job_id: str, max_samples: int = 100):
    """
    预测结果深度分析 - 混淆矩阵 + 错误样本
    
    需要模型已完成训练，且数据集包含验证集
    """
    # 获取任务信息
    rows = execute_query('''
        SELECT j.*, d.file_path as dataset_path, d.labels as dataset_labels
        FROM training_jobs j
        LEFT JOIN datasets d ON j.dataset_id = d.id
        WHERE j.id = %s AND j.project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    job = rows[0]
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="模型训练尚未完成")
    
    model_path = job['model_path']
    dataset_path = job['dataset_path']
    
    if not dataset_path:
        raise HTTPException(status_code=400, detail="找不到关联的数据集")
    
    # 加载标签映射
    label_mapping_path = Path(model_path) / "label_mapping.json"
    if not label_mapping_path.exists():
        raise HTTPException(status_code=404, detail="找不到标签映射文件")
    
    with open(label_mapping_path) as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map['id2label'].items()}
    label2id = {v: int(k) for k, v in label_map['id2label'].items()}
    labels = [id2label[i] for i in range(len(id2label))]
    
    # 加载验证集
    import pandas as pd
    val_path = str(dataset_path).replace('.csv', '_val.csv')
    
    if not Path(val_path).exists():
        # 如果没有预划分的验证集，使用完整数据集
        val_path = dataset_path
    
    try:
        df = pd.read_csv(val_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法加载数据集: {e}")
    
    # 确定文本列和标签列
    text_col = df.columns[0] if 'text' not in df.columns else 'text'
    label_col = df.columns[-1] if 'label' not in df.columns else 'label'
    
    # 限制样本数
    df = df.head(max_samples)
    
    # 加载模型进行预测
    from inference_service import inference_service
    model_id = f"{project_id}/{job_id}"
    
    try:
        inference_service.load_model(model_path, model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型加载失败: {e}")
    
    # 批量预测
    texts = df[text_col].tolist()
    predictions = inference_service.predict(texts, model_id)
    
    # 构建混淆矩阵
    y_true = []
    y_pred = []
    error_samples = []
    
    for idx, (text, pred_result) in enumerate(zip(texts, predictions)):
        true_label = str(df.iloc[idx][label_col])
        pred_label = pred_result.get('prediction', 'unknown')
        confidence = pred_result.get('confidence', 0)
        
        # 转换为ID
        true_id = label2id.get(true_label, -1)
        pred_id = label2id.get(pred_label, -1)
        
        if true_id >= 0 and pred_id >= 0:
            y_true.append(true_id)
            y_pred.append(pred_id)
            
            # 记录错误样本
            if true_id != pred_id:
                error_samples.append({
                    "text": text[:200] + "..." if len(str(text)) > 200 else text,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "error_type": f"{true_label} → {pred_label}"
                })
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    
    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        
        # 格式化混淆矩阵
        cm_formatted = []
        for i, row in enumerate(cm):
            cm_formatted.append({
                "true_label": id2label[i],
                "predictions": {id2label[j]: int(count) for j, count in enumerate(row)}
            })
    else:
        cm_formatted = []
    
    # 计算每个类别的指标
    from sklearn.metrics import classification_report
    
    if len(y_true) > 0:
        report = classification_report(y_true, y_pred, 
                                     target_names=labels, 
                                     output_dict=True,
                                     zero_division=0)
        
        per_class_metrics = []
        for label in labels:
            if label in report:
                per_class_metrics.append({
                    "label": label,
                    "precision": report[label]['precision'],
                    "recall": report[label]['recall'],
                    "f1_score": report[label]['f1-score'],
                    "support": int(report[label]['support'])
                })
    else:
        per_class_metrics = []
    
    return {
        "analysis_summary": {
            "total_samples": len(texts),
            "correct_predictions": len(texts) - len(error_samples),
            "error_count": len(error_samples),
            "accuracy": (len(texts) - len(error_samples)) / len(texts) if texts else 0
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm_formatted
        },
        "per_class_metrics": per_class_metrics,
        "error_analysis": {
            "error_samples": error_samples[:20],  # 最多返回20个错误样本
            "error_patterns": {}
        }
    }


# ============ WebSocket 实时训练监控 ============
from fastapi import WebSocket, WebSocketDisconnect

class WebSocketManager:
    """WebSocket连接管理器"""
    def __init__(self):
        # job_id -> 一组WebSocket连接
        self.active_connections: Dict[str, set] = {}
    
    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)
    
    def disconnect(self, job_id: str, websocket: WebSocket):
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
    
    async def broadcast(self, job_id: str, message: dict):
        """广播消息给所有订阅该任务的客户端"""
        if job_id not in self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections[job_id]:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)
        
        # 清理断开的连接
        for conn in disconnected:
            self.active_connections[job_id].discard(conn)


# 创建全局WebSocket管理器
ws_manager = WebSocketManager()

# 设置到tasks模块（用于训练回调推送）
import tasks
tasks.set_websocket_manager(ws_manager)


# ============ 模型版本对比 API ============

@app.get("/api/projects/{project_id}/models/compare")
async def compare_models(
    project_id: str,
    job_ids: str = Query(..., description="逗号分隔的模型ID列表")
):
    """
    多模型版本对比 - 返回关键指标对比和训练曲线
    
    Args:
        job_ids: 逗号分隔的训练任务ID，如 "id1,id2,id3"
    
    Returns:
        对比数据，包括指标摘要、训练曲线、配置对比
    """
    ids = [j.strip() for j in job_ids.split(',') if j.strip()]
    
    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="至少需要选择2个模型进行对比")
    
    if len(ids) > 5:
        raise HTTPException(status_code=400, detail="最多支持5个模型同时对比")
    
    # 查询模型基本信息
    placeholders = ','.join(['%s'] * len(ids))
    rows = execute_query(f'''
        SELECT id, model_name, status, best_accuracy, best_val_loss,
               current_epoch, total_epochs, eval_report, config,
               early_stopped, stop_reason, created_at, completed_at
        FROM training_jobs 
        WHERE project_id = %s AND id IN ({placeholders})
        ORDER BY created_at DESC
    ''', (project_id, *ids))
    
    if not rows:
        raise HTTPException(status_code=404, detail="未找到指定的模型")
    
    models = []
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']  # 蓝绿黄红紫
    
    for idx, row in enumerate(rows):
        config = row['config'] if row['config'] else {}
        eval_report = row['eval_report'] if row['eval_report'] else {}
        
        models.append({
            "id": row['id'],
            "name": row['model_name'],
            "short_id": row['id'][:8],
            "color": colors[idx % len(colors)],
            "status": row['status'],
            "metrics": {
                "best_accuracy": row['best_accuracy'],
                "best_val_loss": row['best_val_loss'],
                "epochs_trained": row['current_epoch'],
                "total_epochs": row['total_epochs'],
                "early_stopped": bool(row['early_stopped']),
                "stop_reason": row['stop_reason']
            },
            "config": {
                "model_name": config.get('model_name', 'unknown'),
                "epochs": config.get('epochs'),
                "batch_size": config.get('batch_size'),
                "learning_rate": config.get('learning_rate'),
                "max_length": config.get('max_length'),
            },
            "eval_report": eval_report,
            "created_at": row['created_at'],
            "completed_at": row['completed_at']
        })
    
    # 获取训练曲线（用于对比图）
    curves = {}
    for model in models:
        job_id = model['id']
        metrics_rows = execute_query('''
            SELECT epoch, train_loss, val_loss, train_accuracy, val_accuracy
            FROM training_metrics
            WHERE job_id = %s
            ORDER BY epoch ASC
        ''', (job_id,))
        
        curves[model['id']] = {
            "epochs": [r['epoch'] for r in metrics_rows],
            "train_loss": [r['train_loss'] for r in metrics_rows],
            "val_loss": [r['val_loss'] for r in metrics_rows],
            "train_acc": [r['train_accuracy'] for r in metrics_rows],
            "val_acc": [r['val_accuracy'] for r in metrics_rows]
        }
    
    # 计算排名
    rankings = []
    completed_models = [m for m in models if m['status'] == 'completed']
    
    if completed_models:
        # 按准确率排名
        by_accuracy = sorted(completed_models, 
                           key=lambda x: x['metrics']['best_accuracy'] or 0, 
                           reverse=True)
        
        # 按损失排名
        by_loss = sorted(completed_models, 
                        key=lambda x: x['metrics']['best_val_loss'] or float('inf'))
        
        # 按训练速度排名（epochs/耗时）
        by_speed = sorted(completed_models,
                         key=lambda x: (x['metrics']['epochs_trained'] or 0))
        
        rankings = {
            "best_accuracy": {"model_id": by_accuracy[0]['id'], "value": by_accuracy[0]['metrics']['best_accuracy']} if by_accuracy else None,
            "lowest_loss": {"model_id": by_loss[0]['id'], "value": by_loss[0]['metrics']['best_val_loss']} if by_loss else None,
            "fastest_convergence": {"model_id": by_speed[0]['id'], "epochs": by_speed[0]['metrics']['epochs_trained']} if by_speed else None
        }
    
    return {
        "models": models,
        "curves": curves,
        "rankings": rankings,
        "comparison_count": len(models)
    }


@app.post("/api/projects/{project_id}/models/batch-evaluate")
async def batch_evaluate_models(
    project_id: str,
    job_ids: str = Form(...),
    test_texts: str = Form(..., description="JSON数组格式的测试文本列表")
):
    """
    批量评估多个模型 - 在同一批测试数据上对比模型表现
    
    Args:
        job_ids: 逗号分隔的模型ID
        test_texts: JSON数组，如 ["文本1", "文本2", "文本3"]
    
    Returns:
        每个模型的预测结果对比
    """
    import json
    
    ids = [j.strip() for j in job_ids.split(',') if j.strip()]
    texts = json.loads(test_texts)
    
    if not texts:
        raise HTTPException(status_code=400, detail="测试文本不能为空")
    
    # 加载所有模型
    results = []
    for job_id in ids:
        # 加载模型
        from inference_service import inference_service
        
        rows = execute_query('''
            SELECT model_path, status FROM training_jobs 
            WHERE id = %s AND project_id = %s
        ''', (job_id, project_id))
        
        if not rows or rows[0]['status'] != 'completed':
            continue
        
        model_path = rows[0]['model_path']
        model_id = f"{project_id}/{job_id}"
        
        try:
            inference_service.load_model(model_path, model_id)
            predictions = inference_service.predict(texts, model_id)
            
            results.append({
                "model_id": job_id,
                "predictions": predictions
            })
        except Exception as e:
            results.append({
                "model_id": job_id,
                "error": str(e)
            })
    
    # 组织为对比格式
    comparison = []
    for i, text in enumerate(texts):
        item = {"text": text, "predictions": {}}
        for r in results:
            if "error" not in r and i < len(r["predictions"]):
                item["predictions"][r["model_id"]] = r["predictions"][i]
        comparison.append(item)
    
    return {
        "test_samples": len(texts),
        "models_evaluated": len(results),
        "comparison": comparison
    }


@app.websocket("/ws/training/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str):
    """WebSocket端点：实时接收训练更新"""
    await ws_manager.connect(job_id, websocket)
    
    try:
        # 发送初始状态
        rows = execute_query('''
            SELECT status, progress, current_epoch, total_epochs, current_loss, best_accuracy
            FROM training_jobs WHERE id = %s
        ''', (job_id,))
        
        if rows:
            row = rows[0]
            await websocket.send_json({
                "type": "init",
                "status": row['status'],
                "progress": row['progress'],
                "current_epoch": row['current_epoch'],
                "total_epochs": row['total_epochs'],
                "current_loss": row['current_loss'],
                "best_accuracy": row['best_accuracy']
            })
        
        # 保持连接，接收客户端消息（如暂停、取消等）
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")
                
                if action == "ping":
                    await websocket.send_json({"type": "pong"})
                # 可以扩展：pause, resume, stop 等控制命令
                
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, websocket)


# ============ 训练配置模板 ============

TRAINING_TEMPLATES = {
    "fast": {
        "name": "快速训练",
        "description": "适合快速验证，牺牲一定精度换取速度",
        "config": {
            "model_name": "distilbert-base-chinese",
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 5e-5,
            "max_length": 128,
            "early_stopping": True,
            "early_stopping_patience": 2
        }
    },
    "balanced": {
        "name": "均衡配置",
        "description": "速度与精度的平衡，推荐日常使用",
        "config": {
            "model_name": "bert-base-chinese",
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "max_length": 256,
            "early_stopping": True,
            "early_stopping_patience": 3
        }
    },
    "accurate": {
        "name": "高精度",
        "description": "追求最高精度，训练时间较长",
        "config": {
            "model_name": "chinese-roberta-wwm-ext",
            "epochs": 10,
            "batch_size": 8,
            "learning_rate": 1e-5,
            "max_length": 512,
            "early_stopping": True,
            "early_stopping_patience": 5
        }
    },
    "tiny": {
        "name": "超轻量",
        "description": "极小模型，适合边缘设备",
        "config": {
            "model_name": "tiny-bert",
            "epochs": 5,
            "batch_size": 64,
            "learning_rate": 5e-5,
            "max_length": 128,
            "early_stopping": False
        }
    }
}


@app.get("/api/training-templates")
async def get_training_templates():
    """获取预设训练配置模板"""
    return {"templates": TRAINING_TEMPLATES}


# ============ 修改训练启动接口（支持新配置）============

@app.post("/api/projects/{project_id}/train")
async def start_training_v2(
    project_id: str,
    dataset_id: str = Form(...),
    config: str = Form(...),
    template: Optional[str] = Form(None)
):
    """启动训练任务 - 增强版（支持模板）"""
    
    config_dict = json.loads(config)
    
    # 如果指定了模板，合并配置
    if template and template in TRAINING_TEMPLATES:
        template_config = TRAINING_TEMPLATES[template]["config"].copy()
        template_config.update(config_dict)  # 用户配置覆盖模板
        config_dict = template_config
    
    job_id = str(uuid.uuid4())
    log_path = LOGS_DIR / f"{job_id}.log"
    model_path = MODELS_DIR / project_id / f"{job_id}"
    
    execute_update('''
        INSERT INTO training_jobs (
            id, project_id, dataset_id, model_name, total_epochs, 
            log_path, model_path, config, status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (
        job_id, project_id, dataset_id, config_dict.get("model_name", "bert-base-chinese"),
        config_dict.get("epochs", 3), str(log_path), str(model_path),
        json.dumps(config_dict), 'pending'
    ))
    
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 使用新版线程池任务（替代Celery）
    from tasks import submit_training_task
    future = submit_training_task(job_id, project_id, dataset_id, config_dict)
    
    return {
        "success": True,
        "job_id": job_id,
        "status": "started",
        "config": config_dict,
        "websocket_url": f"/ws/training/{job_id}"
    }


# ============ 批量预测API ============

@app.post("/api/projects/{project_id}/jobs/{job_id}/batch-predict")
async def batch_predict(
    project_id: str,
    job_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    批量预测 - 上传文件进行预测
    
    支持CSV/Excel文件，返回带预测结果的文件
    """
    # 获取任务信息
    rows = execute_query('''
        SELECT model_path, status, config FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    model_path = rows[0]['model_path']
    status = rows[0]['status']
    config_json = rows[0]['config']
    
    if status != 'completed':
        raise HTTPException(status_code=400, detail="训练尚未完成")
    
    # 解析配置获取模型类型
    config = json.loads(config_json) if config_json else {}
    task_type = config.get('task_type', 'text_classification')
    model_type = 'ml' if task_type in ['classification', 'regression', 'anomaly_detection'] else 'nlp'
    
    # 保存上传的文件
    temp_dir = Path("/tmp/ai-training/batch")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = temp_dir / f"{job_id}_input_{file.filename}"
    output_path = temp_dir / f"{job_id}_predictions.csv"
    
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 加载模型并预测
    model_id = f"{project_id}/{job_id}"
    try:
        inference_service.load_model(model_path, model_id, model_type)
        result = inference_service.predict_file(str(input_path), model_id, str(output_path))
        
        # 返回结果文件
        return FileResponse(
            path=output_path,
            filename=f"predictions_{file.filename}",
            media_type='text/csv'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


# ============ 内存监控API ============

@app.get("/api/system/memory")
async def get_memory_status():
    """获取系统内存和模型加载状态"""
    status = inference_service.get_memory_status()
    return status


@app.post("/api/inference/cleanup")
async def cleanup_memory(keep_models: List[str] = None):
    """
    手动清理内存
    
    Args:
        keep_models: 要保留的模型ID列表（可选）
    """
    status_before = inference_service.get_memory_status()
    
    # 卸载不需要的模型
    loaded_models = inference_service.list_loaded_models()
    for model in loaded_models:
        if not keep_models or model['model_id'] not in keep_models:
            inference_service.unload_model(model['model_id'])
    
    status_after = inference_service.get_memory_status()
    
    return {
        "success": True,
        "models_unloaded": len(loaded_models) - len(inference_service.list_loaded_models()),
        "memory_before": f"{status_before['memory_percent']:.1f}%",
        "memory_after": f"{status_after['memory_percent']:.1f}%",
        "memory_freed_gb": status_before['memory_used_gb'] - status_after['memory_used_gb']
    }


@app.post("/api/inference/unload")
async def unload_model_api(request: Dict):
    """卸载指定模型"""
    model_id = request.get('model_id')
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    
    inference_service.unload_model(model_id)
    return {"success": True, "message": f"模型 {model_id} 已卸载"}


@app.post("/api/projects/{project_id}/jobs/{job_id}/load")
async def load_model_for_inference(project_id: str, job_id: str):
    """加载模型到内存用于推理"""
    # 获取任务信息
    rows = execute_query('''
        SELECT model_path, model_name, status, model_type FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    row = rows[0]
    model_path = row['model_path']
    status = row['status']
    model_type = row.get('model_type', 'nlp')  # 默认为nlp
    
    if status != 'completed':
        raise HTTPException(status_code=400, detail="训练尚未完成")
    
    # 加载模型到推理服务
    try:
        model_id = f"{project_id}/{job_id}"
        inference_service.load_model(model_path, model_id, model_type)
        return {"success": True, "message": "模型已加载到内存", "model_id": model_id}
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@app.get("/api/projects/{project_id}/inference/stats")
async def get_inference_stats(project_id: str):
    """获取项目推理统计信息"""
    # 从推理日志中获取统计
    rows = execute_query('''
        SELECT COUNT(*) as total_calls,
               AVG(CASE WHEN success THEN latency_ms ELSE NULL END) as avg_latency_ms,
               SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
               SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as error_count
        FROM inference_logs 
        WHERE project_id = %s
    ''', (project_id,))
    
    stats = rows[0] if rows else {}
    total_calls = stats.get('total_calls', 0) or 0
    success_count = stats.get('success_count', 0) or 0
    error_count = stats.get('error_count', 0) or 0
    
    # 计算成功率
    success_rate = round((success_count / total_calls * 100), 2) if total_calls > 0 else 100
    
    # 获取已加载模型数
    loaded_models = inference_service.list_loaded_models()
    project_models = [m for m in loaded_models if m['model_id'].startswith(f"{project_id}/")]
    
    return {
        "totalCalls": total_calls,
        "avgLatency": round(stats.get('avg_latency_ms', 0) or 0, 2),
        "successRate": success_rate,
        "errorCount": error_count,
        "loadedModels": len(project_models)
    }


@app.get("/api/projects/{project_id}/inference/trends")
async def get_inference_trends(project_id: str, hours: int = 24):
    """获取推理调用趋势（按小时）"""
    rows = execute_query('''
        SELECT 
            date_trunc('hour', created_at) as hour,
            COUNT(*) as call_count,
            AVG(CASE WHEN success THEN latency_ms ELSE NULL END) as avg_latency,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
            SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as error_count
        FROM inference_logs 
        WHERE project_id = %s 
          AND created_at >= NOW() - INTERVAL '%s hours'
        GROUP BY date_trunc('hour', created_at)
        ORDER BY hour DESC
        LIMIT %s
    ''', (project_id, hours, hours))
    
    trends = []
    for row in rows:
        trends.append({
            "hour": row['hour'].isoformat() if row['hour'] else None,
            "callCount": row['call_count'],
            "avgLatency": round(row['avg_latency'] or 0, 2),
            "successCount": row['success_count'],
            "errorCount": row['error_count']
        })
    
    return {"trends": trends, "hours": hours}


@app.get("/api/projects/{project_id}/inference/latency-distribution")
async def get_latency_distribution(project_id: str):
    """获取延迟分布统计"""
    rows = execute_query('''
        SELECT 
            CASE 
                WHEN latency_ms < 50 THEN '0-50ms'
                WHEN latency_ms < 100 THEN '50-100ms'
                WHEN latency_ms < 200 THEN '100-200ms'
                WHEN latency_ms < 500 THEN '200-500ms'
                WHEN latency_ms < 1000 THEN '500ms-1s'
                ELSE '>1s'
            END as latency_range,
            COUNT(*) as count
        FROM inference_logs 
        WHERE project_id = %s AND success = TRUE
        GROUP BY 
            CASE 
                WHEN latency_ms < 50 THEN '0-50ms'
                WHEN latency_ms < 100 THEN '50-100ms'
                WHEN latency_ms < 200 THEN '100-200ms'
                WHEN latency_ms < 500 THEN '200-500ms'
                WHEN latency_ms < 1000 THEN '500ms-1s'
                ELSE '>1s'
            END
        ORDER BY count DESC
    ''', (project_id,))
    
    distribution = []
    for row in rows:
        distribution.append({
            "range": row['latency_range'],
            "count": row['count']
        })
    
    return {"distribution": distribution}


@app.get("/api/projects/{project_id}/inference/recent-logs")
async def get_recent_inference_logs(project_id: str, limit: int = 50):
    """获取最近的推理日志"""
    rows = execute_query('''
        SELECT 
            il.id,
            il.model_id,
            il.latency_ms,
            il.success,
            il.error_message,
            il.created_at,
            tj.model_name
        FROM inference_logs il
        LEFT JOIN training_jobs tj ON il.job_id = tj.id
        WHERE il.project_id = %s
        ORDER BY il.created_at DESC
        LIMIT %s
    ''', (project_id, limit))
    
    logs = []
    for row in rows:
        logs.append({
            "id": row['id'],
            "modelId": row['model_id'],
            "modelName": row['model_name'] or 'Unknown',
            "latencyMs": row['latency_ms'],
            "success": row['success'],
            "errorMessage": row['error_message'],
            "createdAt": row['created_at'].isoformat() if row['created_at'] else None
        })
    
    return {"logs": logs}


# ============ 推理服务告警API ============

@app.get("/api/projects/{project_id}/inference/alert-rules")
async def get_inference_alert_rules(project_id: str):
    """获取推理服务告警规则"""
    rows = execute_query('''
        SELECT id, name, rule_type, threshold_value, time_window_minutes, enabled, notify_channels
        FROM inference_alert_rules
        WHERE project_id = %s
        ORDER BY created_at DESC
    ''', (project_id,))
    
    rules = []
    for row in rows:
        rules.append({
            "id": row['id'],
            "name": row['name'],
            "ruleType": row['rule_type'],
            "thresholdValue": row['threshold_value'],
            "timeWindowMinutes": row['time_window_minutes'],
            "enabled": row['enabled'],
            "notifyChannels": row['notify_channels']
        })
    
    return {"rules": rules}


@app.post("/api/projects/{project_id}/inference/alert-rules")
async def create_inference_alert_rule(
    project_id: str,
    name: str = Form(...),
    rule_type: str = Form(...),
    threshold_value: float = Form(...),
    time_window_minutes: int = Form(5),
    notify_channels: str = Form('["web"]')
):
    """创建推理服务告警规则"""
    rule_id = str(uuid.uuid4())
    
    execute_update('''
        INSERT INTO inference_alert_rules 
        (id, project_id, name, rule_type, threshold_value, time_window_minutes, notify_channels)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    ''', (rule_id, project_id, name, rule_type, threshold_value, time_window_minutes, notify_channels))
    
    return {"success": True, "id": rule_id, "message": "告警规则已创建"}


@app.delete("/api/projects/{project_id}/inference/alert-rules/{rule_id}")
async def delete_inference_alert_rule(project_id: str, rule_id: str):
    """删除推理服务告警规则"""
    execute_update('''
        DELETE FROM inference_alert_rules 
        WHERE id = %s AND project_id = %s
    ''', (rule_id, project_id))
    
    return {"success": True, "message": "告警规则已删除"}


@app.post("/api/projects/{project_id}/inference/alert-rules/{rule_id}/toggle")
async def toggle_inference_alert_rule(project_id: str, rule_id: str):
    """启用/禁用告警规则"""
    execute_update('''
        UPDATE inference_alert_rules 
        SET enabled = NOT enabled
        WHERE id = %s AND project_id = %s
    ''', (rule_id, project_id))
    
    return {"success": True, "message": "告警规则状态已更新"}


@app.get("/api/projects/{project_id}/inference/alerts")
async def get_inference_alerts(project_id: str, status: str = None, limit: int = 50):
    """获取推理服务告警记录"""
    if status:
        rows = execute_query('''
            SELECT id, alert_type, severity, title, message, metric_value, threshold_value,
                   status, acknowledged_by, acknowledged_at, resolved_at, created_at
            FROM inference_alerts
            WHERE project_id = %s AND status = %s
            ORDER BY created_at DESC
            LIMIT %s
        ''', (project_id, status, limit))
    else:
        rows = execute_query('''
            SELECT id, alert_type, severity, title, message, metric_value, threshold_value,
                   status, acknowledged_by, acknowledged_at, resolved_at, created_at
            FROM inference_alerts
            WHERE project_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        ''', (project_id, limit))
    
    alerts = []
    for row in rows:
        alerts.append({
            "id": row['id'],
            "alertType": row['alert_type'],
            "severity": row['severity'],
            "title": row['title'],
            "message": row['message'],
            "metricValue": row['metric_value'],
            "thresholdValue": row['threshold_value'],
            "status": row['status'],
            "acknowledgedBy": row['acknowledged_by'],
            "acknowledgedAt": row['acknowledged_at'].isoformat() if row['acknowledged_at'] else None,
            "resolvedAt": row['resolved_at'].isoformat() if row['resolved_at'] else None,
            "createdAt": row['created_at'].isoformat() if row['created_at'] else None
        })
    
    return {"alerts": alerts}


@app.post("/api/inference/alerts/{alert_id}/acknowledge")
async def acknowledge_inference_alert(alert_id: int, user: str = Form("admin")):
    """确认告警"""
    execute_update('''
        UPDATE inference_alerts 
        SET status = 'acknowledged', acknowledged_by = %s, acknowledged_at = NOW()
        WHERE id = %s
    ''', (user, alert_id))
    
    return {"success": True, "message": "告警已确认"}


@app.post("/api/inference/alerts/{alert_id}/resolve")
async def resolve_inference_alert(alert_id: int):
    """解决告警"""
    execute_update('''
        UPDATE inference_alerts 
        SET status = 'resolved', resolved_at = NOW()
        WHERE id = %s
    ''', (alert_id,))
    
    return {"success": True, "message": "告警已解决"}


@app.post("/api/projects/{project_id}/inference/check-alerts")
async def check_inference_alerts(project_id: str):
    """
    检查推理服务告警条件
    根据配置的规则检查最近的推理日志，触发告警
    """
    # 获取启用的告警规则
    rules = execute_query('''
        SELECT id, name, rule_type, threshold_value, time_window_minutes
        FROM inference_alert_rules
        WHERE project_id = %s AND enabled = TRUE
    ''', (project_id,))
    
    triggered_alerts = []
    
    for rule in rules:
        rule_id = rule['id']
        rule_type = rule['rule_type']
        threshold = rule['threshold_value']
        window_minutes = rule['time_window_minutes']
        
        # 检查是否已有未解决的相同类型告警
        existing_alert = execute_query('''
            SELECT id FROM inference_alerts
            WHERE project_id = %s AND rule_id = %s AND status = 'active'
        ''', (project_id, rule_id))
        
        if existing_alert:
            continue  # 已有活跃告警，跳过
        
        # 根据规则类型检查
        if rule_type == 'error_rate':
            # 检查错误率
            stats = execute_query('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as errors
                FROM inference_logs
                WHERE project_id = %s AND created_at >= NOW() - INTERVAL '%s minutes'
            ''', (project_id, window_minutes))
            
            if stats and stats[0]['total'] > 0:
                error_rate = (stats[0]['errors'] / stats[0]['total']) * 100
                if error_rate >= threshold:
                    alert_id = execute_inference_alert(
                        project_id, rule_id, 'error_rate', 'critical',
                        f"推理错误率过高: {error_rate:.1f}%",
                        f"最近{window_minutes}分钟内错误率达到{error_rate:.1f}%，超过阈值{threshold}%",
                        error_rate, threshold
                    )
                    triggered_alerts.append({"type": "error_rate", "value": error_rate})
        
        elif rule_type == 'latency':
            # 检查平均延迟
            stats = execute_query('''
                SELECT AVG(latency_ms) as avg_latency
                FROM inference_logs
                WHERE project_id = %s AND success = TRUE
                  AND created_at >= NOW() - INTERVAL '%s minutes'
            ''', (project_id, window_minutes))
            
            if stats and stats[0]['avg_latency']:
                avg_latency = stats[0]['avg_latency']
                if avg_latency >= threshold:
                    alert_id = execute_inference_alert(
                        project_id, rule_id, 'latency', 'warning',
                        f"推理延迟过高: {avg_latency:.0f}ms",
                        f"最近{window_minutes}分钟内平均延迟达到{avg_latency:.0f}ms，超过阈值{threshold}ms",
                        avg_latency, threshold
                    )
                    triggered_alerts.append({"type": "latency", "value": avg_latency})
        
        elif rule_type == 'memory':
            # 检查内存使用率
            memory_status = inference_service.get_memory_status()
            memory_percent = memory_status['memory_percent']
            
            if memory_percent >= threshold:
                alert_id = execute_inference_alert(
                    project_id, rule_id, 'memory', 'critical' if memory_percent > 90 else 'warning',
                    f"系统内存使用率过高: {memory_percent:.1f}%",
                    f"当前内存使用率{memory_percent:.1f}%，超过阈值{threshold}%",
                    memory_percent, threshold
                )
                triggered_alerts.append({"type": "memory", "value": memory_percent})
    
    return {
        "success": True,
        "checked_rules": len(rules),
        "triggered_alerts": triggered_alerts
    }


def execute_inference_alert(project_id: str, rule_id: str, alert_type: str, severity: str,
                            title: str, message: str, metric_value: float, threshold_value: float):
    """执行告警：写入数据库并发送通知"""
    # 写入告警记录
    rows = execute_query('''
        INSERT INTO inference_alerts 
        (project_id, rule_id, alert_type, severity, title, message, metric_value, threshold_value)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    ''', (project_id, rule_id, alert_type, severity, title, message, metric_value, threshold_value))
    
    alert_id = rows[0]['id'] if rows else None
    
    # TODO: 发送飞书通知（如果需要）
    
    return alert_id


@app.get("/api/projects/{project_id}/inference/recommendations")
async def get_inference_recommendations(project_id: str):
    """
    获取推理服务优化建议
    基于历史数据分析给出运营优化建议
    """
    recommendations = []
    
    # 1. 检查模型冷热情况
    hot_models = execute_query('''
        SELECT model_id, COUNT(*) as call_count
        FROM inference_logs
        WHERE project_id = %s AND created_at >= NOW() - INTERVAL '1 hour'
        GROUP BY model_id
        ORDER BY call_count DESC
        LIMIT 5
    ''', (project_id,))
    
    if hot_models:
        top_model = hot_models[0]
        if top_model['call_count'] > 100:
            recommendations.append({
                "type": "hot_model",
                "priority": "high",
                "title": "热点模型建议常驻内存",
                "description": f"模型 {top_model['model_id']} 最近1小时被调用{top_model['call_count']}次，建议保持加载状态以优化响应时间",
                "action": "保持加载"
            })
    
    # 2. 检查冷模型（已加载但很少使用）
    loaded_models = inference_service.list_loaded_models()
    for model in loaded_models:
        model_id = model['model_id']
        if not model_id.startswith(f"{project_id}/"):
            continue
        
        recent_calls = execute_query('''
            SELECT COUNT(*) as count
            FROM inference_logs
            WHERE project_id = %s AND model_id = %s
              AND created_at >= NOW() - INTERVAL '30 minutes'
        ''', (project_id, model_id))
        
        if recent_calls and recent_calls[0]['count'] == 0:
            recommendations.append({
                "type": "cold_model",
                "priority": "medium",
                "title": "冷模型建议卸载",
                "description": f"模型 {model_id} 已加载但30分钟内无调用，建议手动卸载释放内存",
                "action": "卸载模型"
            })
    
    # 3. 延迟优化建议
    latency_stats = execute_query('''
        SELECT AVG(latency_ms) as avg_latency, COUNT(*) as count
        FROM inference_logs
        WHERE project_id = %s AND success = TRUE
          AND created_at >= NOW() - INTERVAL '1 hour'
    ''', (project_id,))
    
    if latency_stats and latency_stats[0]['avg_latency']:
        avg_latency = latency_stats[0]['avg_latency']
        if avg_latency > 500:
            recommendations.append({
                "type": "latency",
                "priority": "high",
                "title": "推理延迟较高",
                "description": f"最近1小时平均延迟{avg_latency:.0f}ms，建议检查模型大小或升级硬件",
                "action": "优化模型"
            })
    
    # 4. 错误率建议
    error_stats = execute_query('''
        SELECT 
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
            SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as error_count
        FROM inference_logs
        WHERE project_id = %s AND created_at >= NOW() - INTERVAL '1 hour'
    ''', (project_id,))
    
    if error_stats:
        total = (error_stats[0]['success_count'] or 0) + (error_stats[0]['error_count'] or 0)
        errors = error_stats[0]['error_count'] or 0
        if total > 0 and errors / total > 0.05:
            recommendations.append({
                "type": "error_rate",
                "priority": "critical",
                "title": "错误率过高需要关注",
                "description": f"最近1小时错误率达到{errors/total*100:.1f}%，建议检查模型状态或输入数据格式",
                "action": "检查日志"
            })
    
    return {"recommendations": recommendations}


# ============ 飞书通知配置API ============

@app.get("/api/projects/{project_id}/feishu-notification")
async def get_feishu_notification(project_id: str):
    """获取飞书通知配置"""
    rows = execute_query('''
        SELECT id, webhook_url, notify_on_success, notify_on_failure
        FROM feishu_notifications WHERE project_id = %s
    ''', (project_id,))
    
    if not rows:
        return {"enabled": False}
    
    row = rows[0]
    return {
        "enabled": True,
        "id": row['id'],
        "webhook_url": row['webhook_url'],
        "notify_on_success": bool(row['notify_on_success']),
        "notify_on_failure": bool(row['notify_on_failure'])
    }


@app.post("/api/projects/{project_id}/feishu-notification")
async def set_feishu_notification(
    project_id: str,
    webhook_url: str = Form(...),
    notify_on_success: bool = Form(True),
    notify_on_failure: bool = Form(True)
):
    """设置飞书通知配置"""
    # 检查是否已存在
    rows = execute_query('SELECT id FROM feishu_notifications WHERE project_id = %s', (project_id,))
    
    if rows:
        # 更新
        execute_update('''
            UPDATE feishu_notifications
            SET webhook_url = %s, notify_on_success = %s, notify_on_failure = %s
            WHERE project_id = %s
        ''', (webhook_url, notify_on_success, notify_on_failure, project_id))
    else:
        # 创建
        notification_id = str(uuid.uuid4())
        execute_update('''
            INSERT INTO feishu_notifications (id, project_id, webhook_url, notify_on_success, notify_on_failure)
            VALUES (%s, %s, %s, %s, %s)
        ''', (notification_id, project_id, webhook_url, notify_on_success, notify_on_failure))
    
    return {"success": True, "message": "飞书通知配置已保存"}


@app.delete("/api/projects/{project_id}/feishu-notification")
async def delete_feishu_notification(project_id: str):
    """删除飞书通知配置"""
    execute_update('DELETE FROM feishu_notifications WHERE project_id = %s', (project_id,))
    
    return {"success": True, "message": "飞书通知配置已删除"}


# ============ 模型版本管理API ============

@app.get("/api/projects/{project_id}/models")
async def list_model_versions(project_id: str):
    """列出项目的所有模型版本"""
    rows = execute_query('''
        SELECT id, model_name, status, best_accuracy, best_val_loss,
               current_epoch, total_epochs, model_path, config, created_at, completed_at
        FROM training_jobs 
        WHERE project_id = %s AND status = 'completed'
        ORDER BY created_at DESC
    ''', (project_id,))
    
    models = []
    for row in rows:
        config = row['config'] if row['config'] else {}
        
        models.append({
            "id": row['id'],
            "name": row['model_name'],
            "status": row['status'],
            "best_accuracy": row['best_accuracy'],
            "best_val_loss": row['best_val_loss'],
            "epochs_trained": row['current_epoch'],
            "total_epochs": row['total_epochs'],
            "model_path": row['model_path'],
            "config": config,
            "created_at": row['created_at'],
            "completed_at": row['completed_at'],
            "is_deployed": f"{project_id}/{row['id']}" in [m['model_id'] for m in inference_service.list_loaded_models()]
        })
    
    return {"models": models, "count": len(models)}


@app.post("/api/projects/{project_id}/models/{job_id}/deploy")
async def deploy_model_version(
    project_id: str,
    job_id: str,
    deploy_type: str = Form("api")
):
    """部署指定版本的模型"""
    rows = execute_query('''
        SELECT model_path, config, status FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    model_path = rows[0]['model_path']
    config_json = rows[0]['config']
    status = rows[0]['status']
    
    if status != 'completed':
        raise HTTPException(status_code=400, detail="训练尚未完成")
    
    # 解析配置获取模型类型
    config = json.loads(config_json) if config_json else {}
    task_type = config.get('task_type', 'text_classification')
    model_type = 'ml' if task_type in ['classification', 'regression', 'anomaly_detection'] else 'nlp'
    
    # 加载模型
    model_id = f"{project_id}/{job_id}"
    try:
        inference_service.load_model(model_path, model_id, model_type)
        
        return {
            "success": True,
            "model_id": model_id,
            "model_type": model_type,
            "task_type": task_type,
            "endpoint": f"/api/inference/{model_id}",
            "status": "模型已加载到内存"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"部署失败: {str(e)}")


@app.delete("/api/projects/{project_id}/models/{job_id}")
async def delete_model_version(project_id: str, job_id: str):
    """删除模型版本（从数据库和文件系统）"""
    # 获取模型路径
    rows = execute_query('''
        SELECT model_path FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    model_path = rows[0]['model_path']
    
    # 如果模型已加载，先卸载
    model_id = f"{project_id}/{job_id}"
    if model_id in [m['model_id'] for m in inference_service.list_loaded_models()]:
        inference_service.unload_model(model_id)
    
    # 删除模型文件
    import shutil
    if Path(model_path).exists():
        shutil.rmtree(model_path)
    
    # 删除数据库记录
    execute_update('DELETE FROM training_jobs WHERE id = %s', (job_id,))
    execute_update('DELETE FROM training_metrics WHERE job_id = %s', (job_id,))
    
    return {"success": True, "message": f"模型 {job_id} 已删除"}


@app.get("/api/projects/{project_id}/models/compare")
async def compare_models(project_id: str, model_ids: str = None):
    """对比多个模型版本"""
    if not model_ids:
        raise HTTPException(status_code=400, detail="请指定要对比的模型ID")
    
    ids = model_ids.split(',')
    
    placeholders = ','.join(['%s' for _ in ids])
    rows = execute_query(f'''
        SELECT id, model_name, best_accuracy, best_val_loss, eval_report, config, created_at
        FROM training_jobs 
        WHERE project_id = %s AND id IN ({placeholders})
    ''', (project_id, *ids))
    
    models = []
    for row in rows:
        models.append({
            "id": row['id'],
            "name": row['model_name'],
            "best_accuracy": row['best_accuracy'],
            "best_val_loss": row['best_val_loss'],
            "eval_report": row['eval_report'] if row['eval_report'] else {},
            "config": row['config'] if row['config'] else {},
            "created_at": row['created_at']
        })
    
    return {"models": models}


# ============ 定时训练API ============

@app.post("/api/projects/{project_id}/schedule")
async def create_scheduled_training(
    project_id: str,
    schedule: str = Form(...),  # cron表达式或预设：daily, weekly
    dataset_id: str = Form(...),
    config: str = Form(...),
    name: str = Form(...)
):
    """
    创建定时训练任务
    
    schedule: cron表达式（如 "0 2 * * *" 每天2点）或预设 "daily", "weekly"
    """
    # 解析预设
    if schedule == "daily":
        cron = "0 2 * * *"  # 每天凌晨2点
    elif schedule == "weekly":
        cron = "0 2 * * 0"  # 每周日凌晨2点
    else:
        cron = schedule
    
    schedule_id = str(uuid.uuid4())
    
    # 确保表存在（PostgreSQL schema已包含此表）
    execute_update('''
        INSERT INTO training_schedules (id, project_id, name, dataset_id, config, cron_expression)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (schedule_id, project_id, name, dataset_id, config, cron))
    
    # 重新加载定时任务
    reload_scheduler()
    
    return {
        "success": True,
        "schedule_id": schedule_id,
        "cron": cron,
        "message": "定时任务已创建"
    }


@app.get("/api/projects/{project_id}/schedules")
async def list_schedules(project_id: str):
    """列出项目的定时训练任务"""
    # 确保表存在（PostgreSQL schema已包含此表）
    rows = execute_query('''
        SELECT id, name, dataset_id, cron_expression, is_active, last_run_at, next_run_at, created_at
        FROM training_schedules 
        WHERE project_id = %s
        ORDER BY created_at DESC
    ''', (project_id,))
    
    schedules = []
    for row in rows:
        schedules.append({
            "id": row['id'],
            "name": row['name'],
            "dataset_id": row['dataset_id'],
            "cron": row['cron_expression'],
            "is_active": bool(row['is_active']),
            "last_run_at": row['last_run_at'],
            "next_run_at": row['next_run_at'],
            "created_at": row['created_at']
        })
    return {"schedules": schedules}


@app.delete("/api/projects/{project_id}/schedules/{schedule_id}")
async def delete_schedule(project_id: str, schedule_id: str):
    """删除定时任务"""
    execute_update('DELETE FROM training_schedules WHERE id = %s AND project_id = %s', 
                   (schedule_id, project_id))
    
    # 重新加载定时任务
    reload_scheduler()
    
    return {"success": True, "message": "定时任务已删除"}


# ============ 定时任务调度器 ============

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler()

def run_scheduled_training(schedule_id: str, project_id: str, dataset_id: str, config: str):
    """执行定时训练"""
    try:
        config_dict = json.loads(config)
        
        # 生成新的job_id
        job_id = str(uuid.uuid4())
        
        # 获取数据集路径
        rows = execute_query("SELECT file_path FROM datasets WHERE id = %s", (dataset_id,))
        
        if rows:
            # 创建训练任务
            log_path = LOGS_DIR / f"{job_id}.log"
            model_path = MODELS_DIR / project_id / f"{job_id}"
            
            execute_update('''
                INSERT INTO training_jobs (id, project_id, dataset_id, model_name, total_epochs,
                    log_path, model_path, config, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (job_id, project_id, dataset_id, config_dict.get('model_type', 'random_forest'),
                  config_dict.get('epochs', 100), str(log_path), str(model_path),
                  config, 'pending'))
            
            # 更新调度记录
            execute_update('''
                UPDATE training_schedules 
                SET last_run_at = CURRENT_TIMESTAMP
                WHERE id = %s
            ''', (schedule_id,))
            
            # 启动训练
            from tasks import submit_ml_training_task, submit_training_task
            
            task_type = config_dict.get('task_type', 'classification')
            if task_type in ['classification', 'regression', 'anomaly_detection']:
                submit_ml_training_task(job_id, project_id, dataset_id, config_dict)
            else:
                submit_training_task(job_id, project_id, dataset_id, config_dict)
        
    except Exception as e:
        logger.error(f"定时训练执行失败: {e}")


def reload_scheduler():
    """重新加载所有定时任务"""
    global scheduler
    
    # 移除所有现有任务
    for job in scheduler.get_jobs():
        job.remove()
    
    # 从数据库加载任务
    try:
        rows = execute_query('''
            SELECT id, project_id, dataset_id, config, cron_expression
            FROM training_schedules WHERE is_active = true
        ''')
        
        for row in rows:
            schedule_id, project_id, dataset_id, config, cron = row
            
            try:
                trigger = CronTrigger.from_crontab(cron)
                scheduler.add_job(
                    run_scheduled_training,
                    trigger=trigger,
                    id=schedule_id,
                    args=[schedule_id, project_id, dataset_id, config],
                    replace_existing=True
                )
            except Exception as e:
                logger.error(f"加载定时任务失败 {schedule_id}: {e}")
    except Exception as e:
        logger.error(f"重新加载调度器失败: {e}")


# 启动时加载定时任务
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    scheduler.start()
    reload_scheduler()
    
    # 添加智能学习定时检查（每分钟）
    from smart_learning import check_all_scheduled_learning
    scheduler.add_job(
        check_all_scheduled_learning,
        'interval',
        minutes=1,
        id='smart_learning_checker',
        replace_existing=True
    )
    logger.info("智能学习定时检查已启动（每分钟）")
    
    logger.info("定时任务调度器已启动")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    scheduler.shutdown()
    logger.info("定时任务调度器已关闭")


# ============ 时序分析API ============

@app.post("/api/projects/{project_id}/time-series/analyze")
async def analyze_time_series(
    project_id: str,
    dataset_id: str = Form(...),
    forecast_hours: int = Form(24),
    time_col: Optional[str] = Form(None),
    value_cols: Optional[str] = Form(None)
):
    """时序分析 - 趋势预测和异常检测
    
    Args:
        time_col: 时间列名（可选，默认自动检测）
        value_cols: 数值列名，逗号分隔（可选，默认自动检测）
    """
    from time_series_analyzer import analyze_equipment_trends
    import pandas as pd
    import os
    
    # 获取数据集路径
    rows = execute_query("SELECT file_path FROM datasets WHERE id = %s", (dataset_id,))
    
    if not rows:
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    file_path = rows[0]['file_path']
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"数据文件不存在: {file_path}")
    
    # 检查是否是文件夹
    if os.path.isdir(file_path):
        # 查找文件夹中的CSV/Excel文件
        data_files = []
        for f in os.listdir(file_path):
            if f.endswith(('.csv', '.xlsx', '.xls')):
                data_files.append(os.path.join(file_path, f))
        
        if not data_files:
            raise HTTPException(
                status_code=400, 
                detail="该数据集是图片文件夹，不支持时序分析。时序分析适用于CSV/Excel格式的时间序列数据（如传感器数据）。"
            )
        
        # 使用第一个找到的表格文件
        file_path = data_files[0]
        logger.info(f"从文件夹中找到数据文件: {file_path}")
    
    # 检查文件类型
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 图片数据不支持时序分析
    if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        raise HTTPException(
            status_code=400, 
            detail="图片数据不支持时序分析。时序分析适用于CSV/Excel格式的时间序列数据（如传感器数据）。"
        )
    
    # 文本数据不支持时序分析
    if file_ext in ['.txt', '.json']:
        raise HTTPException(
            status_code=400,
            detail="文本数据不支持时序分析。请先进行文本标注和训练，或使用表格类时序数据。"
        )
    
    # 只支持CSV和Excel文件
    if file_ext not in ['.csv', '.xlsx', '.xls']:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式({file_ext or '无扩展名'})。时序分析只支持CSV或Excel格式的表格数据。"
        )
    
    try:
        # 先尝试读取文件检查数据结构
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, nrows=5)
            else:
                df = pd.read_csv(file_path, nrows=5)
        except Exception as e:
            logger.error(f"读取数据文件失败: {e}")
            raise HTTPException(status_code=400, detail=f"无法读取数据文件，请检查文件格式: {str(e)}")
        
        # 检查是否有数据
        if df.empty:
            raise HTTPException(status_code=400, detail="数据文件为空")
        
        # 检查是否有数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise HTTPException(
                status_code=400, 
                detail="数据中没有数值列，无法进行趋势分析。请确保数据包含可分析的数值指标。"
            )
        
        logger.info(f"开始时序分析: file={file_path}, columns={df.columns.tolist()}, numeric_cols={numeric_cols}")
        
        # 解析数值列列表
        value_cols_list = None
        if value_cols:
            value_cols_list = [c.strip() for c in value_cols.split(',') if c.strip()]
        
        result = analyze_equipment_trends(
            file_path, 
            forecast_hours,
            time_col=time_col,
            value_cols=value_cols_list
        )
        
        # 检查分析结果是否有错误
        if 'error' in result:
            logger.warning(f"时序分析返回错误: {result['error']}")
            return {"success": False, "error": result['error'], "analysis": result}
        
        return {"success": True, "analysis": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"时序分析失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.post("/api/projects/{project_id}/time-series/rul")
async def predict_rul(
    project_id: str,
    dataset_id: str = Form(...),
    degradation_col: str = Form(...),
    threshold: float = Form(...)
):
    """预测剩余使用寿命 (RUL)"""
    from time_series_analyzer import TimeSeriesAnalyzer
    import pandas as pd
    
    rows = execute_query("SELECT file_path FROM datasets WHERE id = %s", (dataset_id,))
    
    if not rows:
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    file_path = rows[0]['file_path']
    
    # 检查是否是文件夹
    if os.path.isdir(file_path):
        # 查找文件夹中的CSV/Excel文件
        data_files = []
        for f in os.listdir(file_path):
            if f.endswith(('.csv', '.xlsx', '.xls')):
                data_files.append(os.path.join(file_path, f))
        
        if not data_files:
            raise HTTPException(
                status_code=400, 
                detail="该数据集是图片文件夹，不支持RUL预测。RUL预测适用于CSV/Excel格式的时间序列数据。"
            )
        
        # 使用第一个找到的表格文件
        file_path = data_files[0]
    
    try:
        df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
        analyzer = TimeSeriesAnalyzer()
        result = analyzer.predict_rul(df, 'timestamp', degradation_col, threshold)
        return {"success": True, "rul": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RUL预测失败: {str(e)}")


# ============ 在线学习API ============

@app.post("/api/projects/{project_id}/models/{job_id}/online-learn")
async def start_online_learning(
    project_id: str,
    job_id: str,
    dataset_id: str = Form(...),
    learning_type: str = Form("incremental")
):
    """启动在线学习"""
    from online_learning import OnlineLearningManager
    
    manager = OnlineLearningManager()
    
    # 创建学习任务
    task_id = manager.create_learning_task(project_id, job_id, learning_type)
    
    # 获取数据集路径
    rows = execute_query("SELECT file_path FROM datasets WHERE id = %s", (dataset_id,))
    
    if not rows:
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    # 执行增量学习
    try:
        result = manager.incremental_learn(task_id, rows[0]['file_path'])
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"在线学习失败: {str(e)}")


@app.get("/api/projects/{project_id}/models/{job_id}/learning-history")
async def get_learning_history(project_id: str, job_id: str):
    """获取在线学习历史"""
    from online_learning import OnlineLearningManager
    
    manager = OnlineLearningManager()
    history = manager.get_learning_history(project_id, job_id)
    
    return {"history": history, "count": len(history)}


@app.post("/api/projects/{project_id}/models/{job_id}/rollback")
async def rollback_model(
    project_id: str,
    job_id: str,
    target_version: Optional[str] = Form(None)
):
    """回滚模型到历史版本"""
    from online_learning import OnlineLearningManager
    
    manager = OnlineLearningManager()
    result = manager.rollback_model(job_id, target_version)
    
    if result['success']:
        return result
    else:
        raise HTTPException(status_code=400, detail=result['message'])


@app.post("/api/projects/{project_id}/models/{job_id}/auto-learn")
async def setup_auto_learning(
    project_id: str,
    job_id: str,
    schedule: str = Form("daily"),  # daily, weekly, never
    min_samples: int = Form(100),
    accuracy_threshold: float = Form(0.05)
):
    """设置自动在线学习"""
    from online_learning import OnlineLearningManager
    
    manager = OnlineLearningManager()
    config_id = manager.setup_auto_learning(
        project_id, job_id, schedule, min_samples, accuracy_threshold
    )
    
    return {
        "success": True,
        "config_id": config_id,
        "schedule": schedule,
        "min_samples": min_samples
    }


@app.get("/api/projects/{project_id}/auto-learning-config")
async def get_auto_learning_config(project_id: str):
    """获取自动学习配置"""
    rows = execute_query('''
        SELECT id, job_id, schedule, min_samples, accuracy_threshold, is_active
        FROM auto_learning_config
        WHERE project_id = %s
    ''', (project_id,))
    
    configs = []
    for row in rows:
        configs.append({
            "id": row['id'],
            "job_id": row['job_id'],
            "schedule": row['schedule'],
            "min_samples": row['min_samples'],
            "accuracy_threshold": row['accuracy_threshold'],
            "is_active": bool(row['is_active'])
        })
    return {"configs": configs}


# ============ 智能学习 API (新增) ============

@app.post("/api/projects/{project_id}/models/{job_id}/smart-learning")
async def create_smart_learning_config(
    project_id: str,
    job_id: str,
    trigger_schedule: str = Form("weekly"),
    trigger_min_samples: int = Form(100),
    trigger_performance_drop: float = Form(0.05),
    auto_deploy: bool = Form(False),
    min_improvement: float = Form(0.0)
):
    """
    创建智能学习配置
    自动检测模型类型，选择合适的策略
    sklearn模型 -> 增量学习
    PyTorch/Transformers -> 定时全量重训练
    """
    from smart_learning import SmartLearningScheduler
    
    scheduler = SmartLearningScheduler()
    
    result = scheduler.create_config(
        project_id=project_id,
        job_id=job_id,
        trigger_schedule=trigger_schedule,
        trigger_min_samples=trigger_min_samples,
        trigger_performance_drop=trigger_performance_drop,
        auto_deploy=auto_deploy,
        min_improvement=min_improvement
    )
    
    if result.get('success'):
        return result
    else:
        raise HTTPException(status_code=400, detail=result.get('error', '创建失败'))


@app.get("/api/projects/{project_id}/models/{job_id}/smart-learning")
async def get_smart_learning_config(project_id: str, job_id: str):
    """获取智能学习配置"""
    from smart_learning import SmartLearningScheduler
    
    scheduler = SmartLearningScheduler()
    config = scheduler.get_config(project_id, job_id)
    
    if config:
        return {"config": config}
    else:
        return {"config": None, "message": "未找到配置"}


@app.post("/api/projects/{project_id}/models/{job_id}/trigger-learning")
async def trigger_smart_learning(
    project_id: str,
    job_id: str,
    dataset_id: Optional[str] = Form(None),
    reason: str = Form("manual")
):
    """手动触发智能学习"""
    from smart_learning import SmartLearningScheduler
    
    scheduler = SmartLearningScheduler()
    
    # 获取配置
    config = scheduler.get_config(project_id, job_id)
    if not config:
        raise HTTPException(status_code=404, detail="未找到学习配置")
    
    # 触发学习
    result = scheduler.trigger_learning(config['id'], reason, dataset_id)
    
    if result.get('success'):
        return result
    else:
        raise HTTPException(status_code=500, detail=result.get('error', '触发失败'))


@app.post("/api/projects/{project_id}/models/compare-and-deploy")
async def compare_and_deploy_model(
    project_id: str,
    learning_job_id: str = Form(...),
    new_job_id: str = Form(...),
    auto_deploy: bool = Form(False),
    min_improvement: float = Form(0.0)
):
    """对比新旧模型并决定是否部署"""
    from smart_learning import SmartLearningScheduler
    
    scheduler = SmartLearningScheduler()
    
    result = scheduler.compare_and_deploy(
        learning_job_id=learning_job_id,
        new_job_id=new_job_id,
        auto_deploy=auto_deploy,
        min_improvement=min_improvement
    )
    
    if result.get('success'):
        return result
    else:
        raise HTTPException(status_code=400, detail=result.get('error', '对比失败'))


@app.get("/api/projects/{project_id}/models/{job_id}/unified-learning-history")
async def get_unified_learning_history(project_id: str, job_id: str):
    """获取统一的学习历史（包含增量学习和定时学习）"""
    from smart_learning import SmartLearningScheduler
    
    scheduler = SmartLearningScheduler()
    history = scheduler.get_learning_history(project_id, job_id)
    
    return {"history": history, "count": len(history)}


# ============ DeepSeek 根因分析 API ============

@app.post("/api/projects/{project_id}/analyze-root-cause")
async def analyze_root_cause(
    project_id: str,
    sensor_data: str = Form(...),  # JSON字符串
    prediction_result: str = Form(...)  # JSON字符串
):
    """
    DeepSeek 故障根因分析
    
    当本地模型预测故障概率较高时，调用DeepSeek进行深度分析
    """
    from llm_integration import LLMService
    
    try:
        data = json.loads(sensor_data)
        prediction = json.loads(prediction_result)
        
        llm = LLMService('deepseek')
        result = llm.analyze_equipment_report(data, prediction)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"根因分析失败: {str(e)}")


@app.post("/api/projects/{project_id}/generate-report")
async def generate_maintenance_report(
    project_id: str,
    analysis_data: str = Form(...)  # JSON字符串，包含一段时间的分析结果
):
    """
    生成维护报告
    """
    from llm_integration import LLMService
    
    try:
        data_list = json.loads(analysis_data)
        
        llm = LLMService('deepseek')
        result = llm.generate_maintenance_report(data_list)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"报告生成失败: {str(e)}")


@app.get("/api/llm/status")
async def check_llm_status():
    """检查大模型服务状态"""
    from llm_integration import LLMService
    
    llm = LLMService('deepseek')
    
    return {
        "provider": "deepseek",
        "available": llm.api_key is not None,
        "message": "API已配置" if llm.api_key else "未配置DEEPSEEK_API_KEY环境变量"
    }


# ============ Celery 任务队列管理 API ============

@app.get("/api/celery/status")
async def celery_status():
    """检查 Celery 和 Redis 状态"""
    try:
        # 检查 Redis 连接
        import redis
        r = redis.from_url('redis://localhost:6379/0')
        r.ping()
        redis_status = "connected"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    # 检查 Celery 任务统计
    try:
        inspector = celery_app.control.inspect()
        active_tasks = inspector.active()
        scheduled_tasks = inspector.scheduled()
        
        worker_count = len(active_tasks) if active_tasks else 0
        active_count = sum(len(t) for t in active_tasks.values()) if active_tasks else 0
        scheduled_count = sum(len(t) for t in scheduled_tasks.values()) if scheduled_tasks else 0
        
        return {
            "celery_version": celery_app.__version__ if hasattr(celery_app, '__version__') else "unknown",
            "redis_status": redis_status,
            "worker_count": worker_count,
            "active_tasks": active_count,
            "scheduled_tasks": scheduled_count,
            "broker_url": str(celery_app.conf.broker_url),
            "result_backend": str(celery_app.conf.result_backend)
        }
    except Exception as e:
        return {
            "redis_status": redis_status,
            "error": str(e),
            "broker_url": str(celery_app.conf.broker_url)
        }


@app.get("/api/projects/{project_id}/jobs/{job_id}/celery-status")
async def get_celery_task_status(project_id: str, job_id: str):
    """获取 Celery 任务的详细状态"""
    # 从数据库获取任务信息
    rows = execute_query('''
        SELECT id, status, model_name, progress, current_epoch, total_epochs,
               best_accuracy, created_at, started_at, completed_at, stop_reason
        FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    job = rows[0]
    
    # 尝试获取 Celery 任务状态
    celery_info = {}
    try:
        # 通过任务ID查询Celery结果（如果任务是通过Celery提交的）
        from celery.result import AsyncResult
        result = AsyncResult(job_id, app=celery_app)
        
        if result.id:
            celery_info = {
                "celery_state": result.state,
                "celery_result": result.result if result.ready() else None,
            }
    except Exception as e:
        celery_info = {"error": str(e)}
    
    return {
        "job": {
            "id": job['id'],
            "status": job['status'],
            "model_name": job['model_name'],
            "progress": job['progress'],
            "current_epoch": job['current_epoch'],
            "total_epochs": job['total_epochs'],
            "best_accuracy": job['best_accuracy'],
            "created_at": job['created_at'],
            "started_at": job['started_at'],
            "completed_at": job['completed_at'],
            "stop_reason": job['stop_reason']
        },
        "celery": celery_info
    }


@app.post("/api/celery/retry-task")
async def retry_celery_task(job_id: str = Form(...), project_id: str = Form(...)):
    """重试失败的任务"""
    # 获取任务信息
    rows = execute_query('''
        SELECT id, project_id, dataset_id, model_name, config
        FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    job = rows[0]
    
    # 重置任务状态
    execute_update('''
        UPDATE training_jobs 
        SET status = 'pending', progress = 0, current_epoch = 0, 
            current_loss = NULL, stop_reason = NULL
        WHERE id = %s
    ''', (job_id,))
    
    # 重新提交任务
    config = job['config'] if isinstance(job['config'], dict) else json.loads(job['config'])
    
    if config.get('task_type') == 'image_classification':  # 图片分类任务
        future = submit_image_training_task(job_id, project_id, job['dataset_id'], config)
    elif 'model_name' in config and config.get('task_type') != 'image_classification':  # NLP任务
        future = submit_training_task(job_id, project_id, job['dataset_id'], config)
    else:  # ML任务
        future = submit_ml_training_task(job_id, project_id, job['dataset_id'], config)
    
    return {
        "success": True,
        "message": "任务已重新提交",
        "celery_task_id": future.id if hasattr(future, 'id') else None
    }


# ============ 模型导出 API ============

@app.get("/api/projects/{project_id}/models/{job_id}/export-formats")
async def get_export_formats(project_id: str, job_id: str):
    """获取模型可用的导出格式"""
    rows = execute_query('''
        SELECT model_path, config, status FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    job = rows[0]
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="训练尚未完成")
    
    config = job['config'] if isinstance(job['config'], dict) else json.loads(job['config'])
    
    # 根据任务类型返回可用格式
    formats = []
    
    if config.get('task_type') == 'image_classification':
        formats = [
            {"format": "pth", "name": "PyTorch模型", "ext": ".pth", "desc": "原始PyTorch格式"},
            {"format": "onnx", "name": "ONNX", "ext": ".onnx", "desc": "跨平台推理格式"},
            {"format": "quantized", "name": "INT8量化", "ext": "_quantized.pth", "desc": "压缩模型，适合边缘设备"}
        ]
    elif 'model_name' in config:  # NLP任务
        formats = [
            {"format": "pytorch", "name": "PyTorch", "ext": ".bin", "desc": "HuggingFace格式"},
            {"format": "onnx", "name": "ONNX", "ext": ".onnx", "desc": "ONNX Runtime推理"}
        ]
    else:  # ML任务
        formats = [
            {"format": "pkl", "name": "Pickle", "ext": ".pkl", "desc": "Python序列化格式"},
            {"format": "onnx", "name": "ONNX", "ext": ".onnx", "desc": "通用推理格式"}
        ]
    
    return {"formats": formats}


@app.post("/api/projects/{project_id}/models/{job_id}/export")
async def export_model(
    project_id: str,
    job_id: str,
    format: str = Form(...),
    quantize: bool = Form(False)
):
    """
    导出模型到指定格式
    
    Args:
        format: 导出格式 (onnx, quantized, etc.)
        quantize: 是否量化
    """
    rows = execute_query('''
        SELECT model_path, config, status, task_type FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    job = rows[0]
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="训练尚未完成")
    
    model_path = job['model_path']
    config = job['config'] if isinstance(job['config'], dict) else json.loads(job['config'])
    
    export_results = []
    
    # 检查已有文件
    model_dir = Path(model_path)
    
    if format == "onnx":
        onnx_path = model_dir / "model.onnx"
        if onnx_path.exists():
            export_results.append({
                "format": "onnx",
                "path": str(onnx_path),
                "size_mb": onnx_path.stat().st_size / (1024 * 1024)
            })
        else:
            raise HTTPException(status_code=404, detail="ONNX模型不存在，可能训练时未生成")
    
    elif format == "quantized" or quantize:
        # 执行INT8量化
        quantized_path = model_dir / "model_quantized.pth"
        if not quantized_path.exists():
            result = quantize_model(
                str(model_dir / "best_model.pth"),
                str(quantized_path)
            )
            if not result['success']:
                raise HTTPException(status_code=500, detail=f"量化失败: {result.get('error')}")
        
        export_results.append({
            "format": "quantized",
            "path": str(quantized_path),
            "size_mb": quantized_path.stat().st_size / (1024 * 1024) if quantized_path.exists() else 0
        })
    
    elif format == "pth" or format == "pytorch":
        pth_path = model_dir / "best_model.pth"
        if pth_path.exists():
            export_results.append({
                "format": "pth",
                "path": str(pth_path),
                "size_mb": pth_path.stat().st_size / (1024 * 1024)
            })
    
    return {
        "success": True,
        "exports": export_results,
        "download_url": f"/api/projects/{project_id}/models/{job_id}/download?format={format}"
    }


@app.get("/api/projects/{project_id}/models/{job_id}/download")
async def download_model(
    project_id: str,
    job_id: str,
    format: str = "pth"
):
    """下载导出的模型文件"""
    rows = execute_query('''
        SELECT model_path, config, status FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    job = rows[0]
    model_dir = Path(job['model_path'])
    
    # 根据格式选择文件
    file_map = {
        "pth": model_dir / "best_model.pth",
        "onnx": model_dir / "model.onnx",
        "quantized": model_dir / "model_quantized.pth",
        "final": model_dir / "final_model.pth"
    }
    
    file_path = file_map.get(format, model_dir / "best_model.pth")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{format}格式模型文件不存在")
    
    return FileResponse(
        file_path,
        filename=f"model_{job_id[:8]}_{format}{file_path.suffix}",
        media_type='application/octet-stream'
    )


# ============ XAI 可解释性 API ============

@app.post("/api/xai/image/gradcam")
async def explain_image_gradcam(
    project_id: str = Form(...),
    job_id: str = Form(...),
    image: UploadFile = File(...),
    target_layer: Optional[str] = Form(None)
):
    """
    使用Grad-CAM解释图像分类预测
    
    返回热力图和叠加图(base64编码)
    """
    import torch
    import torch.nn.functional as F
    import cv2
    import numpy as np
    from torchvision import transforms, models
    from PIL import Image as PILImage
    from xai_explainer import GradCAMExplainer
    
    # 获取模型
    rows = execute_query('''
        SELECT model_path, config FROM training_jobs 
        WHERE id = %s AND project_id = %s AND status = 'completed'
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="模型不存在或训练未完成")
    
    job = rows[0]
    model_path = Path(job['model_path']) / 'best_model.pth'
    config = job['config'] if isinstance(job['config'], dict) else json.loads(job['config'])
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="模型文件不存在")
    
    try:
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 重建模型结构
        model_name = config.get('model_name', 'resnet50')
        num_classes = len(checkpoint.get('class_names', []))
        
        if 'resnet' in model_name:
            model = getattr(models, model_name)(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif 'efficientnet' in model_name:
            model = getattr(models, model_name)(pretrained=False)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        elif 'mobilenet' in model_name:
            model = getattr(models, model_name)(pretrained=False)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模型类型: {model_name}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 读取上传的图片
        image_bytes = await image.read()
        pil_img = PILImage.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 预处理
        image_size = config.get('image_size', 224)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(pil_img).unsqueeze(0)
        image_array = np.array(pil_img.resize((image_size, image_size)))
        
        # 创建解释器
        explainer = GradCAMExplainer(model, target_layer)
        
        # 生成解释
        with torch.enable_grad():
            heatmap = explainer.generate_heatmap(image_tensor)
        
        # 获取预测
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            target_class = output.argmax(dim=1).item()
            confidence = probs[0, target_class].item()
        
        class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(num_classes)])
        
        # 可视化
        overlay = explainer.visualize(image_array, heatmap)
        
        # 转为base64
        def array_to_base64(arr: np.ndarray) -> str:
            pil_img = PILImage.fromarray(arr)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        
        # 热力图单独保存
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        return {
            "success": True,
            "prediction": {
                "class_id": target_class,
                "class_name": class_names[target_class],
                "confidence": confidence
            },
            "heatmap": array_to_base64(heatmap_colored),
            "overlay": array_to_base64(overlay),
            "original": array_to_base64(image_array),
            "model_info": {
                "model_name": model_name,
                "target_layer": explainer.target_layer
            }
        }
        
    except Exception as e:
        logger.error(f"Grad-CAM解释失败: {e}")
        raise HTTPException(status_code=500, detail=f"解释失败: {str(e)}")


@app.get("/api/projects/{project_id}/jobs/{job_id}/xai/feature-importance")
async def get_feature_importance(project_id: str, job_id: str):
    """
    获取表格数据模型的特征重要性
    
    支持：RandomForest的feature_importances_，或SHAP值
    """
    import pickle
    
    rows = execute_query('''
        SELECT model_path, config, task_type FROM training_jobs 
        WHERE id = %s AND project_id = %s AND status = 'completed'
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="模型不存在或训练未完成")
    
    job = rows[0]
    model_dir = Path(job['model_path'])
    
    # 尝试加载不同格式的模型
    model = None
    feature_names = []
    
    # 1. 尝试pickle格式
    pkl_path = model_dir / 'model.pkl'
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
    
    # 2. 尝试joblib格式
    if model is None:
        joblib_path = model_dir / 'model.joblib'
        if joblib_path.exists():
            import joblib
            model = joblib.load(joblib_path)
    
    if model is None:
        raise HTTPException(status_code=404, detail="找不到可解释的模型文件")
    
    # 获取特征重要性
    importance_data = []
    
    # RandomForest / GradientBoosting
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        config = job['config'] if isinstance(job['config'], dict) else json.loads(job['config'])
        feature_cols = config.get('feature_columns', [])
        
        if not feature_cols:
            feature_cols = [f'feature_{i}' for i in range(len(importances))]
        
        importance_data = [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(feature_cols, importances)
        ]
        importance_data.sort(key=lambda x: x['importance'], reverse=True)
    
    # Linear models (coef_)
    elif hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        config = job['config'] if isinstance(job['config'], dict) else json.loads(job['config'])
        feature_cols = config.get('feature_columns', [f'feature_{i}' for i in range(len(coefs))])
        
        importance_data = [
            {"feature": name, "importance": float(coef)}
            for name, coef in zip(feature_cols, coefs)
        ]
        importance_data.sort(key=lambda x: x['importance'], reverse=True)
    
    return {
        "success": True,
        "model_type": type(model).__name__,
        "feature_importance": importance_data[:20],  # 前20个
        "total_features": len(importance_data)
    }


@app.post("/api/xai/text/attention")
async def explain_text_attention(
    project_id: str = Form(...),
    job_id: str = Form(...),
    text: str = Form(...)
):
    """
    解释文本分类的Attention权重
    
    返回每个token的attention权重
    """
    # 简化版：返回分词和模拟权重
    # 实际实现需要加载Transformer模型获取真实attention
    
    tokens = text[:200]  # 限制长度
    
    return {
        "success": True,
        "text": text,
        "tokens": [{"token": c, "weight": 0.1, "index": i} for i, c in enumerate(tokens)],
        "note": "这是简化版，完整版需要加载Transformer模型获取Attention权重"
    }


# ============ 告警规则 API ============

@app.get("/api/projects/{project_id}/alert-rules")
async def list_alert_rules(project_id: str):
    """获取项目的所有告警规则"""
    # 验证项目存在
    rows = execute_query('SELECT id FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    rules = get_project_alert_rules(project_id)
    return {"rules": rules, "count": len(rules)}


@app.post("/api/projects/{project_id}/alert-rules")
async def create_alert_rule(
    project_id: str,
    name: str = Form(...),
    description: str = Form(""),
    rule_type: str = Form(...),
    condition_field: str = Form(...),
    condition_operator: str = Form(...),
    condition_value: float = Form(...),
    severity: str = Form("warning"),
    cooldown_minutes: int = Form(60),
    notify_channels: str = Form("[\"feishu\"]")
):
    """创建告警规则"""
    # 验证项目存在
    rows = execute_query('SELECT id FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    try:
        channels = json.loads(notify_channels)
    except:
        channels = ["feishu"]
    
    rule_id = AlertEngine.create_rule(
        project_id=project_id,
        name=name,
        description=description,
        rule_type=rule_type,
        condition_field=condition_field,
        condition_operator=condition_operator,
        condition_value=condition_value,
        severity=severity,
        notify_channels=channels
    )
    
    # 更新冷却时间
    execute_update('''
        UPDATE alert_rules SET cooldown_minutes = %s WHERE id = %s
    ''', (cooldown_minutes, rule_id))
    
    return {"success": True, "rule_id": rule_id, "message": "告警规则创建成功"}


@app.post("/api/projects/{project_id}/alert-rules/template")
async def create_alert_rule_from_template(
    project_id: str,
    template_key: str = Form(...),
    custom_values: str = Form("{}")
):
    """从模板创建告警规则"""
    # 验证项目存在
    rows = execute_query('SELECT id FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    try:
        custom = json.loads(custom_values)
    except:
        custom = {}
    
    try:
        rule_id = AlertEngine.create_rule_from_template(
            project_id=project_id,
            template_key=template_key,
            custom_values=custom
        )
        return {"success": True, "rule_id": rule_id, "message": "从模板创建成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/projects/{project_id}/alert-rules/{rule_id}")
async def update_alert_rule(
    project_id: str,
    rule_id: str,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    condition_value: Optional[float] = Form(None),
    severity: Optional[str] = Form(None),
    enabled: Optional[bool] = Form(None),
    cooldown_minutes: Optional[int] = Form(None)
):
    """更新告警规则"""
    # 验证规则存在
    rows = execute_query(
        'SELECT id FROM alert_rules WHERE id = %s AND project_id = %s',
        (rule_id, project_id)
    )
    if not rows:
        raise HTTPException(status_code=404, detail="告警规则不存在")
    
    updates = []
    values = []
    
    if name:
        updates.append("name = %s")
        values.append(name)
    if description is not None:
        updates.append("description = %s")
        values.append(description)
    if condition_value is not None:
        updates.append("condition_value = %s")
        values.append(condition_value)
    if severity:
        updates.append("severity = %s")
        values.append(severity)
    if enabled is not None:
        updates.append("enabled = %s")
        values.append(enabled)
    if cooldown_minutes is not None:
        updates.append("cooldown_minutes = %s")
        values.append(cooldown_minutes)
    
    if updates:
        updates.append("updated_at = CURRENT_TIMESTAMP")
        query = f"UPDATE alert_rules SET {', '.join(updates)} WHERE id = %s"
        values.append(rule_id)
        execute_update(query, tuple(values))
    
    return {"success": True, "message": "告警规则更新成功"}


@app.delete("/api/projects/{project_id}/alert-rules/{rule_id}")
async def delete_alert_rule(project_id: str, rule_id: str):
    """删除告警规则"""
    rows = execute_query(
        'SELECT id FROM alert_rules WHERE id = %s AND project_id = %s',
        (rule_id, project_id)
    )
    if not rows:
        raise HTTPException(status_code=404, detail="告警规则不存在")
    
    execute_update('DELETE FROM alert_rules WHERE id = %s', (rule_id,))
    return {"success": True, "message": "告警规则已删除"}


@app.get("/api/projects/{project_id}/alerts")
async def list_alerts(
    project_id: str,
    status: Optional[str] = None,
    limit: int = 50
):
    """获取告警历史"""
    rows = execute_query('SELECT id FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    alerts = get_alert_history(project_id, status, limit)
    return {"alerts": alerts, "count": len(alerts)}


@app.post("/api/projects/{project_id}/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(project_id: str, alert_id: str):
    """确认告警"""
    rows = execute_query(
        'SELECT id FROM alert_history WHERE id = %s AND project_id = %s',
        (alert_id, project_id)
    )
    if not rows:
        raise HTTPException(status_code=404, detail="告警不存在")
    
    execute_update('''
        UPDATE alert_history 
        SET status = 'acknowledged', 
            resolved_at = CURRENT_TIMESTAMP
        WHERE id = %s
    ''', (alert_id,))
    
    return {"success": True, "message": "告警已确认"}


@app.post("/api/projects/{project_id}/alerts/{alert_id}/resolve")
async def resolve_alert(project_id: str, alert_id: str):
    """解决告警"""
    rows = execute_query(
        'SELECT id FROM alert_history WHERE id = %s AND project_id = %s',
        (alert_id, project_id)
    )
    if not rows:
        raise HTTPException(status_code=404, detail="告警不存在")
    
    execute_update('''
        UPDATE alert_history 
        SET status = 'resolved', 
            resolved_at = CURRENT_TIMESTAMP
        WHERE id = %s
    ''', (alert_id,))
    
    return {"success": True, "message": "告警已解决"}


@app.get("/api/alert-templates")
async def get_alert_templates():
    """获取告警规则模板列表"""
    templates = AlertEngine.RULE_TEMPLATES
    return {
        "templates": [
            {
                "key": key,
                "name": template["name"],
                "description": template["description"],
                "rule_type": template["rule_type"],
                "severity": template["severity"],
                "default_threshold": template.get("condition_value")
            }
            for key, template in templates.items()
        ]
    }


# 在训练完成时自动检查告警
@app.post("/api/projects/{project_id}/jobs/{job_id}/check-alerts")
async def manual_check_alerts(project_id: str, job_id: str):
    """手动触发告警检查（用于测试）"""
    try:
        AlertEngine.check_training_alerts(job_id, project_id)
        return {"success": True, "message": "告警检查已触发"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 数据标注 API ============

@app.get("/api/projects/{project_id}/datasets/{dataset_id}/images")
async def get_dataset_images(project_id: str, dataset_id: str):
    """获取数据集图片列表"""
    # 验证项目存在
    rows = execute_query('SELECT id FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    # 获取数据集路径
    rows = execute_query('SELECT file_path FROM datasets WHERE id = %s AND project_id = %s', 
                        (dataset_id, project_id))
    if not rows:
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    dataset_path = rows[0]['file_path']
    
    # 扫描图片文件
    import os
    from pathlib import Path
    
    images = []
    path = Path(dataset_path)
    if path.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_path in path.rglob(ext):
                # 获取相对路径
                rel_path = img_path.relative_to(path)
                images.append({
                    'id': str(rel_path).replace('/', '_').replace('\\', '_'),
                    'path': str(rel_path),
                    'filename': img_path.name,
                    'class': rel_path.parts[0] if len(rel_path.parts) > 1 else 'unknown'
                })
    
    return {"images": images, "count": len(images), "dataset_path": dataset_path}


@app.post("/api/projects/{project_id}/annotation-tasks")
async def create_annotation_task(
    project_id: str,
    dataset_id: str = Form(...),
    name: str = Form(...),
    task_type: str = Form("bbox"),
    classes: str = Form("[]")
):
    """创建标注任务"""
    import uuid
    
    rows = execute_query('SELECT id FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    task_id = str(uuid.uuid4())
    
    # 获取图片数量
    images_res = await get_dataset_images(project_id, dataset_id)
    total_images = images_res["count"]
    
    execute_update("""
        INSERT INTO annotation_tasks (id, project_id, dataset_id, name, task_type, classes, total_images)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (task_id, project_id, dataset_id, name, task_type, classes, total_images))
    
    # 初始化annotations记录
    for img in images_res["images"]:
        anno_id = str(uuid.uuid4())
        execute_update("""
            INSERT INTO annotations (id, task_id, image_id, image_path, status)
            VALUES (%s, %s, %s, %s, 'pending')
        """, (anno_id, task_id, img['id'], img['path']))
    
    return {"success": True, "task_id": task_id, "total_images": total_images}


@app.get("/api/projects/{project_id}/annotation-tasks")
async def list_annotation_tasks(project_id: str):
    """获取标注任务列表"""
    rows = execute_query('SELECT id FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    tasks = execute_query("""
        SELECT t.*, d.name as dataset_name,
               (SELECT COUNT(*) FROM annotations WHERE task_id = t.id AND status = 'annotated') as annotated_count
        FROM annotation_tasks t
        JOIN datasets d ON t.dataset_id = d.id
        WHERE t.project_id = %s
        ORDER BY t.created_at DESC
    """, (project_id,))
    
    return {"tasks": [dict(row) for row in tasks], "count": len(tasks)}


@app.get("/api/annotation-tasks/{task_id}/images")
async def get_task_images(task_id: str, status: str = None):
    """获取任务图片列表"""
    if status:
        rows = execute_query("""
            SELECT * FROM annotations 
            WHERE task_id = %s AND status = %s
            ORDER BY created_at
        """, (task_id, status))
    else:
        rows = execute_query("""
            SELECT * FROM annotations 
            WHERE task_id = %s
            ORDER BY created_at
        """, (task_id,))
    
    return {"images": [dict(row) for row in rows], "count": len(rows)}


@app.get("/api/annotation-tasks/{task_id}/images/{image_id}")
async def get_image_annotation(task_id: str, image_id: str):
    """获取单张图片的标注"""
    rows = execute_query("""
        SELECT a.*, t.dataset_id, d.file_path as dataset_path
        FROM annotations a
        JOIN annotation_tasks t ON a.task_id = t.id
        JOIN datasets d ON t.dataset_id = d.id
        WHERE a.task_id = %s AND a.image_id = %s
    """, (task_id, image_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="图片不存在")
    
    return dict(rows[0])


@app.post("/api/annotation-tasks/{task_id}/images/{image_id}/annotate")
async def save_annotation(
    task_id: str,
    image_id: str,
    objects: str = Form(...),  # JSON string of objects
    time_spent: int = Form(0)
):
    """保存标注结果"""
    import json
    
    try:
        objects_data = json.loads(objects)
    except:
        raise HTTPException(status_code=400, detail="objects格式错误")
    
    execute_update("""
        UPDATE annotations 
        SET objects = %s, 
            status = 'annotated',
            time_spent = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE task_id = %s AND image_id = %s
    """, (json.dumps(objects_data), time_spent, task_id, image_id))
    
    # 更新任务进度
    execute_update("""
        UPDATE annotation_tasks 
        SET annotated_count = (
            SELECT COUNT(*) FROM annotations WHERE task_id = %s AND status = 'annotated'
        ),
        status = CASE 
            WHEN annotated_count = total_images THEN 'completed'
            WHEN annotated_count > 0 THEN 'in_progress'
            ELSE 'pending'
        END
        WHERE id = %s
    """, (task_id, task_id))
    
    return {"success": True, "message": "标注已保存"}


@app.post("/api/annotation-tasks/{task_id}/export")
async def export_annotations(task_id: str, format: str = Form("yolo")):
    """导出标注结果为YOLO格式"""
    import json
    from pathlib import Path
    
    # 获取任务信息
    rows = execute_query("""
        SELECT t.*, d.file_path as dataset_path
        FROM annotation_tasks t
        JOIN datasets d ON t.dataset_id = d.id
        WHERE t.id = %s
    """, (task_id,))
    
    if not rows:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = rows[0]
    dataset_path = Path(task['dataset_path'])
    
    # 创建导出目录
    export_dir = dataset_path / f"labels_{format}"
    export_dir.mkdir(exist_ok=True)
    
    # 获取所有标注
    annotations = execute_query("""
        SELECT * FROM annotations WHERE task_id = %s AND status = 'annotated'
    """, (task_id,))
    
    classes = task['classes'] if isinstance(task['classes'], list) else json.loads(task['classes'])
    class_to_id = {cls: i for i, cls in enumerate(classes)}
    
    # 导出为YOLO格式
    for anno in annotations:
        objects = anno['objects'] if isinstance(anno['objects'], list) else json.loads(anno['objects'])
        
        # YOLO格式: class_id center_x center_y width height (归一化)
        yolo_lines = []
        for obj in objects:
            cls = obj.get('class', 'unknown')
            class_id = class_to_id.get(cls, 0)
            bbox = obj.get('bbox', {})  # {x, y, width, height} (归一化0-1)
            
            x = bbox.get('x', 0) + bbox.get('width', 0) / 2  # center x
            y = bbox.get('y', 0) + bbox.get('height', 0) / 2  # center y
            w = bbox.get('width', 0)
            h = bbox.get('height', 0)
            
            yolo_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        
        # 保存文件
        image_path = Path(anno['image_path'])
        label_file = export_dir / f"{image_path.stem}.txt"
        label_file.write_text('\n'.join(yolo_lines))
    
    # 保存类别文件
    classes_file = export_dir / "classes.txt"
    classes_file.write_text('\n'.join(classes))
    
    return {
        "success": True,
        "export_dir": str(export_dir),
        "format": format,
        "annotated_count": len(annotations),
        "classes": classes
    }


# ============ AutoML API ============

@app.get("/api/projects/{project_id}/automl/search-spaces")
async def get_search_spaces(project_id: str):
    """获取可用的搜索空间配置"""
    rows = execute_query('SELECT id FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    return {
        "search_spaces": AutoMLConfig.SEARCH_SPACES,
        "message": "各任务类型的超参数搜索空间"
    }


@app.post("/api/projects/{project_id}/automl/experiments")
async def create_automl_experiment_api(
    project_id: str,
    dataset_id: str = Form(...),
    name: str = Form(...),
    max_trials: int = Form(20),
    base_config: str = Form("{}")
):
    """创建AutoML实验"""
    # 验证项目存在
    rows = execute_query('SELECT id, task_type FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    task_type = rows[0]['task_type'] or 'text_classification'
    
    try:
        config = json.loads(base_config)
    except:
        config = {}
    
    # 根据数据集自动推断任务类型（如果没有指定）
    if not task_type or task_type == 'text_classification':
        # 检查数据集类型
        ds_rows = execute_query('SELECT file_type FROM datasets WHERE id = %s', (dataset_id,))
        if ds_rows:
            file_type = ds_rows[0]['file_type']
            if file_type in ['csv', 'xlsx', 'xls']:
                # 结构化数据，使用 ML 任务
                task_type = 'classification'
    
    experiment_id = create_automl_experiment(
        project_id=project_id,
        dataset_id=dataset_id,
        name=name,
        base_config=config,
        task_type=task_type,
        max_trials=max_trials
    )
    
    return {
        "success": True,
        "experiment_id": experiment_id,
        "message": f"AutoML实验已创建，将运行{max_trials}组超参数搜索"
    }


@app.get("/api/projects/{project_id}/automl/experiments")
async def list_automl_experiments(project_id: str):
    """获取项目的AutoML实验列表"""
    rows = execute_query('SELECT id FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    experiments = execute_query("""
        SELECT e.*, d.name as dataset_name,
               (SELECT COUNT(*) FROM automl_trials WHERE experiment_id = e.id) as trials_count
        FROM automl_experiments e
        JOIN datasets d ON e.dataset_id = d.id
        WHERE e.project_id = %s
        ORDER BY e.created_at DESC
    """, (project_id,))
    
    return {
        "experiments": [dict(row) for row in experiments],
        "count": len(experiments)
    }


@app.get("/api/automl/experiments/{experiment_id}")
async def get_experiment_detail(experiment_id: str):
    """获取实验详情"""
    rows = execute_query("""
        SELECT e.*, d.name as dataset_name
        FROM automl_experiments e
        JOIN datasets d ON e.dataset_id = d.id
        WHERE e.id = %s
    """, (experiment_id,))
    
    if not rows:
        raise HTTPException(status_code=404, detail="实验不存在")
    
    experiment = dict(rows[0])
    
    # 获取所有trials
    trials = execute_query("""
        SELECT * FROM automl_trials
        WHERE experiment_id = %s
        ORDER BY trial_number
    """, (experiment_id,))
    
    experiment['trials'] = [dict(t) for t in trials]
    
    return {"experiment": experiment}


@app.post("/api/automl/experiments/{experiment_id}/run")
async def run_automl_experiment(experiment_id: str):
    """运行AutoML实验（生成一组超参数）"""
    rows = execute_query("""
        SELECT * FROM automl_experiments WHERE id = %s
    """, (experiment_id,))
    
    if not rows:
        raise HTTPException(status_code=404, detail="实验不存在")
    
    exp = rows[0]
    
    if exp['current_trial'] >= exp['max_trials']:
        return {"success": False, "message": "实验已完成所有trial"}
    
    # 更新状态为running
    execute_update("""
        UPDATE automl_experiments SET status = 'running' WHERE id = %s
    """, (experiment_id,))
    
    next_trial = exp['current_trial'] + 1
    
    # 生成超参数并创建trial记录
    result = run_automl_trial(experiment_id, next_trial)
    
    # 更新当前trial数
    execute_update("""
        UPDATE automl_experiments SET current_trial = %s WHERE id = %s
    """, (next_trial, experiment_id))
    
    # 如果所有trials都生成完成，更新状态为ready（等待训练）
    if next_trial >= exp['max_trials']:
        execute_update("""
            UPDATE automl_experiments SET status = 'ready' WHERE id = %s
        """, (experiment_id,))
    
    return {
        "success": True,
        "trial_id": result['trial_id'],
        "trial_number": result['trial_num'],
        "params": result['params'],
        "message": f"第{next_trial}组超参数已生成，请使用这些参数提交训练任务"
    }


@app.post("/api/automl/trials/{trial_id}/status")
async def update_trial_status(
    trial_id: str,
    status: str = Form(...),
    job_id: str = Form(None)
):
    """更新trial状态（用于启动训练时更新为running）"""
    rows = execute_query("SELECT * FROM automl_trials WHERE id = %s", (trial_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="trial不存在")
    
    execute_update("""
        UPDATE automl_trials 
        SET status = %s, job_id = %s
        WHERE id = %s
    """, (status, job_id, trial_id))
    
    return {"success": True, "message": f"trial状态已更新为{status}"}


@app.post("/api/automl/trials/{trial_id}/record")
async def record_trial_result(
    trial_id: str,
    metrics: str = Form(...),
    job_id: str = Form(None)
):
    """记录trial的训练结果"""
    try:
        metrics_data = json.loads(metrics)
    except:
        raise HTTPException(status_code=400, detail="metrics格式错误")
    
    # 获取trial信息
    rows = execute_query("SELECT * FROM automl_trials WHERE id = %s", (trial_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="trial不存在")
    
    # 更新metrics和状态
    execute_update("""
        UPDATE automl_trials 
        SET metrics = %s, job_id = %s, status = 'completed', completed_at = CURRENT_TIMESTAMP
        WHERE id = %s
    """, (json.dumps(metrics_data), job_id, trial_id))
    
    # 更新实验的最佳结果
    accuracy = metrics_data.get('accuracy', 0)
    trial = rows[0]
    
    # 检查是否是最佳
    exp_rows = execute_query("SELECT best_accuracy FROM automl_experiments WHERE id = %s", 
                            (trial['experiment_id'],))
    if exp_rows:
        current_best = exp_rows[0]['best_accuracy'] or 0
        if accuracy > current_best:
            execute_update("""
                UPDATE automl_experiments 
                SET best_accuracy = %s, best_trial_id = %s
                WHERE id = %s
            """, (accuracy, trial_id, trial['experiment_id']))
    
    return {"success": True, "message": "trial结果已记录"}


@app.get("/api/automl/experiments/{experiment_id}/comparison")
async def get_experiment_comparison(experiment_id: str):
    """获取实验对比分析"""
    return compare_trials(experiment_id)


@app.get("/api/automl/experiments/{experiment_id}/recommendation")
async def get_experiment_recommendation(experiment_id: str):
    """获取推荐配置"""
    return get_recommended_params(experiment_id)


@app.get("/api/automl/experiments/{experiment_id}/visualization")
async def get_experiment_visualization(experiment_id: str):
    """
    获取AutoML实验可视化数据
    
    返回格式化的数据用于：
    - 平行坐标图：展示多维度超参数与性能的关系
    - 散点图：展示特定参数与准确率的关系
    - 热力图：展示两个参数组合对性能的影响
    """
    rows = execute_query("""
        SELECT * FROM automl_experiments WHERE id = %s
    """, (experiment_id,))
    
    if not rows:
        raise HTTPException(status_code=404, detail="实验不存在")
    
    experiment = rows[0]
    task_type = experiment['task_type']
    
    # 获取所有trials
    trials_rows = execute_query("""
        SELECT trial_number, params, metrics, status
        FROM automl_trials
        WHERE experiment_id = %s AND status = 'completed'
        ORDER BY trial_number
    """, (experiment_id,))
    
    if not trials_rows:
        return {
            "experiment_id": experiment_id,
            "task_type": task_type,
            "trials_count": 0,
            "message": "暂无完成的trial数据"
        }
    
    trials = []
    for row in trials_rows:
        params = row['params'] if isinstance(row['params'], dict) else json.loads(row['params'])
        metrics = row['metrics'] if isinstance(row['metrics'], dict) else json.loads(row['metrics'])
        trials.append({
            "trial_number": row['trial_number'],
            "params": params,
            "metrics": metrics
        })
    
    # 获取搜索空间定义
    search_space = AutoMLConfig.SEARCH_SPACES.get(task_type, {})
    
    # 提取所有参数名称
    param_names = list(search_space.keys())
    
    # 格式化平行坐标图数据
    parallel_data = []
    for trial in trials:
        item = {"trial_number": trial['trial_number']}
        # 添加所有参数值
        for param in param_names:
            item[param] = trial['params'].get(param)
        # 添加性能指标
        item['accuracy'] = trial['metrics'].get('accuracy', 0)
        item['loss'] = trial['metrics'].get('loss', 0)
        parallel_data.append(item)
    
    # 格式化散点图数据（每个参数 vs 准确率）
    scatter_data = {}
    for param in param_names:
        scatter_data[param] = [
            {"x": trial['params'].get(param), "y": trial['metrics'].get('accuracy', 0), "trial": trial['trial_number']}
            for trial in trials
            if trial['params'].get(param) is not None
        ]
    
    # 找最佳和最差trial
    best_trial = max(trials, key=lambda t: t['metrics'].get('accuracy', 0))
    worst_trial = min(trials, key=lambda t: t['metrics'].get('accuracy', 0))
    
    return {
        "experiment_id": experiment_id,
        "task_type": task_type,
        "trials_count": len(trials),
        "search_space": search_space,
        "param_names": param_names,
        "visualization": {
            "parallel_coordinates": parallel_data,
            "scatter_plots": scatter_data,
            "best_trial": {
                "trial_number": best_trial['trial_number'],
                "accuracy": best_trial['metrics'].get('accuracy', 0),
                "params": best_trial['params']
            },
            "worst_trial": {
                "trial_number": worst_trial['trial_number'],
                "accuracy": worst_trial['metrics'].get('accuracy', 0),
                "params": worst_trial['params']
            }
        }
    }


@app.delete("/api/automl/experiments/{experiment_id}")
async def delete_automl_experiment(experiment_id: str):
    """删除AutoML实验（连带所有trials）"""
    # 验证实验存在
    rows = execute_query("SELECT id FROM automl_experiments WHERE id = %s", (experiment_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="实验不存在")
    
    # 先删除关联的trials
    execute_update("DELETE FROM automl_trials WHERE experiment_id = %s", (experiment_id,))
    
    # 删除实验
    execute_update("DELETE FROM automl_experiments WHERE id = %s", (experiment_id,))
    
    return {"success": True, "message": "实验及关联参数组已删除"}


# ============ A/B 测试 API ============

@app.post("/api/projects/{project_id}/ab-tests")
async def create_ab_test(
    project_id: str,
    name: str = Form(...),
    description: str = Form(""),
    model_a_id: str = Form(...),
    model_b_id: str = Form(...),
    traffic_split_a: int = Form(50),
    traffic_split_b: int = Form(50),
    min_sample_size: int = Form(100)
):
    """
    创建 A/B 测试实验
    
    Args:
        model_a_id: 控制组模型（原模型）
        model_b_id: 实验组模型（新模型）
        traffic_split_a/b: 流量分配比例，默认各50%
        min_sample_size: 最小样本数才判定结果
    """
    # 验证项目存在
    rows = execute_query('SELECT id FROM projects WHERE id = %s', (project_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    # 验证两个模型都存在且已完成训练（支持同一模型作为A和B用于测试）
    for model_id in [model_a_id, model_b_id]:
        model_rows = execute_query('''
            SELECT id, status FROM training_jobs 
            WHERE id = %s AND project_id = %s
        ''', (model_id, project_id))
        
        if not model_rows:
            raise HTTPException(status_code=400, detail=f"模型 {model_id[:8]} 不存在")
        
        if model_rows[0]['status'] != 'completed':
            raise HTTPException(status_code=400, detail=f"模型 {model_id[:8]} 训练尚未完成")
    
    # 验证流量分配
    if traffic_split_a + traffic_split_b != 100:
        raise HTTPException(status_code=400, detail="流量分配比例之和必须等于100")
    
    test_id = str(uuid.uuid4())
    
    execute_update('''
        INSERT INTO ab_tests (
            id, project_id, name, description, 
            model_a_id, model_b_id, traffic_split_a, traffic_split_b,
            min_sample_size, status, started_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
    ''', (
        test_id, project_id, name, description,
        model_a_id, model_b_id, traffic_split_a, traffic_split_b,
        min_sample_size, 'running'
    ))
    
    return {
        "success": True,
        "test_id": test_id,
        "message": f"A/B测试已创建，流量分配: A={traffic_split_a}%, B={traffic_split_b}%"
    }


@app.get("/api/projects/{project_id}/ab-tests")
async def list_ab_tests(project_id: str):
    """获取项目的 A/B 测试列表"""
    rows = execute_query('''
        SELECT t.*, 
               ja.model_name as model_a_name,
               jb.model_name as model_b_name
        FROM ab_tests t
        JOIN training_jobs ja ON t.model_a_id = ja.id
        JOIN training_jobs jb ON t.model_b_id = jb.id
        WHERE t.project_id = %s
        ORDER BY t.created_at DESC
    ''', (project_id,))
    
    tests = []
    for row in rows:
        tests.append({
            "id": row['id'],
            "name": row['name'],
            "description": row['description'],
            "status": row['status'],
            "model_a": {"id": row['model_a_id'], "name": row['model_a_name']},
            "model_b": {"id": row['model_b_id'], "name": row['model_b_name']},
            "traffic_split": {"a": row['traffic_split_a'], "b": row['traffic_split_b']},
            "calls": {"a": row['model_a_calls'], "b": row['model_b_calls']},
            "success": {"a": row['model_a_success'], "b": row['model_b_success']},
            "winner": row['winner_model'],
            "created_at": row['created_at']
        })
    
    return {"tests": tests, "count": len(tests)}


@app.get("/api/projects/{project_id}/ab-tests/{test_id}")
async def get_ab_test_detail(project_id: str, test_id: str):
    """获取 A/B 测试详情和统计结果"""
    rows = execute_query('''
        SELECT t.*, 
               ja.model_name as model_a_name,
               jb.model_name as model_b_name
        FROM ab_tests t
        JOIN training_jobs ja ON t.model_a_id = ja.id
        JOIN training_jobs jb ON t.model_b_id = jb.id
        WHERE t.id = %s AND t.project_id = %s
    ''', (test_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="A/B测试不存在")
    
    test = rows[0]
    
    # 获取详细统计
    stats = execute_query('''
        SELECT 
            variant,
            COUNT(*) as total_calls,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_calls,
            AVG(latency_ms) as avg_latency,
            SUM(CASE WHEN prediction_correct THEN 1 ELSE 0 END) as correct_predictions
        FROM ab_test_calls
        WHERE test_id = %s
        GROUP BY variant
    ''', (test_id,))
    
    stats_dict = {row['variant']: dict(row) for row in stats}
    
    # 计算置信度（简化版，实际应该用统计检验）
    a_success = test['model_a_success']
    b_success = test['model_b_success']
    a_calls = test['model_a_calls']
    b_calls = test['model_b_calls']
    
    a_rate = a_success / a_calls if a_calls > 0 else 0
    b_rate = b_success / b_calls if b_calls > 0 else 0
    
    return {
        "test": {
            "id": test['id'],
            "name": test['name'],
            "status": test['status'],
            "model_a": {"id": test['model_a_id'], "name": test['model_a_name']},
            "model_b": {"id": test['model_b_id'], "name": test['model_b_name']},
            "traffic_split": {"a": test['traffic_split_a'], "b": test['traffic_split_b']},
            "min_sample_size": test['min_sample_size']
        },
        "statistics": {
            "model_a": stats_dict.get('A', {}),
            "model_b": stats_dict.get('B', {})
        },
        "comparison": {
            "a_success_rate": round(a_rate, 4),
            "b_success_rate": round(b_rate, 4),
            "difference": round(b_rate - a_rate, 4),
            "improvement_percent": round((b_rate - a_rate) / a_rate * 100, 2) if a_rate > 0 else 0,
            "total_calls": a_calls + b_calls,
            "has_winner": test['winner_model'] is not None,
            "winner": test['winner_model'],
            "confidence_level": test['confidence_level']
        }
    }


@app.post("/api/projects/{project_id}/ab-tests/{test_id}/predict")
async def ab_test_predict(
    project_id: str,
    test_id: str,
    text: str = Form(...),
    user_id: str = Form(None)
):
    """
    A/B 测试预测 - 根据流量分配自动路由到模型A或B
    
    如果有user_id，确保同一用户始终分配到同一组（一致性哈希）
    """
    # 获取测试信息
    test_rows = execute_query('''
        SELECT * FROM ab_tests 
        WHERE id = %s AND project_id = %s AND status = 'running'
    ''', (test_id, project_id))
    
    if not test_rows:
        raise HTTPException(status_code=404, detail="A/B测试不存在或已停止")
    
    test = test_rows[0]
    
    # 决定分配到哪个组
    import hashlib
    if user_id:
        # 使用 user_id 做一致性哈希，确保同一用户始终分配到同一组
        hash_val = int(hashlib.md5(f"{test_id}:{user_id}".encode()).hexdigest(), 16)
        bucket = hash_val % 100
        variant = 'A' if bucket < test['traffic_split_a'] else 'B'
    else:
        # 随机分配
        import random
        variant = 'A' if random.randint(1, 100) <= test['traffic_split_a'] else 'B'
    
    model_id = test['model_a_id'] if variant == 'A' else test['model_b_id']
    
    # 获取模型路径并加载
    job_rows = execute_query('SELECT model_path FROM training_jobs WHERE id = %s', (model_id,))
    if not job_rows:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    model_path = job_rows[0]['model_path']
    
    # 加载模型并预测
    import time
    start_time = time.time()
    
    try:
        inference_model_id = f"{project_id}/{model_id}"
        inference_service.load_model(model_path, inference_model_id)
        result = inference_service.predict([text], inference_model_id)[0]
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # 记录调用
        execute_update('''
            INSERT INTO ab_test_calls 
            (test_id, variant, model_id, input_data, output_data, latency_ms, user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (
            test_id, variant, model_id,
            json.dumps({"text": text}),
            json.dumps(result),
            latency_ms, user_id
        ))
        
        # 更新统计计数
        if variant == 'A':
            execute_update('UPDATE ab_tests SET model_a_calls = model_a_calls + 1 WHERE id = %s', (test_id,))
        else:
            execute_update('UPDATE ab_tests SET model_b_calls = model_b_calls + 1 WHERE id = %s', (test_id,))
        
        return {
            "success": True,
            "variant": variant,
            "model_used": model_id[:8],
            "result": result,
            "latency_ms": latency_ms
        }
        
    except Exception as e:
        # 记录失败
        execute_update('''
            INSERT INTO ab_test_calls 
            (test_id, variant, model_id, input_data, success, error_message)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (test_id, variant, model_id, json.dumps({"text": text}), False, str(e)))
        
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/api/projects/{project_id}/ab-tests/{test_id}/feedback")
async def submit_ab_test_feedback(
    project_id: str,
    test_id: str,
    call_id: int = Form(...),
    is_correct: bool = Form(...)
):
    """提交 A/B 测试预测结果反馈（用于计算准确率）"""
    # 验证测试存在
    test_rows = execute_query('SELECT id FROM ab_tests WHERE id = %s AND project_id = %s', 
                             (test_id, project_id))
    if not test_rows:
        raise HTTPException(status_code=404, detail="A/B测试不存在")
    
    # 更新反馈
    execute_update('''
        UPDATE ab_test_calls 
        SET prediction_correct = %s 
        WHERE id = %s AND test_id = %s
    ''', (is_correct, call_id, test_id))
    
    # 更新成功计数
    call_rows = execute_query('SELECT variant FROM ab_test_calls WHERE id = %s', (call_id,))
    if call_rows:
        variant = call_rows[0]['variant']
        if is_correct:
            if variant == 'A':
                execute_update('UPDATE ab_tests SET model_a_success = model_a_success + 1 WHERE id = %s', (test_id,))
            else:
                execute_update('UPDATE ab_tests SET model_b_success = model_b_success + 1 WHERE id = %s', (test_id,))
    
    return {"success": True, "message": "反馈已记录"}


@app.post("/api/projects/{project_id}/ab-tests/{test_id}/stop")
async def stop_ab_test(
    project_id: str,
    test_id: str,
    winner: str = Form(None)  # 'model_a', 'model_b', 'tie'
):
    """停止 A/B 测试并声明获胜方"""
    rows = execute_query('SELECT * FROM ab_tests WHERE id = %s AND project_id = %s',
                        (test_id, project_id))
    if not rows:
        raise HTTPException(status_code=404, detail="A/B测试不存在")
    
    test = rows[0]
    
    # 计算最终统计
    a_rate = test['model_a_success'] / test['model_a_calls'] if test['model_a_calls'] > 0 else 0
    b_rate = test['model_b_success'] / test['model_b_calls'] if test['model_b_calls'] > 0 else 0
    
    # 如果没有指定获胜方，自动判断
    if not winner:
        if a_rate > b_rate:
            winner = 'model_a'
        elif b_rate > a_rate:
            winner = 'model_b'
        else:
            winner = 'tie'
    
    improvement = abs(b_rate - a_rate) / max(a_rate, 0.0001) * 100
    
    execute_update('''
        UPDATE ab_tests 
        SET status = 'completed',
            winner_model = %s,
            ended_at = CURRENT_TIMESTAMP,
            improvement_percent = %s
        WHERE id = %s
    ''', (winner, improvement, test_id))
    
    return {
        "success": True,
        "message": f"A/B测试已停止，获胜方: {winner}",
        "result": {
            "winner": winner,
            "model_a_rate": round(a_rate, 4),
            "model_b_rate": round(b_rate, 4),
            "improvement_percent": round(improvement, 2)
        }
    }


@app.delete("/api/projects/{project_id}/ab-tests/{test_id}")
async def delete_ab_test(project_id: str, test_id: str):
    """删除 A/B 测试（连带所有调用记录）"""
    rows = execute_query('SELECT id FROM ab_tests WHERE id = %s AND project_id = %s',
                        (test_id, project_id))
    if not rows:
        raise HTTPException(status_code=404, detail="A/B测试不存在")
    
    # 删除调用记录
    execute_update('DELETE FROM ab_test_calls WHERE test_id = %s', (test_id,))
    
    # 删除测试
    execute_update('DELETE FROM ab_tests WHERE id = %s', (test_id,))
    
    return {"success": True, "message": "A/B测试已删除"}


# ============ 模型压缩 API ============

from model_compression import quantize_model_pytorch, export_to_onnx, get_model_info

@app.post("/api/projects/{project_id}/models/{job_id}/compress")
async def compress_model(
    project_id: str,
    job_id: str,
    compression_type: str = Form("quantize"),  # quantize, onnx
    quantization_type: str = Form("dynamic")   # dynamic, static
):
    """
    模型压缩 - 量化或ONNX导出
    
    Args:
        compression_type: 'quantize'(PyTorch量化), 'onnx'(ONNX导出)
        quantization_type: 'dynamic'(动态量化), 'static'(静态量化)
    """
    # 验证模型存在
    rows = execute_query('''
        SELECT model_path, status, model_name FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    job = rows[0]
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="模型训练尚未完成")
    
    model_path = Path(job['model_path'])
    
    # 查找模型文件
    model_file = model_path / "pytorch_model.bin"
    if not model_file.exists():
        model_file = model_path / "model.pt"
    if not model_file.exists():
        model_file = model_path / "best_model.pth"
    
    if not model_file.exists():
        raise HTTPException(status_code=404, detail="找不到模型文件")
    
    # 创建压缩目录
    compressed_dir = model_path / "compressed"
    compressed_dir.mkdir(exist_ok=True)
    
    # 执行压缩
    if compression_type == "quantize":
        output_file = compressed_dir / f"model_quantized_{quantization_type}.pth"
        result = quantize_model_pytorch(
            str(model_file), 
            str(output_file), 
            quantization_type
        )
        
    elif compression_type == "onnx":
        output_file = compressed_dir / "model.onnx"
        result = export_to_onnx(str(model_file), str(output_file))
        
    else:
        raise HTTPException(status_code=400, detail=f"不支持的压缩类型: {compression_type}")
    
    if result.get("success"):
        result["compressed_path"] = str(output_file)
        result["download_url"] = f"/api/projects/{project_id}/models/{job_id}/download-compressed"
    
    return result


@app.get("/api/projects/{project_id}/models/{job_id}/info")
async def get_model_information(project_id: str, job_id: str):
    """获取模型信息（大小、参数数量）"""
    rows = execute_query('''
        SELECT model_path, status FROM training_jobs 
        WHERE id = %s AND project_id = %s
    ''', (job_id, project_id))
    
    if not rows:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    model_path = Path(rows[0]['model_path'])
    
    # 查找模型文件
    info = {"files": []}
    for pattern in ["pytorch_model.bin", "model.pt", "best_model.pth", "model.onnx", "*.pth"]:
        files = list(model_path.glob(pattern))
        for f in files:
            file_info = get_model_info(str(f))
            if "error" not in file_info:
                info["files"].append(file_info)
    
    # 如果有压缩目录，也包含压缩后的文件
    compressed_dir = model_path / "compressed"
    if compressed_dir.exists():
        for f in compressed_dir.glob("*"):
            file_info = get_model_info(str(f))
            file_info["is_compressed"] = True
            info["files"].append(file_info)
    
    info["total_files"] = len(info["files"])
    info["model_path"] = str(model_path)
    
    return info


# ============ 预训练模型库 API ============

PRETRAINED_MODELS = {
    "nlp": {
        "distilbert-base-chinese": {
            "name": "DistilBERT-Base-Chinese",
            "description": "轻量级中文BERT，速度快3倍，保留97%精度",
            "size_mb": 300,
            "params_millions": 66,
            "tasks": ["文本分类", "情感分析", "命名实体识别"],
            "recommended": True
        },
        "bert-base-chinese": {
            "name": "BERT-Base-Chinese",
            "description": "Google官方中文BERT，精度高",
            "size_mb": 400,
            "params_millions": 110,
            "tasks": ["文本分类", "问答", "语义匹配"],
            "recommended": False
        },
        "chinese-roberta-wwm-ext": {
            "name": "Chinese-RoBERTa-WWM-Ext",
            "description": "哈工大版RoBERTa，中文效果更好",
            "size_mb": 400,
            "params_millions": 110,
            "tasks": ["文本分类", "阅读理解", "文本生成"],
            "recommended": False
        }
    },
    "vision": {
        "resnet50": {
            "name": "ResNet-50",
            "description": "经典图像分类模型，精度速度平衡",
            "size_mb": 100,
            "params_millions": 25,
            "tasks": ["图像分类", "特征提取"],
            "recommended": True
        },
        "efficientnet_b0": {
            "name": "EfficientNet-B0",
            "description": "高效图像分类模型，参数更少精度更高",
            "size_mb": 20,
            "params_millions": 5.3,
            "tasks": ["图像分类", "移动端部署"],
            "recommended": True
        },
        "mobilenet_v2": {
            "name": "MobileNet-V2",
            "description": "移动端首选，超轻量级",
            "size_mb": 13,
            "params_millions": 3.5,
            "tasks": ["移动端图像分类", "边缘设备"],
            "recommended": True
        },
        "yolov8n": {
            "name": "YOLOv8-Nano",
            "description": "超轻量目标检测模型",
            "size_mb": 6,
            "params_millions": 3.2,
            "tasks": ["目标检测", "缺陷检测"],
            "recommended": True
        }
    },
    "tabular": {
        "random_forest": {
            "name": "Random Forest",
            "description": "经典表格数据模型，可解释性强",
            "size_mb": 10,
            "tasks": ["分类", "回归", "特征重要性"],
            "recommended": True
        },
        "xgboost": {
            "name": "XGBoost",
            "description": "梯度提升树，表格数据首选",
            "size_mb": 5,
            "tasks": ["分类", "回归", "排序"],
            "recommended": True
        }
    }
}


@app.get("/api/models/pretrained")
async def list_pretrained_models(category: str = None):
    """
    获取预训练模型列表
    
    Args:
        category: 'nlp', 'vision', 'tabular'，不传返回全部
    """
    if category:
        if category not in PRETRAINED_MODELS:
            raise HTTPException(status_code=400, detail=f"不支持的类别: {category}")
        return {
            "category": category,
            "models": PRETRAINED_MODELS[category]
        }
    
    return {
        "categories": list(PRETRAINED_MODELS.keys()),
        "models": PRETRAINED_MODELS
    }


@app.post("/api/models/pretrained/download")
async def download_pretrained_model(
    model_id: str = Form(...),
    cache_dir: str = Form("/var/www/ai-training/pretrained_models")
):
    """
    下载预训练模型到本地缓存
    
    Args:
        model_id: 模型ID，如 'distilbert-base-chinese'
        cache_dir: 本地缓存目录
    """
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    cache_path = Path(cache_dir) / model_id
    cache_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if model_id in PRETRAINED_MODELS.get("nlp", {}):
            # 下载 Transformer 模型
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"正在下载 {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
            
            # 保存到本地
            tokenizer.save_pretrained(str(cache_path))
            model.save_pretrained(str(cache_path))
            
            return {
                "success": True,
                "model_id": model_id,
                "cache_path": str(cache_path),
                "message": f"模型 {model_id} 已下载到本地缓存"
            }
            
        else:
            return {
                "success": False,
                "error": f"暂不支持下载模型: {model_id}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"下载失败: {str(e)}"
        }


@app.get("/api/models/pretrained/{model_id}/info")
async def get_pretrained_model_info(model_id: str):
    """获取预训练模型详细信息"""
    for category, models in PRETRAINED_MODELS.items():
        if model_id in models:
            info = models[model_id].copy()
            info["id"] = model_id
            info["category"] = category
            
            # 检查本地缓存
            cache_path = Path("/var/www/ai-training/pretrained_models") / model_id
            info["is_cached"] = cache_path.exists()
            if cache_path.exists():
                info["cache_path"] = str(cache_path)
            
            return info
    
    raise HTTPException(status_code=404, detail="模型不存在")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
