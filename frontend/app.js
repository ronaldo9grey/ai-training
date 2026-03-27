const { useState, useEffect, useRef } = React;
const API_BASE = '/ai-training/api';

// 帮助文档组件
const HelpTooltip = ({ title, content }) => {
  const [show, setShow] = useState(false);
  return (
    <span className="relative inline-block ml-1">
      <span 
        className="text-gray-400 cursor-help hover:text-purple-500"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
      >
        ⓘ
      </span>
      {show && (
        <div className="absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64 p-3 bg-gray-800 text-white text-xs rounded-lg shadow-lg">
          <div className="font-semibold mb-1">{title}</div>
          <div className="text-gray-200">{content}</div>
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-800"></div>
        </div>
      )}
    </span>
  );
};

// 模型说明配置
const MODEL_HELP = {
  "bert-base-chinese": {
    name: "BERT-Base-Chinese",
    desc: "谷歌发布的中文基础BERT模型，110M参数",
    pros: "准确率高，适合大多数任务",
    cons: "模型较大，推理稍慢"
  },
  "distilbert-base-chinese": {
    name: "DistilBERT-Chinese",
    desc: "BERT的蒸馏版本，保留97%性能但只有66M参数",
    pros: "速度快，内存占用小",
    cons: "准确率略低于BERT"
  }
};

// 参数说明配置
const PARAM_HELP = {
  learning_rate: {
    title: "学习率",
    desc: "模型每次更新权重的步长。太大：训练不稳定；太小：收敛慢。建议：2e-5 ~ 5e-5"
  },
  batch_size: {
    title: "批次大小",
    desc: "每次迭代处理的样本数。显存大可以设大点。建议：8-32"
  },
  epochs: {
    title: "训练轮数",
    desc: "完整遍历数据集的次数。太少：欠拟合；太多：过拟合。建议：3-10"
  },
  max_length: {
    title: "最大长度",
    desc: "输入文本的最大token数。超长会被截断。建议：128-512"
  }
};

// 图标组件
const Icons = {
  Plus: () => '➕',
  Folder: () => '📁',
  Upload: () => '📤',
  Train: () => '🚂',
  Chart: () => '📊',
  Tag: () => '🏷️',
  Rocket: () => '🚀',
  Brain: () => '🧠',
  Database: () => '🗄️',
  Settings: () => '⚙️',
  Check: () => '✅',
  Loading: () => '⏳',
  Error: () => '❌',
  Download: () => '⬇️',
  Eye: () => '👁️',
  Help: () => '❓',
  Close: () => '✕',
  TrendingUp: () => '📈',
  Activity: () => '📉',
  Award: () => '🏆',
  Trash: () => '🗑️',
  Compare: () => '⚖️',
  Clock: () => '⏰',
  File: () => '📄',
  Bell: () => '🔔',
  Memory: () => '🧠',
  Sparkles: () => '✨',
  Refresh: () => '🔄',
  Edit: () => '✏️',
  Image: () => '🖼️',
  Server: () => '🖥️'
};

// 训练配置模板
const TRAINING_TEMPLATES = {
  fast: {
    name: "⚡ 快速训练",
    desc: "适合快速验证，3分钟出结果",
    config: {
      model_name: "distilbert-base-chinese",
      epochs: 2,
      batch_size: 16,
      learning_rate: 5e-5,
      max_length: 128,
      early_stopping: true,
      early_stopping_patience: 1,
      early_stopping_threshold: 0.001,
      lr_scheduler_type: "linear",
      warmup_ratio: 0.1
    }
  },
  balanced: {
    name: "⚖️ 均衡配置",
    desc: "速度与精度的平衡，推荐日常使用",
    config: {
      model_name: "bert-base-chinese",
      epochs: 3,
      batch_size: 8,
      learning_rate: 2e-5,
      max_length: 128,
      early_stopping: true,
      early_stopping_patience: 2,
      early_stopping_threshold: 0.001,
      lr_scheduler_type: "cosine",
      warmup_ratio: 0.1
    }
  },
  accurate: {
    name: "🎯 高精度",
    desc: "追求最高准确率，训练时间较长",
    config: {
      model_name: "chinese-roberta-wwm-ext",
      epochs: 5,
      batch_size: 4,
      learning_rate: 1e-5,
      max_length: 256,
      early_stopping: true,
      early_stopping_patience: 3,
      early_stopping_threshold: 0.0005,
      lr_scheduler_type: "cosine_with_restarts",
      warmup_ratio: 0.2
    }
  }
};

// 学习率调度器说明
const LR_SCHEDULER_HELP = {
  linear: { name: "线性衰减", desc: "学习率从初始值线性下降到0，最稳定" },
  cosine: { name: "余弦退火", desc: "学习率按余弦曲线下降，收敛平滑" },
  cosine_with_restarts: { name: "余弦重启", desc: "周期性重启学习率，可能找到更好解" },
  polynomial: { name: "多项式衰减", desc: "学习率按多项式曲线下降" },
  constant: { name: "恒定不变", desc: "学习率保持不变，适合微调" }
};

// 早停阈值说明
const EARLY_STOPPING_HELP = {
  patience: { title: "耐心值", desc: "连续多少轮验证指标不提升就停止。太小容易过早停止，太大训练时间长。" },
  threshold: { title: "改善阈值", desc: "指标提升多少才算改善。越小越严格，越大越宽松。" }
};

// 主应用
function App() {
  const [currentView, setCurrentView] = useState('projects');
  const [currentProject, setCurrentProject] = useState(null);
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadProjects();
    
    // 检查URL参数，如果有project参数则自动打开对应项目
    const urlParams = new URLSearchParams(window.location.search);
    const projectId = urlParams.get('project');
    if (projectId) {
      // 等待项目列表加载完成后再尝试打开
      const checkAndOpen = setInterval(() => {
        if (projects.length > 0) {
          const targetProject = projects.find(p => p.id === projectId);
          if (targetProject) {
            setCurrentProject(targetProject);
            setCurrentView('project-detail');
            clearInterval(checkAndOpen);
          }
        }
      }, 500);
      // 10秒后停止尝试
      setTimeout(() => clearInterval(checkAndOpen), 10000);
    }
  }, [projects]);

  const loadProjects = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects`);
      const data = await res.json();
      setProjects(data.projects || []);
    } catch (e) {
      console.error('加载项目失败:', e);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-xl border-b border-gray-200/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-xl flex items-center justify-center text-white text-xl">
                <Icons.Brain />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-600 to-purple-600 bg-clip-text text-transparent">
                  AI模型训练工坊
                </h1>
                <p className="text-xs text-gray-500">从数据到部署的完整MLOps平台</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button 
                onClick={() => {setCurrentView('projects'); setCurrentProject(null);}}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  currentView === 'projects' 
                    ? 'bg-purple-100 text-purple-700' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Icons.Folder /> 项目
              </button>
              <a 
                href="/ai-training/docs/index.html"
                target="_blank"
                className="px-3 py-1.5 rounded-lg text-sm font-medium text-gray-600 hover:bg-gray-100 transition-all flex items-center gap-1"
              >
                <Icons.File /> 文档
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {currentView === 'projects' && (
          <ProjectList 
            projects={projects} 
            onRefresh={loadProjects}
            onSelect={(p) => { setCurrentProject(p); setCurrentView('project-detail'); }}
          />
        )}
        {currentView === 'project-detail' && currentProject && (
          <ProjectDetail 
            project={currentProject} 
            onBack={() => setCurrentView('projects')}
          />
        )}
      </main>
    </div>
  );
}

// 项目列表
// 任务类型配置
const TASK_TYPES = {
  text_classification: { name: '文本分类', icon: '📝', desc: '对文本进行分类，如情感分析、主题分类' },
  classification: { name: '表格分类', icon: '📊', desc: '对结构化数据进行分类，如故障/正常' },
  regression: { name: '数值预测', icon: '📈', desc: '预测连续数值，如产量、能耗' },
  anomaly_detection: { name: '异常检测', icon: '🔍', desc: '发现数据中的异常模式，无需标注' },
  time_series: { name: '时序分析', icon: '⏱️', desc: '分析时间序列数据，趋势预测和RUL' },
  image_classification: { name: '图像分类', icon: '🖼️', desc: '训练图像分类模型，如缺陷检测、场景识别' },
  object_detection: { name: '目标检测', icon: '🎯', desc: '检测图片中物体位置，如缺陷定位、计数' }
};

function ProjectList({ projects, onRefresh, onSelect }) {
  const [showCreate, setShowCreate] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newTaskType, setNewTaskType] = useState('text_classification');
  const [creating, setCreating] = useState(false);

  const createProject = async () => {
    if (!newProjectName.trim()) return;
    setCreating(true);
    try {
      const res = await fetch(`${API_BASE}/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newProjectName,
          description: TASK_TYPES[newTaskType].desc,
          task_type: newTaskType
        })
      });
      if (res.ok) {
        setNewProjectName('');
        setShowCreate(false);
        onRefresh();
      }
    } catch (e) {
      alert('创建失败: ' + e.message);
    }
    setCreating(false);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">训练项目</h2>
          <p className="text-gray-500 mt-1">管理您的AI模型训练项目</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-purple-600 text-white rounded-xl font-medium hover:shadow-lg transition-all flex items-center gap-2"
        >
          <Icons.Plus /> 新建项目
        </button>
      </div>

      {showCreate && (
        <div className="mb-6 bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
          <h3 className="font-semibold text-gray-800 mb-4">创建新项目</h3>
          
          {/* 任务类型选择 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            {Object.entries(TASK_TYPES).map(([key, task]) => (
              <button
                key={key}
                onClick={() => setNewTaskType(key)}
                className={`p-3 rounded-xl border-2 text-left transition-all ${
                  newTaskType === key
                    ? 'border-purple-500 bg-purple-50'
                    : 'border-gray-200 hover:border-purple-300'
                }`}
              >
                <div className="text-2xl mb-1">{task.icon}</div>
                <div className="font-medium text-gray-800 text-sm">{task.name}</div>
                <div className="text-xs text-gray-500 mt-1 line-clamp-2">{task.desc}</div>
              </button>
            ))}
          </div>
          
          <div className="flex gap-3">
            <input
              type="text"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="输入项目名称"
              className="flex-1 px-4 py-2 rounded-xl border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
              onKeyPress={(e) => e.key === 'Enter' && createProject()}
            />
            <button
              onClick={createProject}
              disabled={creating}
              className="px-4 py-2 bg-purple-600 text-white rounded-xl font-medium hover:bg-purple-700 disabled:opacity-50"
            >
              {creating ? <Icons.Loading /> : '创建'}
            </button>
            <button
              onClick={() => setShowCreate(false)}
              className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-xl"
            >
              取消
            </button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {projects.map(project => (
          <div
            key={project.id}
            onClick={() => onSelect(project)}
            className="bg-white/80 backdrop-blur-xl rounded-2xl p-5 border border-gray-200/50 shadow-sm hover:shadow-lg transition-all cursor-pointer group"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="w-12 h-12 bg-gradient-to-br from-cyan-100 to-purple-100 rounded-xl flex items-center justify-center text-2xl group-hover:scale-110 transition-transform">
                <Icons.Database />
              </div>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                project.status === 'completed' ? 'bg-green-100 text-green-700' :
                project.status === 'training' ? 'bg-blue-100 text-blue-700' :
                'bg-gray-100 text-gray-600'
              }`}>
                {project.status === 'created' ? '未开始' : 
                 project.status === 'training' ? '训练中' : '已完成'}
              </span>
            </div>
            <h3 className="font-semibold text-gray-800 mb-1">{project.name}</h3>
            <p className="text-sm text-gray-500 mb-3">
              {project.description || '文本分类项目'}
            </p>
            <div className="flex items-center justify-between text-xs text-gray-400">
              <span>创建于 {new Date(project.created_at).toLocaleDateString()}</span>
              <span className="group-hover:text-purple-600 transition-colors">查看详情 →</span>
            </div>
          </div>
        ))}
      </div>

      {projects.length === 0 && (
        <div className="text-center py-12">
          <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-cyan-100 to-purple-100 rounded-2xl flex items-center justify-center text-4xl">
            <Icons.Folder />
          </div>
          <h3 className="text-lg font-medium text-gray-700 mb-1">还没有项目</h3>
          <p className="text-gray-500 mb-4">创建您的第一个AI训练项目</p>
          <button
            onClick={() => setShowCreate(true)}
            className="px-4 py-2 bg-purple-600 text-white rounded-xl font-medium hover:bg-purple-700"
          >
            创建项目
          </button>
        </div>
      )}
    </div>
  );
}

// 项目详情
function ProjectDetail({ project, onBack }) {
  const [activeTab, setActiveTab] = useState('data');
  const [currentProject, setCurrentProject] = useState(project);
  const [datasets, setDatasets] = useState([]);
  const [jobs, setJobs] = useState([]);

  useEffect(() => {
    loadDatasets();
    loadJobs();
  }, [project.id]);

  const loadDatasets = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${project.id}/datasets`);
      const data = await res.json();
      setDatasets(data.datasets || []);
    } catch (e) {}
  };

  const loadJobs = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${project.id}/jobs`);
      const data = await res.json();
      setJobs(data.jobs || []);
    } catch (e) {}
  };

  return (
    <div>
      <button
        onClick={onBack}
        className="mb-4 px-3 py-1.5 text-gray-600 hover:bg-white/50 rounded-lg flex items-center gap-1 transition-all"
      >
        ← 返回项目列表
      </button>

      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg mb-6">
        <div className="flex flex-col gap-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-800">{project.name}</h2>
            <p className="text-gray-500 mt-1">ID: {project.id}</p>
          </div>
          
          {/* 核心流程 */}
          <div>
            <p className="text-base text-gray-500 mb-3 font-semibold flex items-center gap-2">
              核心流程
              <span className="text-xs bg-purple-100 text-purple-600 px-2 py-1 rounded-full">按顺序执行</span>
            </p>
            <div className="flex items-center gap-1 overflow-x-auto pb-2">
              {[
                { id: 'data', icon: <Icons.Upload />, label: '数据', step: 1, desc: '上传数据集' },
                { id: 'annotate', icon: <Icons.Edit />, label: '标注', step: 2, desc: '标注数据' },
                { id: 'train', icon: <Icons.Train />, label: '训练', step: 3, desc: '训练模型' },
                { id: 'deploy', icon: <Icons.Rocket />, label: '部署', step: 4, desc: '部署服务' },
                { id: 'test', icon: <Icons.Eye />, label: '测试', step: 5, desc: '测试应用' },
              ].map((item, idx, arr) => (
                <React.Fragment key={item.id}>
                  <button 
                    onClick={() => setActiveTab(item.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all whitespace-nowrap ${
                      activeTab === item.id 
                        ? 'bg-purple-100 text-purple-700 ring-2 ring-purple-300' 
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                    title={item.desc}
                  >
                    <span className="flex items-center justify-center w-5 h-5 text-xs bg-white/50 rounded-full font-bold">
                      {item.step}
                    </span>
                    {item.icon}
                    <span>{item.label}</span>
                  </button>
                  {idx < arr.length - 1 && (
                    <span className="text-gray-300 px-1">→</span>
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
          
          {/* 高级工具 - 动态显示关联 */}
          <div>
            <p className="text-xs text-gray-400 mb-2 font-medium flex items-center gap-2">
              高级工具
              <span className="text-[10px] bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full">按阶段可用</span>
            </p>
            <div className="flex gap-2 overflow-x-auto pb-2">
              {[
                { 
                  id: 'timeseries', 
                  icon: <Icons.Chart />, 
                  label: '时序分析', 
                  stage: '数据处理',
                  stageColor: 'blue',
                  desc: '时间序列特征提取',
                  availableIn: ['data', 'annotate']
                },
                { 
                  id: 'modelservices', 
                  icon: <Icons.Server />, 
                  label: '模型服务', 
                  stage: '部署后',
                  stageColor: 'indigo',
                  desc: '管理推理服务和监控',
                  availableIn: ['deploy', 'test']
                },
                { 
                  id: 'xai', 
                  icon: <Icons.Help />, 
                  label: '可解释性', 
                  stage: '训练后',
                  stageColor: 'green',
                  desc: '分析模型决策原因',
                  availableIn: ['train', 'deploy', 'test']
                },
                { 
                  id: 'onlinelearning', 
                  icon: <Icons.Brain />, 
                  label: '在线学习', 
                  stage: '部署后',
                  stageColor: 'purple',
                  desc: '持续学习新数据',
                  availableIn: ['deploy', 'test']
                },
                { 
                  id: 'alert', 
                  icon: <Icons.Bell />, 
                  label: '告警', 
                  stage: '部署后',
                  stageColor: 'red',
                  desc: '模型性能监控告警',
                  availableIn: ['deploy', 'test']
                },
              ].map(tool => {
                const isAvailable = tool.availableIn.includes(activeTab);
                const isActive = activeTab === tool.id;
                return (
                  <button 
                    key={tool.id}
                    onClick={() => setActiveTab(tool.id)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
                      isActive 
                        ? 'bg-cyan-100 text-cyan-700 ring-2 ring-cyan-300' 
                        : isAvailable
                          ? 'text-gray-600 hover:bg-gray-100 border border-gray-200'
                          : 'text-gray-300 cursor-not-allowed opacity-60'
                    }`}
                    title={`${tool.desc} · ${tool.stage}可用`}
                    disabled={!isAvailable && !isActive}
                  >
                    {tool.icon}
                    <span>{tool.label}</span>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded bg-${tool.stageColor}-100 text-${tool.stageColor}-600`}>
                      {tool.stage}
                    </span>
                  </button>
                );
              })}
            </div>
            {/* 当前阶段提示 */}
            <p className="text-xs text-gray-400 mt-1">
              {activeTab === 'data' && '💡 当前阶段可用: 时序分析'}
              {activeTab === 'annotate' && '💡 当前阶段可用: 时序分析'}
              {activeTab === 'train' && '💡 当前阶段可用: AutoML超参优化、可解释性'}
              {activeTab === 'deploy' && '💡 当前阶段可用: 可解释性、模型服务、在线学习、告警'}
              {activeTab === 'test' && '💡 当前阶段可用: 可解释性、模型服务、在线学习、告警'}
              {['data', 'annotate', 'train', 'deploy', 'test'].includes(activeTab) ? '' : '💡 在核心流程中使用高级工具'}
            </p>
          </div>
        </div>
      </div>

      {activeTab === 'data' && (
        <DataTab projectId={project.id} datasets={datasets} onRefresh={loadDatasets} />
      )}
      {activeTab === 'train' && (
        <TrainTab projectId={project.id} projectType={project.task_type} datasets={datasets} jobs={jobs} onRefresh={loadJobs} />
      )}
      {activeTab === 'deploy' && (
        <DeployTab projectId={project.id} projectType={project.task_type} datasets={datasets} jobs={jobs} />
      )}
      {activeTab === 'test' && (
        <TestTab projectId={project.id} projectType={project.task_type} jobs={jobs} />
      )}
      {activeTab === 'notification' && (
        <NotificationTab projectId={project.id} />
      )}
      {activeTab === 'timeseries' && (
        <TimeSeriesTab projectId={project.id} datasets={datasets} />
      )}
      {activeTab === 'alert' && (
        <AlertTab projectId={project.id} />
      )}
      {activeTab === 'annotate' && (
        <AnnotationTab projectId={project.id} projectType={project.task_type} datasets={datasets} />
      )}
      {activeTab === 'onlinelearning' && (
        <OnlineLearningTab projectId={project.id} jobs={jobs} datasets={datasets} />
      )}
      {activeTab === 'xai' && (
        <XAITab projectId={project.id} jobs={jobs} projectType={project.task_type} />
      )}
      {activeTab === 'modelservices' && (
        <ModelServicesTab projectId={project.id} jobs={jobs} />
      )}

    </div>
  );
}

// 数据标注Tab
function AnnotationTab({ projectId, projectType, datasets }) {
  const [activeDataset, setActiveDataset] = useState(null);
  const [samples, setSamples] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [labels, setLabels] = useState([]);
  const [newLabel, setNewLabel] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({ total: 0, labeled: 0, unlabeled: 0 });
  const [previewData, setPreviewData] = useState(null);
  
  // 本次会话标注统计（新增）
  const [sessionStats, setSessionStats] = useState({ annotated: 0, modified: 0 });
  const initialLabelsRef = useRef({}); // 记录初始标签状态
  
  // 画框标注相关状态
  const [imageAnnotations, setImageAnnotations] = useState({}); // { imageIndex: [ {x, y, width, height, label, id}, ... ] }
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState(null);
  const [drawEnd, setDrawEnd] = useState(null);
  const [selectedBox, setSelectedBox] = useState(null);
  const [activeLabel, setActiveLabel] = useState('');
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  // 判断是否为图片数据集 - 优先根据项目类型判断
  const isImageDataset = (dataset) => {
    // 首先根据项目类型判断（最准确）
    if (projectType === 'object_detection' || projectType === 'image_classification') {
      return true;
    }
    // 文本/表格类项目一定不是图片
    if (projectType === 'text_classification' || projectType === 'nlp' || projectType === 'classification' || projectType === 'regression' || projectType === 'time_series') {
      return false;
    }
    // 根据数据集类型判断
    return dataset && dataset.file_type === 'image_folder' || 
           dataset && dataset.file_type === 'image' ||
           dataset && (dataset.name) && dataset.name.toLowerCase().includes('image') ||
           dataset && (dataset.name) && dataset.name.toLowerCase().includes('图片');
  };

  // 加载数据集样本
  const loadSamples = async (dataset) => {
    if (!dataset) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/${dataset.id}/preview?limit=100`);
      const data = await res.json();
      console.log('Dataset preview:', data);
      
      if (res.ok && data.preview && data.preview.length > 0) {
        // 处理图片数据集
        if (data.type === 'image_folder' || isImageDataset(dataset)) {
          const imageSamples = data.preview.map((row, idx) => ({
            id: idx,
            file_path: row['图片路径'] || row.file_path || row.path || Object.values(row)[0],
            label: row['类别文件夹'] || row.label || row.category || null,
            raw: row
          }));
          setSamples(imageSamples);
          setStats({
            total: imageSamples.length,
            labeled: imageSamples.filter(s => s.label && s.label !== '根目录').length,
            unlabeled: imageSamples.filter(s => !s.label || s.label === '根目录').length
          });
        } else {
          // 处理CSV数据：将每行转换为样本
          const loadedSamples = data.preview.map((row, idx) => {
            const keys = Object.keys(row);
            const textKeys = keys.filter(k => 
              k.toLowerCase().includes('text') || 
              k.toLowerCase().includes('content') || 
              k.toLowerCase().includes('sentence') ||
              k.toLowerCase().includes('描述') ||
              k.toLowerCase().includes('内容')
            );
            const labelKeys = keys.filter(k => 
              k.toLowerCase() === 'label' || 
              k.toLowerCase() === 'target' || 
              k.toLowerCase() === 'class' ||
              k.toLowerCase() === 'category' ||
              k.toLowerCase() === '标签' ||
              k.toLowerCase() === '类别'
            );
            
            let text = '';
            if (textKeys.length > 0) {
              text = row[textKeys[0]];
            } else {
              const nonLabelKeys = keys.filter(k => !labelKeys.includes(k));
              text = nonLabelKeys.length > 0 ? String(row[nonLabelKeys[0]]) : JSON.stringify(row);
            }
            
            return {
              id: idx,
              text: text,
              label: labelKeys.length > 0 ? row[labelKeys[0]] : null,
              raw: row
            };
          });
          
          setSamples(loadedSamples);
          
          // 记录初始标签状态（用于计算本次会话进度）
          const initialLabels = {};
          loadedSamples.forEach((s, idx) => {
            initialLabels[idx] = s.label;
          });
          initialLabelsRef.current = initialLabels;
          
          // 重置本次会话统计
          setSessionStats({ annotated: 0, modified: 0 });
          
          setStats({
            total: loadedSamples.length,
            labeled: loadedSamples.filter(s => s.label).length,
            unlabeled: loadedSamples.filter(s => !s.label).length
          });
          
          const existingLabels = [...new Set(loadedSamples.filter(s => s.label).map(s => s.label))];
          if (existingLabels.length > 0) {
            setLabels(prev => [...new Set([...prev, ...existingLabels])]);
          }
        }
      } else {
        setSamples([]);
        setStats({ total: 0, labeled: 0, unlabeled: 0 });
        initialLabelsRef.current = {};
        setSessionStats({ annotated: 0, modified: 0 });
      }
      
      if (dataset.labels && Array.isArray(dataset.labels)) {
        setLabels(prev => [...new Set([...prev, ...dataset.labels])]);
      }
    } catch (e) {
      console.error('加载样本失败:', e);
      alert('加载样本失败: ' + e.message);
    }
    setLoading(false);
  };

  // 标注当前样本
  const annotateSample = (label) => {
    const updated = [...samples];
    const oldLabel = updated[currentIndex].label;
    updated[currentIndex].label = label;
    setSamples(updated);
    
    // 更新本次会话统计
    const initialLabel = initialLabelsRef.current[currentIndex];
    setSessionStats(prev => {
      let { annotated, modified } = prev;
      
      // 如果是新标注（之前没有标签，现在有标签）
      if (!initialLabel && label) {
        annotated += 1;
      }
      // 如果是修改标注（之前有标签，现在标签不同）
      else if (initialLabel && label && initialLabel !== label) {
        modified += 1;
      }
      // 如果是清除标注（之前有标签，现在无标签）
      else if (initialLabel && !label) {
        annotated = Math.max(0, annotated - 1);
      }
      
      return { annotated, modified };
    });
    
    setStats({
      total: samples.length,
      labeled: updated.filter(s => s.label).length,
      unlabeled: updated.filter(s => !s.label).length
    });
    // 自动下一条
    if (currentIndex < samples.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  // 添加新标签
  const addLabel = () => {
    if (newLabel && !labels.includes(newLabel)) {
      setLabels([...labels, newLabel]);
      setNewLabel('');
    }
  };

  // 删除标签
  const removeLabel = (labelToRemove) => {
    setLabels(labels.filter(l => l !== labelToRemove));
    // 清除已标注的该标签
    const updated = samples.map(s => s.label === labelToRemove ? { ...s, label: null } : s);
    setSamples(updated);
  };

  // 导出标注结果
  const exportAnnotations = () => {
    const labeledData = samples.filter(s => s.label);
    const csv = labeledData.map(s => {
      const text = projectType === 'text_classification' ? `"${(s.text || s.content || '').replace(/"/g, '""')}"` : s.file_path;
      return `${text},${s.label}`;
    }).join('\n');
    const header = projectType === 'text_classification' ? 'text,label\n' : 'file_path,label\n';
    const blob = new Blob([header + csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `annotations_${projectId}_${Date.now()}.csv`;
    a.click();
  };

  // ============ 画框标注相关函数 ============
  
  // 获取当前图片的所有标注框
  const getCurrentAnnotations = () => {
    return imageAnnotations[currentIndex] || [];
  };

  // 添加标注框
  const addBox = (box) => {
    const newBox = { ...box, id: Date.now(), label: activeLabel || labels[0] || '未命名' };
    setImageAnnotations(prev => ({
      ...prev,
      [currentIndex]: [...(prev[currentIndex] || []), newBox]
    }));
    updateStats();
  };

  // 删除标注框
  const deleteBox = (boxId) => {
    setImageAnnotations(prev => ({
      ...prev,
      [currentIndex]: (prev[currentIndex] || []).filter(b => b.id !== boxId)
    }));
    setSelectedBox(null);
    updateStats();
  };

  // 更新标注框标签
  const updateBoxLabel = (boxId, label) => {
    setImageAnnotations(prev => ({
      ...prev,
      [currentIndex]: (prev[currentIndex] || []).map(b => 
        b.id === boxId ? { ...b, label } : b
      )
    }));
  };

  // 更新统计
  const updateStats = () => {
    const totalBoxes = Object.values(imageAnnotations).flat().length;
    const annotatedImages = Object.keys(imageAnnotations).filter(k => imageAnnotations[k].length > 0).length;
    setStats({
      total: samples.length,
      labeled: annotatedImages,
      unlabeled: samples.length - annotatedImages,
      totalBoxes
    });
  };

  // 图片加载完成
  const handleImageLoad = (e) => {
    const img = e.target;
    setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    setImageLoaded(true);
    drawCanvas();
  };

  // 绘制Canvas
  const drawCanvas = () => {
    if (!canvasRef.current || !imageRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imageRef.current;
    
    // 设置canvas尺寸与图片一致
    canvas.width = img.offsetWidth;
    canvas.height = img.offsetHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 绘制所有标注框
    const annotations = getCurrentAnnotations();
    const scaleX = canvas.width / img.naturalWidth;
    const scaleY = canvas.height / img.naturalHeight;
    
    annotations.forEach((box, idx) => {
      const x = box.x * scaleX;
      const y = box.y * scaleY;
      const w = box.width * scaleX;
      const h = box.height * scaleY;
      
      // 高亮选中的框
      const isSelected = selectedBox === box.id;
      ctx.strokeStyle = isSelected ? '#ff0000' : '#00ff00';
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(x, y, w, h);
      
      // 填充半透明背景
      ctx.fillStyle = isSelected ? 'rgba(255, 0, 0, 0.1)' : 'rgba(0, 255, 0, 0.1)';
      ctx.fillRect(x, y, w, h);
      
      // 绘制标签
      ctx.fillStyle = isSelected ? '#ff0000' : '#00ff00';
      ctx.font = 'bold 14px Arial';
      ctx.fillText(`${idx + 1}. ${box.label}`, x + 2, y - 5);
    });
    
    // 绘制正在拖拽的框
    if (isDrawing && drawStart && drawEnd) {
      const x = Math.min(drawStart.x, drawEnd.x);
      const y = Math.min(drawStart.y, drawEnd.y);
      const w = Math.abs(drawEnd.x - drawStart.x);
      const h = Math.abs(drawEnd.y - drawStart.y);
      
      ctx.strokeStyle = '#ff00ff';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }
  };

  // 鼠标事件处理
  const handleMouseDown = (e) => {
    if (!imageLoaded) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // 检查是否点击了已有的框
    const annotations = getCurrentAnnotations();
    const scaleX = canvas.width / imageRef.current.naturalWidth;
    const scaleY = canvas.height / imageRef.current.naturalHeight;
    
    let clickedBox = null;
    for (let i = annotations.length - 1; i >= 0; i--) {
      const box = annotations[i];
      const bx = box.x * scaleX;
      const by = box.y * scaleY;
      const bw = box.width * scaleX;
      const bh = box.height * scaleY;
      
      if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
        clickedBox = box.id;
        break;
      }
    }
    
    if (clickedBox) {
      setSelectedBox(clickedBox);
    } else {
      // 开始画新框
      setIsDrawing(true);
      setDrawStart({ x, y });
      setDrawEnd({ x, y });
      setSelectedBox(null);
    }
  };

  const handleMouseMove = (e) => {
    if (!isDrawing || !drawStart) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    setDrawEnd({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
    drawCanvas();
  };

  const handleMouseUp = () => {
    if (!isDrawing || !drawStart || !drawEnd) {
      setIsDrawing(false);
      return;
    }
    
    const canvas = canvasRef.current;
    const minSize = 10; // 最小框尺寸
    const w = Math.abs(drawEnd.x - drawStart.x);
    const h = Math.abs(drawEnd.y - drawStart.y);
    
    if (w > minSize && h > minSize) {
      // 转换为原始图片坐标
      const scaleX = imageRef.current.naturalWidth / canvas.width;
      const scaleY = imageRef.current.naturalHeight / canvas.height;
      
      const box = {
        x: Math.min(drawStart.x, drawEnd.x) * scaleX,
        y: Math.min(drawStart.y, drawEnd.y) * scaleY,
        width: w * scaleX,
        height: h * scaleY
      };
      
      addBox(box);
    }
    
    setIsDrawing(false);
    setDrawStart(null);
    setDrawEnd(null);
    drawCanvas();
  };

  // 切换图片时重绘
  useEffect(() => {
    if (isImageDataset(activeDataset) && imageLoaded) {
      drawCanvas();
    }
  }, [currentIndex, imageAnnotations, selectedBox]);

  const currentSample = samples[currentIndex];
  const isTextTask = !projectType || projectType === 'text_classification' || projectType === 'nlp';

  return (
    <div className="space-y-6">
      {/* 数据集选择 */}
      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4">选择数据集</h3>
        <div className="flex gap-2 flex-wrap">
          {datasets.map(ds => (
            <button
              key={ds.id}
              onClick={() => { setActiveDataset(ds); loadSamples(ds); setCurrentIndex(0); }}
              className={`px-4 py-2 rounded-xl font-medium transition-all ${
                activeDataset && activeDataset.id === ds.id
                  ? 'bg-purple-100 text-purple-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {ds.name}
            </button>
          ))}
          {datasets.length === 0 && (
            <p className="text-gray-400">请先上传数据集</p>
          )}
        </div>
      </div>

      {activeDataset && (
        <React.Fragment>
          {/* 统计区域 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
              <h4 className="text-sm text-gray-500 mb-2">总标注进度</h4>
              <div className="text-3xl font-bold text-purple-600">{stats.labeled}/{stats.total}</div>
              <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all"
                  style={{width: `${stats.total > 0 ? (stats.labeled / stats.total) * 100 : 0}%`}}
                />
              </div>
              {/* 本次会话进度 */}
              {sessionStats.annotated > 0 || sessionStats.modified > 0 ? (
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <p className="text-xs text-gray-500">本次会话</p>
                  <p className="text-sm font-medium text-green-600">
                    新增标注: {sessionStats.annotated}条
                    {sessionStats.modified > 0 && ` · 修改: ${sessionStats.modified}条`}
                  </p>
                </div>
              ) : (
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <p className="text-xs text-gray-400">本次会话尚未标注</p>
                </div>
              )}
            </div>
            
            {isImageDataset(activeDataset) ? (
              // 图片数据集：显示画框统计
              <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg md:col-span-2">
                <h4 className="text-sm text-gray-500 mb-3">画框标注统计</h4>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{stats.total || samples.length}</div>
                    <div className="text-xs text-gray-500">总图片</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">{stats.labeled || 0}</div>
                    <div className="text-xs text-gray-500">已标注</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">{Object.values(imageAnnotations).flat().length}</div>
                    <div className="text-xs text-gray-500">总框数</div>
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {labels.map(label => (
                    <span key={label} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                      {label}
                    </span>
                  ))}
                  {labels.length === 0 && <span className="text-sm text-gray-400">暂无标签，请添加</span>}
                </div>
              </div>
            ) : (
              // 文本/表格数据集：显示标签管理
              <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg md:col-span-2">
                <h4 className="text-sm text-gray-500 mb-3">标签管理</h4>
                <div className="flex flex-wrap gap-2 mb-3">
                  {labels.map(label => (
                    <span key={label} className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm flex items-center gap-1">
                      {label}
                      <button onClick={() => removeLabel(label)} className="hover:text-red-500">×</button>
                    </span>
                  ))}
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newLabel}
                    onChange={(e) => setNewLabel(e.target.value)}
                    placeholder="添加新标签"
                    className="flex-1 px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    onKeyPress={(e) => e.key === 'Enter' && addLabel()}
                  />
                  <button onClick={addLabel} className="px-4 py-2 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200">
                    添加
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* 标注工作区 */}
          {samples.length > 0 ? (
            <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm text-gray-500">
                  {isImageDataset(activeDataset) ? '图片' : '样本'} {currentIndex + 1} / {samples.length}
                </span>
                {!isImageDataset(activeDataset) && (
                  <div className="flex gap-2">
                    <button onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))} className="px-3 py-1 bg-gray-100 rounded-lg hover:bg-gray-200">上一条</button>
                    <button onClick={() => setCurrentIndex(Math.min(samples.length - 1, currentIndex + 1))} className="px-3 py-1 bg-gray-100 rounded-lg hover:bg-gray-200">下一条</button>
                  </div>
                )}
              </div>

              {/* 样本内容显示 */}
              <div className="bg-gray-50 rounded-xl p-6 mb-6 min-h-[400px] flex items-center justify-center">
                {isImageDataset(activeDataset) || currentSample && currentSample.file_path ? (
                  // 图片类型 - Canvas画框标注
                  <div className="w-full">
                    {currentSample && currentSample.file_path ? (
                      <div className="relative inline-block mx-auto" style={{ maxWidth: '100%' }}>
                        {/* 底层图片 */}
                        <img 
                          ref={imageRef}
                          src={`${API_BASE}/projects/${projectId}/datasets/${activeDataset.id}/image?path=${encodeURIComponent(currentSample.file_path)}`}
                          alt="标注样本"
                          className="max-h-[600px] max-w-full rounded-lg shadow-lg"
                          onLoad={handleImageLoad}
                          onError={(e) => { 
                            console.error('图片加载失败:', e.target.src);
                            setImageLoaded(false);
                          }}
                          style={{ display: 'block' }}
                        />
                        {/* Canvas画框层 */}
                        {imageLoaded && (
                          <canvas
                            ref={canvasRef}
                            className="absolute top-0 left-0 cursor-crosshair"
                            style={{ touchAction: 'none' }}
                            onMouseDown={handleMouseDown}
                            onMouseMove={handleMouseMove}
                            onMouseUp={handleMouseUp}
                            onMouseLeave={handleMouseUp}
                          />
                        )}
                      </div>
                    ) : (
                      <div className="text-gray-400 py-12 text-center">
                        <Icons.Image className="w-16 h-16 mx-auto mb-2" />
                        <p>图片路径未找到</p>
                      </div>
                    )}
                    <p className="mt-2 text-sm text-gray-500 text-center">{currentSample.file_path}</p>
                  </div>
                ) : currentSample && currentSample.raw && Object.keys(currentSample.raw).length > 1 ? (
                  // 表格数据类型 - 显示为表格
                  <div className="w-full max-w-4xl">
                    <h5 className="text-sm font-medium text-gray-500 mb-3">样本数据</h5>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <tbody>
                          {Object.entries(currentSample.raw).map(([key, value]) => (
                            <tr key={key} className="border-b border-gray-200 last:border-0">
                              <td className="py-2 px-3 font-medium text-gray-600 w-32 whitespace-nowrap">{key}</td>
                              <td className="py-2 px-3 text-gray-800 break-words">{String(value)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <h5 className="text-sm font-medium text-gray-500 mb-2">主要内容</h5>
                      <p className="text-gray-800 text-lg leading-relaxed break-words">{currentSample && currentSample.text || '无文本内容'}</p>
                    </div>
                  </div>
                ) : (
                  // 纯文本类型
                  <div className="w-full max-w-4xl">
                    <p className="text-gray-800 text-lg leading-relaxed break-words">{currentSample && currentSample.text || currentSample && currentSample.content || '无文本内容'}</p>
                  </div>
                )}
              </div>

              {/* 标注操作区域 */}
              {isImageDataset(activeDataset) ? (
                // 图片数据集：画框标注控制面板
                <div className="space-y-4">
                  {/* 当前标注框列表 */}
                  {getCurrentAnnotations().length > 0 && (
                    <div className="bg-gray-50 rounded-xl p-4">
                      <h4 className="text-sm font-medium text-gray-600 mb-3">当前标注 ({getCurrentAnnotations().length}个框)</h4>
                      <div className="flex flex-wrap gap-2">
                        {getCurrentAnnotations().map((box, idx) => (
                          <div 
                            key={box.id}
                            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm ${
                              selectedBox === box.id 
                                ? 'bg-red-100 text-red-700 ring-2 ring-red-500' 
                                : 'bg-white border border-gray-200'
                            }`}
                          >
                            <span className="font-medium">{idx + 1}.</span>
                            <select
                              value={box.label}
                              onChange={(e) => updateBoxLabel(box.id, e.target.value)}
                              className="bg-transparent border-none text-sm focus:outline-none cursor-pointer"
                            >
                              {labels.map(l => (
                                <option key={l} value={l}>{l}</option>
                              ))}
                              {labels.length === 0 && <option value={box.label}>{box.label}</option>}
                            </select>
                            <button 
                              onClick={() => deleteBox(box.id)}
                              className="text-red-500 hover:text-red-700 ml-1"
                              title="删除"
                            >
                              ×
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* 当前选择标签 */}
                  <div className="flex items-center justify-center gap-4">
                    <span className="text-sm text-gray-500">新建框标签:</span>
                    <select
                      value={activeLabel}
                      onChange={(e) => setActiveLabel(e.target.value)}
                      className="px-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      {labels.length > 0 ? labels.map(l => (
                        <option key={l} value={l}>{l}</option>
                      )) : (
                        <option value="">请先添加标签</option>
                      )}
                    </select>
                    <input
                      type="text"
                      value={newLabel}
                      onChange={(e) => setNewLabel(e.target.value)}
                      placeholder="新标签"
                      className="px-3 py-2 border border-gray-200 rounded-lg w-24"
                      onKeyPress={(e) => e.key === 'Enter' && (addLabel(), setActiveLabel(newLabel))}
                    />
                    <button 
                      onClick={() => { addLabel(); if (newLabel) setActiveLabel(newLabel); }}
                      className="px-3 py-2 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200"
                    >
                      +
                    </button>
                  </div>

                  {/* 图片导航 */}
                  <div className="flex justify-center gap-4 pt-4 border-t border-gray-200">
                    <button 
                      onClick={() => { setCurrentIndex(Math.max(0, currentIndex - 1)); setSelectedBox(null); }}
                      disabled={currentIndex === 0}
                      className="px-6 py-3 bg-gray-100 text-gray-700 rounded-xl font-medium hover:bg-gray-200 disabled:opacity-50"
                    >
                      ← 上一张
                    </button>
                    <span className="px-6 py-3 text-gray-500">
                      {currentIndex + 1} / {samples.length}
                    </span>
                    <button 
                      onClick={() => { setCurrentIndex(Math.min(samples.length - 1, currentIndex + 1)); setSelectedBox(null); }}
                      disabled={currentIndex >= samples.length - 1}
                      className="px-6 py-3 bg-purple-100 text-purple-700 rounded-xl font-medium hover:bg-purple-200 disabled:opacity-50"
                    >
                      下一张 →
                    </button>
                  </div>

                  {/* 操作提示 */}
                  <p className="text-sm text-gray-500 text-center">
                    💡 在图片上拖拽鼠标画框，点击框可选中，×删除
                  </p>
                </div>
              ) : (
                // 文本/表格数据集：显示分类标注按钮
                <React.Fragment>
                  <div className="flex flex-wrap gap-3 justify-center">
                    {labels.map(label => {
                      // 标签中英文映射
                      const labelMap = {
                        'positive': '正面',
                        'negative': '负面', 
                        'neutral': '中性',
                        'pos': '正面',
                        'neg': '负面',
                        'neu': '中性',
                        'good': '好评',
                        'bad': '差评',
                        'normal': '正常',
                        'fault': '故障',
                        'warning': '警告',
                        'success': '成功',
                        'fail': '失败',
                        'true': '是',
                        'false': '否',
                        'yes': '是',
                        'no': '否',
                        '1': '正例',
                        '0': '负例'
                      };
                      const chineseLabel = labelMap[label.toLowerCase()] || label;
                      const displayText = chineseLabel !== label ? `${chineseLabel} (${label})` : label;
                      
                      return (
                        <button
                          key={label}
                          onClick={() => annotateSample(label)}
                          className={`px-6 py-3 rounded-xl font-medium transition-all ${
                            currentSample && currentSample.label === label
                              ? 'bg-green-100 text-green-700 ring-2 ring-green-500'
                              : 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                          }`}
                        >
                          {displayText} {currentSample && currentSample.label === label && '✓'}
                        </button>
                      );
                    })}
                    <button
                      onClick={() => annotateSample(null)}
                      className="px-6 py-3 rounded-xl font-medium bg-gray-100 text-gray-600 hover:bg-gray-200"
                    >
                      清除标注
                    </button>
                  </div>

                  {/* 批量操作 */}
                  <div className="mt-6 pt-6 border-t border-gray-200 flex justify-between">
                    <button onClick={exportAnnotations} className="px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 flex items-center gap-2">
                      <Icons.Download /> 导出标注结果
                    </button>
                    <p className="text-sm text-gray-500">
                      提示: 点击标签按钮即可标注并自动跳转到下一条
                    </p>
                  </div>
                </React.Fragment>
              )}
            </div>
          ) : loading ? (
            <div className="text-center py-12">
              <Icons.Loading className="w-8 h-8 animate-spin mx-auto text-purple-500" />
              <p className="mt-2 text-gray-500">加载样本中...</p>
            </div>
          ) : (
            <div className="text-center py-12 text-gray-400">
              <Icons.Database className="w-12 h-12 mx-auto mb-4" />
              <p>该数据集暂无样本数据</p>
            </div>
          )}
        </React.Fragment>
      )}
    </div>
  );
}

// 数据管理Tab
function DataTab({ projectId, datasets, onRefresh }) {
  const [uploading, setUploading] = useState(false);
  const [previewDataset, setPreviewDataset] = useState(null);
  const [previewData, setPreviewData] = useState(null);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', file.name);

    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/datasets`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        onRefresh();
      }
    } catch (e) {
      alert('上传失败');
    }
    setUploading(false);
  };

  const viewPreview = async (dataset) => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/${dataset.id}/preview?limit=50`);
      const data = await res.json();
      if (!res.ok) {
        alert(data.detail || '预览失败');
        return;
      }
      if (!data.columns || !data.preview) {
        alert('数据格式错误');
        return;
      }
      setPreviewData(data);
      setPreviewDataset(dataset);
    } catch (e) {
      alert('预览失败: ' + e.message);
    }
  };

  const downloadDataset = (datasetId) => {
    window.open(`${API_BASE}/projects/${projectId}/datasets/${datasetId}/download`, '_blank');
  };

  return (
    <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-semibold text-gray-800">数据集管理</h3>
          <p className="text-sm text-gray-500">上传CSV格式的文本数据，最后一列为标签</p>
        </div>
        <label className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-purple-600 text-white rounded-xl font-medium hover:shadow-lg transition-all cursor-pointer flex items-center gap-2">
          <Icons.Upload /> {uploading ? '上传中...' : '上传数据'}
          <input type="file" accept=".csv,.txt" className="hidden" onChange={handleUpload} />
        </label>
      </div>

      {/* 数据预览弹窗 */}
      {previewDataset && previewData && previewData.columns && previewData.preview && (
        <div className="fixed inset-0 bg-black/50 flex items-start justify-center z-50 p-4 pt-12">
          <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[80vh] flex flex-col shadow-2xl">
            <div className="p-4 border-b flex items-center justify-between bg-gray-50 rounded-t-2xl">
              <div>
                <h4 className="font-semibold">数据预览: {previewDataset.name}</h4>
                <p className="text-sm text-gray-500">
                  显示 {previewData.preview_rows || previewData.preview && preview.length || 0} / {previewData.total_rows || previewData.preview && preview.length || 0} 条数据
                  {(previewData.total_rows || 0) > 100 && " (最多显示100条)"}
                  {previewData.type === 'image_folder' && " · 图片数据集"}
                </p>
              </div>
              <button onClick={() => { setPreviewDataset(null); setPreviewData(null); }} className="text-gray-400 hover:text-gray-600 p-2 hover:bg-gray-200 rounded-lg transition-colors">
                <Icons.Close />
              </button>
            </div>
            <div className="flex-1 overflow-auto p-4">
              {previewData.type === 'image_folder' ? (
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {previewData.preview.map((row, i) => (
                    <div key={i} className="group relative">
                      <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden border border-gray-200">
                        <img 
                          src={`/ai-training/api/projects/${projectId}/datasets/${previewDataset.id}/image?path=${encodeURIComponent(row['图片路径'])}`}
                          alt={row['图片路径']}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                          onError={(e) => { e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🖼️</text></svg>'; }}
                        />
                      </div>
                      <p className="text-xs text-gray-500 mt-1 truncate" title={row['类别文件夹']}>
                        {row['类别文件夹']}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      {previewData.columns.map(col => (
                        <th key={col} className="px-3 py-2 text-left font-medium text-gray-700">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewData.preview.map((row, i) => (
                      <tr key={i} className="border-b">
                        {previewData.columns.map(col => (
                          <td key={col} className="px-3 py-2 text-gray-600 truncate max-w-xs">{row[col] !== undefined ? String(row[col]) : '-'}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="space-y-3">
        {datasets && datasets.map(ds => (
          <div key={ds.id} className="p-4 bg-gray-50/80 rounded-xl border border-gray-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-cyan-100 rounded-lg flex items-center justify-center">
                  <Icons.Database />
                </div>
                <div>
                  <h4 className="font-medium text-gray-800">{ds.name}</h4>
                  <p className="text-sm text-gray-500">
                    {ds.total_samples} 条数据 · {ds.labels && ds.labels.length || 0} 个类别
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => viewPreview(ds)}
                  className="px-3 py-1.5 bg-blue-100 text-blue-700 rounded-lg text-sm hover:bg-blue-200 flex items-center gap-1"
                >
                  <Icons.Eye /> 浏览
                </button>
                <button
                  onClick={() => downloadDataset(ds.id)}
                  className="px-3 py-1.5 bg-green-100 text-green-700 rounded-lg text-sm hover:bg-green-200 flex items-center gap-1"
                >
                  <Icons.Download /> 下载
                </button>
                <span className={`px-2 py-1 rounded-full text-xs ${
                  ds.status === 'preprocessed' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
                }`}>
                  {ds.status === 'preprocessed' ? '已处理' : '待处理'}
                </span>
              </div>
            </div>
            {ds.labels && ds.labels.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {ds.labels.map(label => (
                  <span key={label} className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs">
                    {label}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {(!datasets || datasets.length === 0) && (
        <div className="text-center py-8 border-2 border-dashed border-gray-200 rounded-xl">
          <div className="text-4xl mb-2"><Icons.Upload /></div>
          <p className="text-gray-500">暂无数据集，请上传CSV文件</p>
          <p className="text-sm text-gray-400 mt-1">格式：文本列, 标签列</p>
        </div>
      )}
    </div>
  );
}

// 标注Tab - 带Demo说明
function AnnotateTab({ projectId }) {
  const [annotations, setAnnotations] = useState([]);
  const [newContent, setNewContent] = useState('');
  const [newLabel, setNewLabel] = useState('');
  const [showDemo, setShowDemo] = useState(true);

  useEffect(() => {
    loadAnnotations();
  }, [projectId]);

  const loadAnnotations = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/annotations?status=pending`);
      const data = await res.json();
      setAnnotations(data.annotations || []);
    } catch (e) {}
  };

  const addAnnotation = async () => {
    if (!newContent.trim()) return;
    try {
      await fetch(`${API_BASE}/projects/${projectId}/annotations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: newContent, label: newLabel || null })
      });
      setNewContent('');
      setNewLabel('');
      loadAnnotations();
    } catch (e) {}
  };

  const updateLabel = async (id, label) => {
    try {
      await fetch(`${API_BASE}/projects/${projectId}/annotations/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label })
      });
      loadAnnotations();
    } catch (e) {}
  };

  // Demo标注示例
  const demoExamples = [
    { content: "这家餐厅的服务态度真的很好，菜品也很新鲜", label: "正面", explain: "表达了积极的用餐体验" },
    { content: "等了一个小时还没上菜，太糟糕了", label: "负面", explain: "表达了不满和抱怨" },
    { content: "价格适中，味道还可以，就是环境有点吵", label: "中性", explain: "有优点也有缺点，整体中立" },
  ];

  return (
    <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
      <h3 className="font-semibold text-gray-800 mb-2">数据标注</h3>
      
      {/* Demo说明 */}
      {showDemo && (
        <div className="mb-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border border-blue-200">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-medium text-blue-800 flex items-center gap-2">
              <Icons.Help /> 标注示例（点击学习）
            </h4>
            <button onClick={() => setShowDemo(false)} className="text-blue-600 text-sm">隐藏</button>
          </div>
          <div className="space-y-2">
            {demoExamples.map((demo, i) => (
              <div key={i} className="p-3 bg-white rounded-lg border border-blue-100">
                <div className="text-gray-700 mb-1">"{demo.content}"</div>
                <div className="flex items-center gap-2 text-sm">
                  <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded">{demo.label}</span>
                  <span className="text-gray-500">→ {demo.explain}</span>
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            💡 标注是为文本打上类别标签，帮助模型学习分类规则
          </p>
        </div>
      )}
      
      <div className="mb-4 p-4 bg-gray-50 rounded-xl">
        <textarea
          value={newContent}
          onChange={(e) => setNewContent(e.target.value)}
          placeholder="输入待标注的文本..."
          className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none mb-2"
          rows={3}
        />
        <div className="flex gap-2">
          <input
            type="text"
            value={newLabel}
            onChange={(e) => setNewLabel(e.target.value)}
            placeholder="标签（如：正面/负面/中性）"
            className="px-3 py-1.5 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none text-sm flex-1"
          />
          <button
            onClick={addAnnotation}
            className="px-4 py-1.5 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700"
          >
            添加
          </button>
        </div>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {annotations.map(anno => (
          <div key={anno.id} className="p-3 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-700 mb-2">{anno.content}</p>
            <div className="flex gap-2">
              <input
                type="text"
                defaultValue={anno.label || ''}
                placeholder="输入标签"
                className="flex-1 px-2 py-1 rounded border border-gray-200 text-sm"
                onBlur={(e) => updateLabel(anno.id, e.target.value)}
              />
              <button 
                onClick={() => updateLabel(anno.id, anno.label)}
                className="px-3 py-1 bg-green-500 text-white rounded text-sm"
              >
                <Icons.Check />
              </button>
            </div>
          </div>
        ))}
      </div>
      
      {annotations.length === 0 && (
        <div className="text-center py-8 text-gray-400">
          <Icons.Tag />
          <p className="mt-2">暂无待标注数据</p>
          <p className="text-sm">在上方添加文本进行标注</p>
        </div>
      )}
    </div>
  );
}

// ML任务配置模板
const ML_TEMPLATES = {
  classification: {
    fast: { model_type: 'logistic', n_estimators: 50 },
    balanced: { model_type: 'random_forest', n_estimators: 100 },
    accurate: { model_type: 'random_forest', n_estimators: 200, max_depth: 20 }
  },
  regression: {
    fast: { model_type: 'linear' },
    balanced: { model_type: 'random_forest', n_estimators: 100 },
    accurate: { model_type: 'random_forest', n_estimators: 200 }
  },
  anomaly_detection: {
    fast: { model_type: 'isolation_forest', contamination: 0.1 },
    balanced: { model_type: 'isolation_forest', contamination: 0.05 },
    accurate: { model_type: 'lof', contamination: 0.05 }
  },
  time_series: {
    fast: { model_type: 'random_forest', n_estimators: 50, task_type: 'time_series' },
    balanced: { model_type: 'random_forest', n_estimators: 100, task_type: 'time_series' },
    accurate: { model_type: 'gradient_boosting', n_estimators: 100, task_type: 'time_series' }
  },
  image_classification: {
    fast: { model_name: 'mobilenet_v2', epochs: 5, batch_size: 32, freeze_backbone: true, image_size: 224 },
    balanced: { model_name: 'resnet50', epochs: 10, batch_size: 16, freeze_backbone: true, image_size: 224 },
    accurate: { model_name: 'efficientnet_b0', epochs: 15, batch_size: 16, freeze_backbone: false, image_size: 256 }
  },
  object_detection: {
    fast: { model: 'yolov8n', epochs: 50, batch_size: 16, image_size: 640, confidence_threshold: 0.25 },
    balanced: { model: 'yolov8s', epochs: 100, batch_size: 16, image_size: 640, confidence_threshold: 0.25 },
    accurate: { model: 'yolov8m', epochs: 150, batch_size: 8, image_size: 640, confidence_threshold: 0.25 }
  }
};

// 图片分类模型说明
const IMAGE_MODEL_HELP = {
  mobilenet_v2: { name: 'MobileNet V2', desc: '轻量级模型，适合边缘设备', size: '14MB', speed: '快' },
  resnet18: { name: 'ResNet-18', desc: '小型ResNet，平衡速度与精度', size: '45MB', speed: '较快' },
  resnet50: { name: 'ResNet-50', desc: '经典模型，精度较高', size: '98MB', speed: '中等' },
  efficientnet_b0: { name: 'EfficientNet-B0', desc: '高效模型，精度速度平衡', size: '21MB', speed: '中等' },
  vit_base: { name: 'Vision Transformer', desc: '视觉Transformer，精度最高', size: '346MB', speed: '较慢' }
};

// YOLO模型说明
const YOLO_MODEL_HELP = {
  yolov8n: { name: 'YOLOv8 Nano', desc: '超轻量，适合边缘设备', size: '6MB', speed: '最快', ap: '37.3' },
  yolov8s: { name: 'YOLOv8 Small', desc: '小型，平衡性能', size: '22MB', speed: '快', ap: '44.9' },
  yolov8m: { name: 'YOLOv8 Medium', desc: '中等，精度较高', size: '54MB', speed: '中等', ap: '50.2' },
  yolov8l: { name: 'YOLOv8 Large', desc: '大型，精度最高', size: '89MB', speed: '慢', ap: '52.9' }
};

// 训练Tab - 支持多种任务类型
function TrainTab({ projectId, projectType, datasets, jobs, onRefresh }) {
  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState('balanced');
  const [config, setConfig] = useState({});
  const [starting, setStarting] = useState(false);
  const [showHelp, setShowHelp] = useState(true);
  const [detailJob, setDetailJob] = useState(null);
  const [columns, setColumns] = useState([]);
  const [datasetPreview, setDatasetPreview] = useState(null);
  
  // AutoML 相关状态
  const [showAutoML, setShowAutoML] = useState(false);
  const [autoMLExperiments, setAutoMLExperiments] = useState([]);
  const [autoMLForm, setAutoMLForm] = useState({ name: '', max_trials: 10 });
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [recommendedParams, setRecommendedParams] = useState(null);

  // 判断任务类型
  const isNLP = projectType === 'text_classification';
  const isTimeSeries = projectType === 'time_series';
  const isImage = projectType === 'image_classification';
  const isDetection = projectType === 'object_detection';

  // 初始化配置
  useEffect(() => {
    if (isNLP) {
      setConfig(TRAINING_TEMPLATES.balanced.config);
    } else if (isTimeSeries) {
      // 时序分析配置
      setConfig({
        task_type: 'time_series',
        model_type: 'random_forest',
        target_column: '',
        feature_columns: [],
        time_column: 'timestamp',
        n_estimators: 100,
        forecast_horizon: 24
      });
    } else if (isImage) {
      // 图片分类配置
      setConfig({
        model_name: 'resnet50',
        epochs: 10,
        batch_size: 16,
        learning_rate: 1e-4,
        image_size: 224,
        freeze_backbone: true,
        val_split: 0.2,
        quantize: false
      });
    } else if (isDetection) {
      // 目标检测配置
      setConfig({
        model: 'yolov8s',
        epochs: 100,
        batch_size: 16,
        image_size: 640,
        confidence_threshold: 0.25,
        iou_threshold: 0.45
      });
    } else {
      // ML任务默认配置
      setConfig({
        task_type: projectType,
        model_type: 'random_forest',
        target_column: '',
        feature_columns: [],
        n_estimators: 100,
        contamination: 0.1
      });
    }
  }, [projectType, isNLP, isTimeSeries, isImage, isDetection]);

  // 加载 AutoML 实验列表
  const loadAutoMLExperiments = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/automl/experiments`);
      const data = await res.json();
      setAutoMLExperiments(data.experiments || []);
    } catch (e) {
      console.error('加载AutoML实验失败:', e);
    }
  };

  // 创建 AutoML 实验
  const createAutoMLExperiment = async () => {
    if (!autoMLForm.name || !selectedDataset) {
      alert('请填写实验名称并选择数据集');
      return;
    }
    try {
      const form = new FormData();
      form.append('dataset_id', selectedDataset);
      form.append('name', autoMLForm.name);
      form.append('max_trials', autoMLForm.max_trials);
      const res = await fetch(`${API_BASE}/projects/${projectId}/automl/experiments`, { method: 'POST', body: form });
      if (!res.ok) {
        const data = await res.json();
        alert('创建实验失败: ' + (data.detail || data.error));
        return;
      }
      setAutoMLForm({ name: '', max_trials: 10 });
      loadAutoMLExperiments();
    } catch (e) {
      alert('创建实验失败: ' + e.message);
    }
  };

  // 运行 AutoML Trial
  const runAutoMLTrial = async (expId) => {
    try {
      const res = await fetch(`${API_BASE}/automl/experiments/${expId}/run`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        alert('运行实验失败: ' + (data.detail || data.error));
        return;
      }
      if (data.success) {
        alert(`第${data.trial_number}组参数已生成！`);
        loadAutoMLExperiments();
      }
    } catch (e) {
      alert('运行实验失败: ' + e.message);
    }
  };

  // 获取推荐参数
  const getRecommendedParams = async (expId) => {
    try {
      const res = await fetch(`${API_BASE}/automl/experiments/${expId}/recommendation`);
      const data = await res.json();
      if (data.recommended) {
        setRecommendedParams(data);
        setSelectedExperiment(expId);
      }
    } catch (e) {
      console.error('获取推荐参数失败:', e);
    }
  };

  // 应用推荐参数到训练配置
  const applyRecommendedParams = () => {
    if (!recommendedParams || !recommendedParams.recommended) return;
    
    const params = recommendedParams.recommended;
    setConfig(prev => ({
      ...prev,
      ...params
    }));
    
    alert('最优参数已应用到训练配置！');
    setShowAutoML(false);
  };

  // 加载数据集列信息
  const loadDatasetColumns = async (datasetId) => {
    if (!datasetId) return;
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/${datasetId}/preview?limit=5`);
      const data = await res.json();
      setColumns(data.columns || []);
      setDatasetPreview(data);
      
      // 自动设置目标列（最后一列）
      if (data.columns && data.columns.length > 0 && !isNLP) {
        const lastCol = data.columns[data.columns.length - 1];
        const featureCols = data.columns.slice(0, -1);
        setConfig(prev => ({
          ...prev,
          target_column: lastCol,
          feature_columns: featureCols
        }));
      }
    } catch (e) {}
  };

  // 应用模板
  const applyTemplate = (templateKey) => {
    setSelectedTemplate(templateKey);
    if (isNLP && TRAINING_TEMPLATES[templateKey]) {
      setConfig(TRAINING_TEMPLATES[templateKey].config);
    } else if (isTimeSeries && ML_TEMPLATES.time_series && ML_TEMPLATES.time_series[templateKey]) {
      setConfig({ ...config, ...ML_TEMPLATES.time_series[templateKey] });
    } else if (isImage && ML_TEMPLATES.image_classification && ML_TEMPLATES.image_classification[templateKey]) {
      setConfig({ ...config, ...ML_TEMPLATES.image_classification[templateKey] });
    } else if (isDetection && ML_TEMPLATES.object_detection && ML_TEMPLATES.object_detection[templateKey]) {
      setConfig({ ...config, ...ML_TEMPLATES.object_detection[templateKey] });
    } else if (!isNLP && !isTimeSeries && !isImage && !isDetection && ML_TEMPLATES[projectType] && ML_TEMPLATES[projectType][templateKey]) {
      setConfig({ ...config, ...ML_TEMPLATES[projectType][templateKey] });
    }
  };

  const startTraining = async () => {
    if (!selectedDataset) {
      alert('请先选择数据集');
      return;
    }
    if (!isNLP && (!config.target_column || config.feature_columns.length === 0)) {
      alert('请配置目标列和特征列');
      return;
    }
    
    setStarting(true);
    try {
      const formData = new FormData();
      formData.append('dataset_id', selectedDataset);
      formData.append('config', JSON.stringify(config));
      formData.append('template', selectedTemplate);
      
      const res = await fetch(`${API_BASE}/projects/${projectId}/train`, {
        method: 'POST',
        body: formData
      });
      
      if (res.ok) {
        onRefresh();
      } else {
        alert('启动失败');
      }
    } catch (e) {
      alert('启动失败: ' + e.message);
    }
    setStarting(false);
  };

  // 重试失败的任务
  const retryJob = async (jobId) => {
    if (!confirm('确定要重试这个失败的任务吗？')) return;
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/jobs/${jobId}/retry`, {
        method: 'POST'
      });
      if (res.ok) {
        alert('重试成功，任务已重新加入队列');
        onRefresh();
      } else {
        const data = await res.json();
        alert('重试失败: ' + (data.detail || '未知错误'));
      }
    } catch (e) {
      alert('重试失败: ' + e.message);
    }
  };

  // 删除任务
  const deleteJob = async (jobId) => {
    if (!confirm('确定要删除这个训练任务吗？此操作不可恢复。')) return;
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/jobs/${jobId}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        alert('删除成功');
        onRefresh();
      } else {
        const data = await res.json();
        alert('删除失败: ' + (data.detail || '未知错误'));
      }
    } catch (e) {
      alert('删除失败: ' + e.message);
    }
  };

  const selectedModel = isNLP ? MODEL_HELP[config.model_name] : null;

  return (
    <div className="space-y-4">
      {/* 训练指南 */}
      {showHelp && (
        <div className="bg-gradient-to-r from-cyan-50 to-blue-50 rounded-2xl p-5 border border-cyan-200">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-cyan-800 flex items-center gap-2">
              <Icons.Help /> {isTimeSeries ? '时序分析指南' : '训练配置指南'}
            </h4>
            <button onClick={() => setShowHelp(false)} className="text-cyan-600 text-sm">隐藏</button>
          </div>
          {isTimeSeries ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">📊 数据格式</div>
                <p className="text-gray-600">CSV文件，必须包含时间列（如timestamp）和数值列（温度、压力等传感器数据）。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">📈 趋势分析</div>
                <p className="text-gray-600">系统会自动检测数据趋势，预测未来走势，识别异常模式。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">⏱️ RUL预测</div>
                <p className="text-gray-600">基于劣化趋势预测设备剩余使用寿命，提前规划维护。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">🚨 异常检测</div>
                <p className="text-gray-600">自动识别渐变劣化、突变点、波动性异常等模式。</p>
              </div>
            </div>
          ) : isImage ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">🖼️ 数据格式</div>
                <p className="text-gray-600">上传图片压缩包，按文件夹组织（如defect/001.jpg, normal/002.jpg）。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">🤖 模型选择</div>
                <p className="text-gray-600">MobileNet适合边缘部署，ResNet平衡性能，EfficientNet精度最高。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">⚡ 迁移学习</div>
                <p className="text-gray-600">默认冻结骨干网络，只训练分类头。数据量少时建议保持开启。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">📦 模型导出</div>
                <p className="text-gray-600">支持ONNX格式导出，可选INT8量化压缩，适配边缘设备部署。</p>
              </div>
            </div>
          ) : isDetection ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">🎯 数据格式</div>
                <p className="text-gray-600">上传图片文件夹，每个类别一个子文件夹。系统会自动转换为YOLO格式。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">📍 定位能力</div>
                <p className="text-gray-600">不仅能识别缺陷类型，还能框出缺陷位置，支持一张图多个缺陷。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">⚡ YOLO模型</div>
                <p className="text-gray-600">Nano最快适合实时检测，Large最准适合离线分析，Small/Medium平衡。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">📊 评估指标</div>
                <p className="text-gray-600">mAP@0.5-0.95是主要指标，表示检测精度和定位准确度的综合。</p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">📊 数据准备</div>
                <p className="text-gray-600">上传CSV文件，第一列为文本，最后一列为标签。建议每类至少10条数据。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">🤖 模型选择</div>
                <p className="text-gray-600">BERT准确率高但慢，DistilBERT快但略低。小白建议先用DistilBERT试错。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">⚡ 学习率</div>
                <p className="text-gray-600">控制模型学习速度。2e-5是常用值，太大模型学不好，太小学得慢。</p>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="font-medium text-gray-800 mb-1">🔄 训练轮数</div>
                <p className="text-gray-600">数据集遍历次数。3-5轮通常够用，太多会过拟合（只记住训练数据）。</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 训练配置模板选择 */}
      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Icons.Settings /> 快速配置
        </h3>
        
        <div className="grid grid-cols-3 gap-3 mb-6">
          {Object.entries(TRAINING_TEMPLATES).map(([key, template]) => (
            <button
              key={key}
              onClick={() => applyTemplate(key)}
              className={`p-4 rounded-xl border-2 text-left transition-all ${
                selectedTemplate === key
                  ? 'border-purple-500 bg-purple-50'
                  : 'border-gray-200 hover:border-purple-300'
              }`}
            >
              <div className="font-medium text-gray-800 mb-1">{template.name}</div>
              <div className="text-xs text-gray-500">{template.desc}</div>
              <div className="text-xs text-purple-600 mt-2">
                {template.config.model_name === 'distilbert-base-chinese' ? '⚡ 快速' : 
                 template.config.model_name === 'bert-base-chinese' ? '⚖️ 平衡' : '🎯 精确'}
                · {template.config.epochs}轮
              </div>
            </button>
          ))}
        </div>

        {/* AutoML 超参数优化 */}
        <div className="mb-6 p-4 bg-gradient-to-r from-orange-50 to-amber-50 rounded-xl border border-orange-200">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Icons.Sparkles />
              <span className="font-semibold text-gray-800">AutoML 超参数优化</span>
              <HelpTooltip title="AutoML" content="自动搜索最优超参数组合，省去手动调参的繁琐过程" />
            </div>
            <button
              onClick={() => {
                setShowAutoML(!showAutoML);
                if (!showAutoML) loadAutoMLExperiments();
              }}
              className="px-3 py-1.5 bg-orange-500 text-white rounded-lg text-sm font-medium hover:bg-orange-600"
            >
              {showAutoML ? '收起' : '打开'}
            </button>
          </div>
          
          {!showAutoML && recommendedParams && (
            <div className="p-3 bg-white rounded-lg border border-orange-200">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-sm text-gray-600">已找到最优参数 (准确率: {(recommendedParams.accuracy * 100).toFixed(1)}%)</span>
                  <div className="text-xs text-gray-500 mt-1">
                    {Object.entries(recommendedParams.recommended).slice(0, 3).map(([k, v]) => `${k}: ${v}`).join(' · ')}
                  </div>
                </div>
                <button
                  onClick={applyRecommendedParams}
                  className="px-4 py-2 bg-green-500 text-white rounded-lg text-sm font-medium hover:bg-green-600"
                >
                  应用到配置
                </button>
              </div>
            </div>
          )}
          
          {showAutoML && (
            <div className="space-y-3">
              {/* 创建实验 */}
              <div className="p-3 bg-white rounded-lg">
                <div className="grid grid-cols-3 gap-2 mb-2">
                  <input
                    type="text"
                    placeholder="实验名称"
                    value={autoMLForm.name}
                    onChange={(e) => setAutoMLForm({...autoMLForm, name: e.target.value})}
                    className="px-3 py-2 rounded-lg border text-sm"
                  />
                  <input
                    type="number"
                    placeholder="Trial数量"
                    value={autoMLForm.max_trials}
                    onChange={(e) => setAutoMLForm({...autoMLForm, max_trials: parseInt(e.target.value) || 10})}
                    className="px-3 py-2 rounded-lg border text-sm"
                  />
                  <button
                    onClick={createAutoMLExperiment}
                    disabled={!autoMLForm.name || !selectedDataset}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium disabled:opacity-50"
                  >
                    创建实验
                  </button>
                </div>
                {!selectedDataset && <p className="text-xs text-orange-600">⚠️ 请先选择数据集</p>}
              </div>
              
              {/* 实验列表 */}
              <div className="space-y-2 max-h-48 overflow-auto">
                {autoMLExperiments.length === 0 ? (
                  <p className="text-sm text-gray-500 text-center py-2">暂无实验，创建一个开始搜索最优参数</p>
                ) : (
                  autoMLExperiments.map(exp => (
                    <div key={exp.id} className="p-3 bg-white rounded-lg flex items-center justify-between">
                      <div>
                        <div className="font-medium text-sm">{exp.name}</div>
                        <div className="text-xs text-gray-500">
                          {exp.trials_count || 0}/{exp.max_trials} trials
                          {exp.best_accuracy && ` · 最佳准确率: ${(exp.best_accuracy * 100).toFixed(1)}%`}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => runAutoMLTrial(exp.id)}
                          disabled={exp.trials_count >= exp.max_trials}
                          className="px-3 py-1.5 bg-purple-100 text-purple-700 rounded text-xs font-medium disabled:opacity-50"
                        >
                          {exp.trials_count >= exp.max_trials ? '已完成' : '运行Trial'}
                        </button>
                        <button
                          onClick={() => getRecommendedParams(exp.id)}
                          className="px-3 py-1.5 bg-green-100 text-green-700 rounded text-xs font-medium"
                        >
                          查看结果
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
              
              {/* 推荐参数预览 */}
              {recommendedParams && selectedExperiment && (
                <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm text-green-800">推荐参数配置</span>
                    <button
                      onClick={applyRecommendedParams}
                      className="px-4 py-1.5 bg-green-500 text-white rounded text-xs font-medium hover:bg-green-600"
                    >
                      应用到训练配置
                    </button>
                  </div>
                  <div className="text-xs text-green-700 space-y-1">
                    {Object.entries(recommendedParams.recommended).map(([k, v]) => (
                      <div key={k} className="flex justify-between">
                        <span>{k}:</span>
                        <span className="font-mono">{typeof v === 'number' ? v.toExponential ? v.toExponential(2) : v : v}</span>
                      </div>
                    ))}
                  </div>
                  {recommendedParams.accuracy && (
                    <div className="mt-2 pt-2 border-t border-green-200 text-xs text-green-800">
                      预期准确率: {(recommendedParams.accuracy * 100).toFixed(2)}%
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
        
        <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
          高级配置
          <HelpTooltip title="训练配置" content="根据数据量和硬件条件调整参数，不确定就用默认值" />
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              选择数据集
              <HelpTooltip title="数据集" content="用于训练的标注数据，需要预先上传" />
            </label>
            <select
              value={selectedDataset}
              onChange={(e) => {
                setSelectedDataset(e.target.value);
                loadDatasetColumns(e.target.value);
              }}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
            >
              <option value="">请选择</option>
              {datasets.map(ds => (
                <option key={ds.id} value={ds.id}>{ds.name}</option>
              ))}
            </select>
          </div>
          
          {isNLP ? (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                预训练模型
                <HelpTooltip title="预训练模型" content="基于哪个基础模型微调。BERT更准确，DistilBERT更快" />
              </label>
              <select
                value={config.model_name}
                onChange={(e) => setConfig({...config, model_name: e.target.value})}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
              >
                <option value="bert-base-chinese">BERT-Base-Chinese（准确率高）</option>
                <option value="distilbert-base-chinese">DistilBERT-Chinese（速度快）</option>
              </select>
              {selectedModel && (
                <div className="mt-2 p-2 bg-gray-50 rounded text-xs text-gray-600">
                  <div className="font-medium">{selectedModel.name}</div>
                  <div>{selectedModel.desc}</div>
                </div>
              )}
            </div>
          ) : (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                目标列（要预测的值）
                <HelpTooltip title="目标列" content="你希望模型预测的那一列数据" />
              </label>
              <select
                value={config.target_column}
                onChange={(e) => setConfig({...config, target_column: e.target.value})}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
              >
                <option value="">请选择</option>
                {columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
          )}
        </div>
        
        {/* ML任务：特征列选择 */}
        {!isNLP && columns.length > 0 && (
          <div className="mb-4 p-4 bg-gray-50 rounded-xl">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              特征列（用于预测的字段）
              <HelpTooltip title="特征列" content="用于预测目标值的输入字段，可以多选" />
            </label>
            <div className="flex flex-wrap gap-2">
              {columns.filter(c => c !== config.target_column).map(col => (
                <label key={col} className="flex items-center gap-1 px-3 py-1 bg-white rounded-lg border cursor-pointer hover:bg-purple-50">
                  <input
                    type="checkbox"
                    checked={config.feature_columns && feature_columns.includes(col)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setConfig({...config, feature_columns: [...(config.feature_columns || []), col]});
                      } else {
                        setConfig({...config, feature_columns: config.feature_columns && feature_columns.filter(c => c !== col)});
                      }
                    }}
                  />
                  <span className="text-sm">{col}</span>
                </label>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-2">
              已选择 {config.feature_columns && feature_columns.length || 0} 个特征列
            </p>
          </div>
        )}
        
        {/* 时序分析特有配置 */}
        {isTimeSeries && columns.length > 0 && (
          <div className="mb-4 p-4 bg-gradient-to-r from-orange-50 to-yellow-50 rounded-xl border border-orange-200">
            <h4 className="font-medium text-orange-800 mb-3 flex items-center gap-2">
              <Icons.Chart /> 时序分析配置
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  时间列
                  <HelpTooltip title="时间列" content="包含时间戳的列，用于识别数据的时间顺序" />
                </label>
                <select
                  value={config.time_column}
                  onChange={(e) => setConfig({...config, time_column: e.target.value})}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
                >
                  {columns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  预测时长（小时）
                  <HelpTooltip title="预测时长" content="预测未来多长时间的趋势" />
                </label>
                <select
                  value={config.forecast_horizon}
                  onChange={(e) => setConfig({...config, forecast_horizon: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
                >
                  <option value={12}>12小时</option>
                  <option value={24}>24小时</option>
                  <option value={48}>48小时</option>
                  <option value={72}>72小时</option>
                </select>
              </div>
            </div>
          </div>
        )}
        
        {/* 图片分类特有配置 */}
        {isImage && (
          <div className="mb-4 p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border border-purple-200">
            <h4 className="font-medium text-purple-800 mb-3 flex items-center gap-2">
              <Icons.Chart /> 图像分类配置
            </h4>
            
            {/* 模型选择 */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                预训练模型
                <HelpTooltip title="模型选择" content="轻量级模型适合边缘部署，大模型精度更高" />
              </label>
              <select
                value={config.model_name}
                onChange={(e) => setConfig({...config, model_name: e.target.value})}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
              >
                <option value="mobilenet_v2">MobileNet V2 (轻量，14MB)</option>
                <option value="resnet18">ResNet-18 (小型，45MB)</option>
                <option value="resnet50">ResNet-50 (标准，98MB)</option>
                <option value="efficientnet_b0">EfficientNet-B0 (高效，21MB)</option>
                <option value="vit_base">Vision Transformer (高精度，346MB)</option>
              </select>
              {IMAGE_MODEL_HELP[config.model_name] && (
                <div className="mt-2 p-2 bg-white rounded-lg text-sm text-gray-600">
                  <span className="font-medium">{IMAGE_MODEL_HELP[config.model_name].name}</span>
                  <span className="mx-2">•</span>
                  <span>{IMAGE_MODEL_HELP[config.model_name].desc}</span>
                </div>
              )}
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  图片尺寸
                  <HelpTooltip title="图片尺寸" content="输入网络的图片大小，224是标准值" />
                </label>
                <select
                  value={config.image_size}
                  onChange={(e) => setConfig({...config, image_size: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
                >
                  <option value={224}>224×224 (标准)</option>
                  <option value={256}>256×256 (高精度)</option>
                  <option value={384}>384×384 (超精度)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  批次大小
                  <HelpTooltip title="批次大小" content="每次训练的样本数，显存大可以设大点" />
                </label>
                <select
                  value={config.batch_size}
                  onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
                >
                  <option value={8}>8 (低显存)</option>
                  <option value={16}>16 (推荐)</option>
                  <option value={32}>32 (高显存)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  训练轮数
                  <HelpTooltip title="训练轮数" content="数据集遍历次数，10-20轮通常够用" />
                </label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={config.epochs}
                  onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
                />
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <label className="flex items-center gap-2 p-3 bg-white rounded-lg cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.freeze_backbone}
                  onChange={(e) => setConfig({...config, freeze_backbone: e.target.checked})}
                  className="w-4 h-4"
                />
                <div>
                  <span className="text-sm font-medium text-gray-700">冻结骨干网络</span>
                  <p className="text-xs text-gray-500">只训练分类头，适合小数据集</p>
                </div>
              </label>
              <label className="flex items-center gap-2 p-3 bg-white rounded-lg cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.quantize}
                  onChange={(e) => setConfig({...config, quantize: e.target.checked})}
                  className="w-4 h-4"
                />
                <div>
                  <span className="text-sm font-medium text-gray-700">INT8量化</span>
                  <p className="text-xs text-gray-500">导出压缩模型，适合边缘部署</p>
                </div>
              </label>
            </div>
          </div>
        )}

        {/* 目标检测特有配置 */}
        {isDetection && (
          <div className="mb-4 p-4 bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl border border-blue-200">
            <h4 className="font-medium text-blue-800 mb-3 flex items-center gap-2">
              <Icons.Chart /> 目标检测配置 (YOLOv8)
            </h4>

            {/* 模型选择 */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                YOLO模型
                <HelpTooltip title="YOLO模型" content="Nano最快，Large最准，根据场景选择" />
              </label>
              <select
                value={config.model}
                onChange={(e) => setConfig({...config, model: e.target.value})}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
              >
                <option value="yolov8n">YOLOv8 Nano (6MB, 最快)</option>
                <option value="yolov8s">YOLOv8 Small (22MB, 平衡)</option>
                <option value="yolov8m">YOLOv8 Medium (54MB, 高精度)</option>
                <option value="yolov8l">YOLOv8 Large (89MB, 最高精度)</option>
              </select>
              {YOLO_MODEL_HELP[config.model] && (
                <div className="mt-2 p-2 bg-white rounded-lg text-sm text-gray-600">
                  <span className="font-medium">{YOLO_MODEL_HELP[config.model].name}</span>
                  <span className="mx-2">•</span>
                  <span>{YOLO_MODEL_HELP[config.model].desc}</span>
                  <span className="mx-2">•</span>
                  <span>mAP: {YOLO_MODEL_HELP[config.model].ap}%</span>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  图片尺寸
                  <HelpTooltip title="图片尺寸" content="YOLO标准640，大图片可用1280" />
                </label>
                <select
                  value={config.image_size}
                  onChange={(e) => setConfig({...config, image_size: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
                >
                  <option value={640}>640×640 (标准)</option>
                  <option value={1280}>1280×1280 (高精度)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  批次大小
                  <HelpTooltip title="批次大小" content="根据显存调整" />
                </label>
                <select
                  value={config.batch_size}
                  onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
                >
                  <option value={8}>8 (低显存)</option>
                  <option value={16}>16 (推荐)</option>
                  <option value={32}>32 (高显存)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  训练轮数
                  <HelpTooltip title="训练轮数" content="YOLO通常100轮以上" />
                </label>
                <input
                  type="number"
                  min={10}
                  max={500}
                  step={10}
                  value={config.epochs}
                  onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  置信度阈值
                  <HelpTooltip title="置信度阈值" content="低于此值的检测框会被过滤" />
                </label>
                <input
                  type="number"
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={config.confidence_threshold}
                  onChange={(e) => setConfig({...config, confidence_threshold: parseFloat(e.target.value)})}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
                />
                <p className="text-xs text-gray-500 mt-1">建议：0.25-0.5</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  NMS IoU阈值
                  <HelpTooltip title="NMS IoU阈值" content="去除重叠框的阈值，越大保留越多框" />
                </label>
                <input
                  type="number"
                  min={0.3}
                  max={0.7}
                  step={0.05}
                  value={config.iou_threshold}
                  onChange={(e) => setConfig({...config, iou_threshold: parseFloat(e.target.value)})}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
                />
                <p className="text-xs text-gray-500 mt-1">建议：0.45-0.65</p>
              </div>
            </div>

            <div className="mt-4 p-3 bg-blue-100 rounded-lg text-sm text-blue-800">
              <strong>💡 目标检测 vs 图像分类：</strong><br/>
              目标检测不仅能识别<span className="font-medium">「有什么」</span>，还能定位<span className="font-medium">「在哪里」</span>，
              支持一张图检测多个目标，适合缺陷定位、物体计数等场景。
            </div>
          </div>
        )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              学习率
              <HelpTooltip {...PARAM_HELP.learning_rate} />
            </label>
            <select
              value={config.learning_rate}
              onChange={(e) => setConfig({...config, learning_rate: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
            >
              <option value={5e-5}>5e-5（快但可能不稳定）</option>
              <option value={2e-5}>2e-5（推荐）</option>
              <option value={1e-5}>1e-5（慢但更稳定）</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              批次大小
              <HelpTooltip {...PARAM_HELP.batch_size} />
            </label>
            <select
              value={config.batch_size}
              onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
            >
              <option value={4}>4（内存小）</option>
              <option value={8}>8（推荐）</option>
              <option value={16}>16（内存大）</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              训练轮数
              <HelpTooltip {...PARAM_HELP.epochs} />
            </label>
            <input
              type="number"
              min={1}
              max={10}
              value={config.epochs}
              onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
            />
            <p className="text-xs text-gray-500 mt-1">建议：3-5轮，数据多可以设大点</p>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              最大长度
              <HelpTooltip {...PARAM_HELP.max_length} />
            </label>
            <select
              value={config.max_length}
              onChange={(e) => setConfig({...config, max_length: parseInt(e.target.value)})}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
            >
              <option value={128}>128（短文本，更快）</option>
              <option value={256}>256（中等长度）</option>
              <option value={512}>512（长文本，更慢）</option>
            </select>
          </div>
        </div>
        
        {/* 自动调参配置 */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <h4 className="font-medium text-gray-800 mb-4 flex items-center gap-2">
            <Icons.Settings /> 自动调参设置
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 学习率调度器 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                学习率调度
                <HelpTooltip title="学习率调度" content="控制学习率如何随训练轮数变化，影响收敛效果" />
              </label>
              <select
                value={config.lr_scheduler_type || 'cosine'}
                onChange={(e) => setConfig({...config, lr_scheduler_type: e.target.value})}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
              >
                <option value="linear">线性衰减 - 最稳定</option>
                <option value="cosine">余弦退火 - 平滑收敛</option>
                <option value="cosine_with_restarts">余弦重启 - 可能更好</option>
                <option value="polynomial">多项式衰减</option>
                <option value="constant">恒定不变 - 微调用</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {(LR_SCHEDULER_HELP[config.lr_scheduler_type || 'cosine'] && LR_SCHEDULER_HELP[config.lr_scheduler_type || 'cosine'].desc)}
              </p>
            </div>
            
            {/* Warmup比例 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Warmup比例
                <HelpTooltip title="Warmup" content="训练初期学习率从0逐渐上升到初始值的比例，帮助模型稳定启动" />
              </label>
              <select
                value={config.warmup_ratio || 0.1}
                onChange={(e) => setConfig({...config, warmup_ratio: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
              >
                <option value={0}>0% - 不warmup</option>
                <option value={0.05}>5% - 快速启动</option>
                <option value={0.1}>10% - 推荐</option>
                <option value={0.2}>20% - 保守启动</option>
              </select>
            </div>
            
            {/* 早停开关 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                早停机制
                <HelpTooltip title="早停" content="验证指标不再提升时自动停止训练，防止过拟合" />
              </label>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setConfig({...config, early_stopping: !config.early_stopping})}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    config.early_stopping 
                      ? 'bg-green-100 text-green-700 border border-green-300' 
                      : 'bg-gray-100 text-gray-600 border border-gray-200'
                  }`}
                >
                  {config.early_stopping ? '✓ 已开启' : '✗ 已关闭'}
                </button>
                <span className="text-xs text-gray-500">
                  {config.early_stopping ? '防止过拟合，推荐开启' : '可能过拟合，训练更久'}
                </span>
              </div>
            </div>
            
            {/* 早停耐心值 */}
            <div className={config.early_stopping ? '' : 'opacity-50'}>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                早停耐心值
                <HelpTooltip {...EARLY_STOPPING_HELP.patience} />
              </label>
              <select
                value={config.early_stopping_patience || 3}
                onChange={(e) => setConfig({...config, early_stopping_patience: parseInt(e.target.value)})}
                disabled={!config.early_stopping}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none disabled:bg-gray-100"
              >
                <option value={1}>1轮 - 严格（易早停）</option>
                <option value={2}>2轮 - 较严</option>
                <option value={3}>3轮 - 推荐</option>
                <option value={5}>5轮 - 宽松</option>
              </select>
            </div>
            
            {/* 早停阈值 */}
            <div className={config.early_stopping ? '' : 'opacity-50'}>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                改善阈值
                <HelpTooltip {...EARLY_STOPPING_HELP.threshold} />
              </label>
              <select
                value={config.early_stopping_threshold || 0.001}
                onChange={(e) => setConfig({...config, early_stopping_threshold: parseFloat(e.target.value)})}
                disabled={!config.early_stopping}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none disabled:bg-gray-100"
              >
                <option value={0.0001}>0.01% - 极严格</option>
                <option value={0.0005}>0.05% - 严格</option>
                <option value={0.001}>0.1% - 推荐</option>
                <option value={0.005}>0.5% - 宽松</option>
                <option value={0.01}>1% - 极宽松</option>
              </select>
            </div>
          </div>
        </div>

        <button
          onClick={startTraining}
          disabled={starting || !selectedDataset}
          className="w-full mt-6 py-3 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 text-white rounded-xl font-semibold hover:shadow-lg transition-all disabled:opacity-50"
        >
          {starting ? <Icons.Loading /> : <Icons.Train />} 开始训练
        </button>

      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4">训练任务</h3>
        <div className="space-y-3">
          {jobs.map(job => (
            <div key={job.id} className="p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-gray-800">{job.model_name}</span>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-1 rounded-full text-xs ${
                    job.status === 'completed' ? 'bg-green-100 text-green-700' :
                    job.status === 'training' ? 'bg-blue-100 text-blue-700 animate-pulse' :
                    'bg-yellow-100 text-yellow-700'
                  }`}>
                    {job.status === 'pending' ? '等待中' : 
                     job.status === 'training' ? '训练中' : 
                     job.status === 'failed' ? '失败' : '已完成'}
                  </span>
                  <button
                    onClick={() => setDetailJob(job)}
                    className="px-3 py-1 bg-purple-100 text-purple-700 rounded-lg text-xs hover:bg-purple-200 flex items-center gap-1"
                  >
                    <Icons.Chart /> 详情
                  </button>
                  {job.status === 'failed' && (
                    <button
                      onClick={() => retryJob(job.id)}
                      className="px-3 py-1 bg-blue-100 text-blue-700 rounded-lg text-xs hover:bg-blue-200 flex items-center gap-1"
                    >
                      <Icons.Refresh /> 重试
                    </button>
                  )}
                  <button
                    onClick={() => deleteJob(job.id)}
                    className="px-3 py-1 bg-red-100 text-red-700 rounded-lg text-xs hover:bg-red-200 flex items-center gap-1"
                  >
                    <Icons.Trash /> 删除
                  </button>
                </div>
              </div>
              <div className="flex items-center gap-4 text-sm text-gray-500">
                <span>轮次: {job.current_epoch}/{job.total_epochs}</span>
                <span>进度: {job.progress}%</span>
                {job.best_accuracy && <span>最佳准确率: {(job.best_accuracy * 100).toFixed(2)}%</span>}
                {job.early_stopped && <span className="text-orange-600">⚠️ 早停</span>}
              </div>
              {job.status === 'training' && (
                <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 transition-all"
                    style={{width: `${job.progress}%`}}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
        
        {jobs.length === 0 && (
          <div className="text-center py-8 text-gray-400">
            <Icons.Train />
            <p className="mt-2">暂无训练任务</p>
            <p className="text-sm">选择数据集开始训练</p>
          </div>
        )}
      </div>
      
      {/* 训练详情弹窗 */}
      {detailJob && (
        <TrainingDetailModal 
          projectId={projectId}
          job={detailJob}
          onClose={() => setDetailJob(null)}
        />
      )}
    </div>
  );
}

// 在线测试组件 - 支持NLP和ML模型
function TestPanel({ endpoint, projectType, projectId, sensorData = null }) {
  const [testText, setTestText] = useState('');
  const [testFeatures, setTestFeatures] = useState({});
  const [testResult, setTestResult] = useState(null);
  const [testing, setTesting] = useState(false);
  const [llmAnalyzing, setLlmAnalyzing] = useState(false);
  const [llmResult, setLlmResult] = useState(null);
  
  const isNLP = !projectType || projectType === 'text_classification';

  const runTest = async () => {
    setTesting(true);
    try {
      let body;
      if (isNLP) {
        body = new URLSearchParams({ text: testText });
      } else {
        // ML模型：构建特征JSON
        body = new URLSearchParams({ features: JSON.stringify(testFeatures) });
      }
      
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        body: body
      });
      const data = await res.json();
      if (!res.ok) {
        setTestResult({ error: data.detail || data.error || `请求失败 (${res.status})` });
        setTesting(false);
        return;
      }
      setTestResult(data);
    } catch (e) {
      setTestResult({ error: e.message });
    }
    setTesting(false);
  };

  return (
    <div className="mt-4 p-4 bg-blue-50 rounded-xl border border-blue-200">
      <h5 className="font-medium text-blue-800 mb-3 flex items-center gap-2">
        🧪 在线测试
      </h5>
      
      {isNLP ? (
        <div className="flex gap-2 mb-3">
          <input
            type="text"
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            placeholder="输入测试文本..."
            className="flex-1 px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-blue-400 focus:outline-none text-sm"
            onKeyPress={(e) => e.key === 'Enter' && runTest()}
          />
          <button
            onClick={runTest}
            disabled={testing || !testText.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50"
          >
            {testing ? <Icons.Loading /> : '测试'}
          </button>
        </div>
      ) : (
        <div className="mb-3">
          <p className="text-xs text-gray-600 mb-2">输入特征值（JSON格式）:</p>
          <textarea
            value={JSON.stringify(testFeatures, null, 2)}
            onChange={(e) => {
              try {
                setTestFeatures(JSON.parse(e.target.value));
              } catch (e) {}
            }}
            placeholder='{"温度": 85, "压力": 12, ...}'
            className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white text-xs font-mono"
            rows={3}
          />
          <button
            onClick={runTest}
            disabled={testing}
            className="mt-2 px-4 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50"
          >
            {testing ? <Icons.Loading /> : '测试'}
          </button>
        </div>
      )}

      {testResult && (
        <div className={`p-3 rounded-lg text-sm ${testResult.error ? 'bg-red-50 border border-red-200' : 'bg-white border border-blue-200'}`}>
          {testResult.error ? (
            <div className="text-red-700">测试失败: {testResult.error}</div>
          ) : (
            <div>
              {isNLP && (
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-gray-600">输入:</span>
                  <span className="text-gray-800 font-medium">"{testResult.result && result.text || testText}"</span>
                </div>
              )}
              <div className="flex items-center gap-2 mb-2">
                <span className="text-gray-600">预测结果:</span>
                <span className={`px-2 py-1 rounded font-medium ${
                  testResult.result && result.prediction === '正面' || testResult.result && result.prediction === '正常' ? 'bg-green-100 text-green-700' :
                  testResult.result && result.prediction === '负面' || testResult.result && result.prediction === '故障' ? 'bg-red-100 text-red-700' :
                  'bg-gray-100 text-gray-700'
                }`}>
                  {testResult.result && result.prediction || (testResult.result && result.is_anomaly ? '异常' : '正常')}
                </span>
                {testResult.result && result.confidence && (
                  <span className="text-gray-500">
                    (置信度: {(testResult.result && result.confidence * 100).toFixed(1)}%)
                  </span>
                )}
              </div>
              
              {/* 概率分布条形图 */}
              {testResult.result && result.all_probabilities && (
                <div className="mt-3 space-y-1">
                  <div className="text-xs text-gray-500 mb-1">各类别概率分布:</div>
                  {Object.entries(testResult.result.all_probabilities)
                    .sort(([,a], [,b]) => b - a)
                    .map(([label, prob]) => (
                      <div key={label} className="flex items-center gap-2">
                        <span className="w-12 text-xs text-gray-600">{label}</span>
                        <div className="flex-1 h-4 bg-gray-100 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${
                              label === '正面' || label === '正常' ? 'bg-green-400' :
                              label === '负面' || label === '故障' ? 'bg-red-400' :
                              'bg-gray-400'
                            }`}
                            style={{width: `${prob * 100}%`}}
                          />
                        </div>
                        <span className="w-12 text-xs text-gray-600 text-right">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))
                  }
                </div>
              )}
              
              <div className="text-xs text-gray-400 mt-2">
                响应时间: &lt;100ms | 设备: {testResult.device}
              </div>
              
              {/* DeepSeek 根因分析按钮 */}
              {(testResult.result && result.prediction === '故障' || testResult.result && result.is_anomaly || (testResult.result && result.confidence && testResult.result.confidence > 0.5)) && projectId && (
                <div className="mt-4 pt-3 border-t border-blue-200">
                  {!llmResult ? (
                    <button
                      onClick={async () => {
                        setLlmAnalyzing(true);
                        try {
                          const sensorData = isNLP 
                            ? { text: testText }
                            : testFeatures;
                          
                          const formData = new FormData();
                          formData.append('sensor_data', JSON.stringify(sensorData));
                          formData.append('prediction_result', JSON.stringify({
                            failure_probability: testResult.result && result.confidence || 0.8,
                            key_risk_factors: testResult.result && result.prediction || '异常检测'
                          }));
                          
                          const res = await fetch(`${API_BASE}/projects/${projectId}/analyze-root-cause`, {
                            method: 'POST',
                            body: formData
                          });
                          
                          const data = await res.json();
                          if (data.success) {
                            setLlmResult(data.analysis);
                          } else {
                            alert('分析失败: ' + (data.error || '未知错误'));
                          }
                        } catch (e) {
                          alert('调用失败: ' + e.message);
                        }
                        setLlmAnalyzing(false);
                      }}
                      disabled={llmAnalyzing}
                      className="w-full py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg text-sm font-medium hover:shadow-lg disabled:opacity-50 flex items-center justify-center gap-2"
                    >
                      {llmAnalyzing ? <Icons.Loading /> : <Icons.Sparkles />}
                      {llmAnalyzing ? 'DeepSeek分析中...' : '✨ DeepSeek 根因分析'}
                    </button>
                  ) : (
                    <div className="bg-purple-50 rounded-lg p-3 border border-purple-200">
                      <div className="flex items-center gap-2 mb-2">
                        <Icons.Sparkles />
                        <span className="font-medium text-purple-800">DeepSeek 根因分析</span>
                      </div>
                      <div className="text-sm text-purple-700 whitespace-pre-wrap">
                        {llmResult}
                      </div>
                      <button
                        onClick={() => setLlmResult(null)}
                        className="mt-2 text-xs text-purple-600 hover:text-purple-800"
                      >
                        重新分析
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// 模型版本管理Tab
function DeployTab({ projectId, jobs, datasets, projectType }) {
  const completedJobs = jobs.filter(j => j.status === 'completed');
  const [activeSubTab, setActiveSubTab] = useState('deploy'); // deploy, versions, batch, schedule
  const [deployStatus, setDeployStatus] = useState({});
  const [models, setModels] = useState([]);
  const [schedules, setSchedules] = useState([]);
  const [batchFile, setBatchFile] = useState(null);
  const [batchPredicting, setBatchPredicting] = useState(false);
  const [scheduleConfig, setScheduleConfig] = useState({
    name: '',
    dataset_id: '',
    schedule: 'daily',
    config: '{}'
  });
  const [selectedModels, setSelectedModels] = useState([]);

  // 加载模型版本列表
  const loadModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/models`);
      const data = await res.json();
      setModels(data.models || []);
    } catch (e) {}
  };

  // 加载定时任务
  const loadSchedules = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/schedules`);
      const data = await res.json();
      setSchedules(data.schedules || []);
    } catch (e) {}
  };

  useEffect(() => {
    loadModels();
    loadSchedules();
  }, [projectId]);

  const deploy = async (jobId) => {
    setDeployStatus({...deployStatus, [jobId]: { loading: true }});
    try {
      const formData = new FormData();
      formData.append('deploy_type', 'api');
      
      const res = await fetch(`${API_BASE}/projects/${projectId}/models/${jobId}/deploy`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      
      setDeployStatus({
        ...deployStatus, 
        [jobId]: { loading: false, success: data.success, ...data }
      });
      loadModels(); // 刷新模型列表
    } catch (e) {
      setDeployStatus({
        ...deployStatus, 
        [jobId]: { loading: false, error: e.message }
      });
    }
  };

  const deleteModel = async (jobId) => {
    if (!confirm('确定要删除这个模型版本吗？此操作不可恢复。')) return;
    
    try {
      await fetch(`${API_BASE}/projects/${projectId}/models/${jobId}`, {
        method: 'DELETE'
      });
      loadModels();
    } catch (e) {
      alert('删除失败: ' + e.message);
    }
  };

  const handleBatchPredict = async (jobId) => {
    if (!batchFile) {
      alert('请先选择文件');
      return;
    }
    
    setBatchPredicting(true);
    try {
      const formData = new FormData();
      formData.append('file', batchFile);
      
      const res = await fetch(`${API_BASE}/projects/${projectId}/jobs/${jobId}/batch-predict`, {
        method: 'POST',
        body: formData
      });
      
      if (res.ok) {
        // 下载结果文件
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `predictions_${batchFile.name}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert('批量预测失败');
      }
    } catch (e) {
      alert('批量预测失败: ' + e.message);
    }
    setBatchPredicting(false);
  };

  const createSchedule = async () => {
    try {
      const formData = new FormData();
      formData.append('name', scheduleConfig.name);
      formData.append('dataset_id', scheduleConfig.dataset_id);
      formData.append('schedule', scheduleConfig.schedule);
      formData.append('config', scheduleConfig.config);
      
      await fetch(`${API_BASE}/projects/${projectId}/schedule`, {
        method: 'POST',
        body: formData
      });
      
      loadSchedules();
      setScheduleConfig({ name: '', dataset_id: '', schedule: 'daily', config: '{}' });
    } catch (e) {
      alert('创建定时任务失败');
    }
  };

  const deleteSchedule = async (scheduleId) => {
    if (!confirm('确定删除此定时任务？')) return;
    
    try {
      await fetch(`${API_BASE}/projects/${projectId}/schedules/${scheduleId}`, {
        method: 'DELETE'
      });
      loadSchedules();
    } catch (e) {
      alert('删除失败');
    }
  };

  return (
    <div className="space-y-4">
      {/* 子Tab导航 */}
      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-2 border border-gray-200/50">
        <div className="flex gap-2">
          <button
            onClick={() => setActiveSubTab('deploy')}
            className={`flex-1 py-2 rounded-xl text-sm font-medium transition-all ${
              activeSubTab === 'deploy' ? 'bg-purple-100 text-purple-700' : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <Icons.Rocket /> 模型部署
          </button>
          <button
            onClick={() => setActiveSubTab('versions')}
            className={`flex-1 py-2 rounded-xl text-sm font-medium transition-all ${
              activeSubTab === 'versions' ? 'bg-purple-100 text-purple-700' : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <Icons.Database /> 版本管理
          </button>
          <button
            onClick={() => setActiveSubTab('batch')}
            className={`flex-1 py-2 rounded-xl text-sm font-medium transition-all ${
              activeSubTab === 'batch' ? 'bg-purple-100 text-purple-700' : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <Icons.File /> 批量预测
          </button>
          <button
            onClick={() => setActiveSubTab('schedule')}
            className={`flex-1 py-2 rounded-xl text-sm font-medium transition-all ${
              activeSubTab === 'schedule' ? 'bg-purple-100 text-purple-700' : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <Icons.Clock /> 定时训练
          </button>
        </div>
      </div>

      {/* 模型部署 */}
      {activeSubTab === 'deploy' && (
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
          <h3 className="font-semibold text-gray-800 mb-4">模型部署</h3>
          
          {/* 内存监控面板 */}
          <MemoryMonitor />
          
          {completedJobs.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Icons.Rocket />
              <p className="mt-2">暂无已完成的训练任务</p>
            </div>
          ) : (
            <div className="space-y-4">
              {completedJobs.map(job => {
                const status = deployStatus[job.id];
                const isDeployed = status && status.success || job.is_deployed;
                
                return (
                  <div key={job.id} className="p-4 bg-gray-50 rounded-xl">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <h4 className="font-medium text-gray-800">{job.model_name}</h4>
                        <p className="text-sm text-gray-500">
                          准确率: {job.best_accuracy ? (job.best_accuracy * 100).toFixed(2) + '%' : '-'}
                          {' · '}
                          创建于 {new Date(job.created_at).toLocaleDateString()}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        {!isDeployed ? (
                          <button
                            onClick={() => deploy(job.id)}
                            disabled={status && status.loading}
                            className="px-4 py-2 bg-cyan-500 text-white rounded-lg text-sm font-medium hover:bg-cyan-600 disabled:opacity-50"
                          >
                            {status && status.loading ? <Icons.Loading /> : <Icons.Rocket />} 部署
                          </button>
                        ) : (
                          <span className="px-3 py-1 bg-green-100 text-green-700 rounded-lg text-sm flex items-center gap-1">
                            <Icons.Check /> 已部署
                          </span>
                        )}
                      </div>
                    </div>

                    {status && status.success && (
                      <div className="mt-3 p-3 bg-green-50 rounded-lg text-sm">
                        <div className="flex items-center gap-2 text-green-700 font-medium mb-2">
                          <Icons.Check /> 部署成功
                        </div>
                        <div className="space-y-2">
                          <div className="text-gray-700">
                            <span className="font-medium">API端点：</span>
                            <code className="bg-white px-2 py-1 rounded text-cyan-600">{status.endpoint}</code>
                          </div>
                          <div className="bg-gray-800 text-green-400 p-2 rounded text-xs font-mono overflow-x-auto">
                            {`curl -X POST "http://你的IP${status.endpoint}" \\
  -F "text=要预测的文本"`}
                          </div>
                          <TestPanel endpoint={status.endpoint} projectType={projectType} projectId={projectId} />
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* 版本管理 */}
      {activeSubTab === 'versions' && (
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
          <h3 className="font-semibold text-gray-800 mb-4">模型版本管理</h3>
          
          {models.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Icons.Database />
              <p className="mt-2">暂无模型版本</p>
            </div>
          ) : (
            <div className="space-y-3">
              {models.map(model => (
                <div key={model.id} className="p-4 bg-gray-50 rounded-xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(model.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedModels([...selectedModels, model.id]);
                          } else {
                            setSelectedModels(selectedModels.filter(id => id !== model.id));
                          }
                        }}
                        className="w-4 h-4"
                      />
                      <div>
                        <div className="font-medium text-gray-800">{model.name}</div>
                        <div className="text-sm text-gray-500">
                          准确率: {model.best_accuracy ? (model.best_accuracy * 100).toFixed(2) + '%' : '-'} · 
                          {new Date(model.created_at).toLocaleDateString()}
                        </div>
                      </div>
                      {model.is_deployed && (
                        <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs">
                          已部署
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => deleteModel(model.id)}
                      className="px-3 py-1.5 bg-red-100 text-red-700 rounded-lg text-sm hover:bg-red-200"
                    >
                      <Icons.Trash />
                    </button>
                  </div>
                </div>
              ))}
              
              {selectedModels.length >= 2 && (
                <div className="p-4 bg-blue-50 rounded-xl">
                  <button
                    onClick={async () => {
                      const res = await fetch(`${API_BASE}/projects/${projectId}/models/compare?model_ids=${selectedModels.join(',')}`);
                      const data = await res.json();
                      // 显示对比结果
                      alert(`已选择 ${selectedModels.length} 个模型进行对比，详细对比功能待完善`);
                    }}
                    className="w-full py-2 bg-blue-500 text-white rounded-lg font-medium"
                  >
                    <Icons.Compare /> 对比选中的 {selectedModels.length} 个模型
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* 批量预测 */}
      {activeSubTab === 'batch' && (
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
          <h3 className="font-semibold text-gray-800 mb-4">批量预测</h3>
          
          {completedJobs.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Icons.File />
              <p className="mt-2">请先完成模型训练</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">选择模型</label>
                <select
                  id="batch-model-select"
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
                >
                  <option value="">请选择模型</option>
                  {completedJobs.map(job => (
                    <option key={job.id} value={job.id}>{job.model_name} ({job.best_accuracy ? (job.best_accuracy * 100).toFixed(1) + '%' : '训练中'})</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">上传数据文件</label>
                <input
                  type="file"
                  accept=".csv,.xlsx"
                  onChange={(e) => setBatchFile(e.target.files[0])}
                  className="w-full px-3 py-2 border border-gray-200 rounded-lg"
                />
                <p className="text-xs text-gray-500 mt-1">
                  支持 CSV/Excel 格式，文件格式需与训练数据一致
                </p>
              </div>
              
              <button
                onClick={() => {
                  const select = document.getElementById('batch-model-select');
                  const jobId = select && select.value;
                  if (jobId) handleBatchPredict(jobId);
                }}
                disabled={batchPredicting || !batchFile}
                className="w-full py-3 bg-gradient-to-r from-cyan-500 to-purple-500 text-white rounded-xl font-semibold disabled:opacity-50"
              >
                {batchPredicting ? <Icons.Loading /> : <Icons.File />} 开始批量预测
              </button>
            </div>
          )}
        </div>
      )}

      {/* 定时训练 */}
      {activeSubTab === 'schedule' && (
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
          <h3 className="font-semibold text-gray-800 mb-4">定时训练</h3>
          
          {/* 创建新定时任务 */}
          <div className="mb-6 p-4 bg-gray-50 rounded-xl">
            <h4 className="font-medium text-gray-700 mb-3">创建定时任务</h4>
            <div className="space-y-3">
              <input
                type="text"
                placeholder="任务名称"
                value={scheduleConfig.name}
                onChange={(e) => setScheduleConfig({...scheduleConfig, name: e.target.value})}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
              />
              
              <select
                value={scheduleConfig.dataset_id}
                onChange={(e) => setScheduleConfig({...scheduleConfig, dataset_id: e.target.value})}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
              >
                <option value="">选择数据集</option>
                {datasets.map(ds => (
                  <option key={ds.id} value={ds.id}>{ds.name}</option>
                ))}
              </select>
              
              <select
                value={scheduleConfig.schedule}
                onChange={(e) => setScheduleConfig({...scheduleConfig, schedule: e.target.value})}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
              >
                <option value="daily">每天凌晨2点</option>
                <option value="weekly">每周日凌晨2点</option>
              </select>
              
              <button
                onClick={createSchedule}
                disabled={!scheduleConfig.name || !scheduleConfig.dataset_id}
                className="w-full py-2 bg-purple-600 text-white rounded-lg font-medium disabled:opacity-50"
              >
                <Icons.Clock /> 创建定时任务
              </button>
            </div>
          </div>
          
          {/* 定时任务列表 */}
          <div>
            <h4 className="font-medium text-gray-700 mb-3">定时任务列表</h4>
            {schedules.length === 0 ? (
              <p className="text-gray-500 text-center py-4">暂无定时任务</p>
            ) : (
              <div className="space-y-2">
                {schedules.map(schedule => (
                  <div key={schedule.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <div className="font-medium text-gray-800">{schedule.name}</div>
                      <div className="text-sm text-gray-500">
                        {schedule.cron === '0 2 * * *' ? '每天' : '每周'} · 
                        上次运行: {schedule.last_run_at ? new Date(schedule.last_run_at).toLocaleString() : '未运行'}
                      </div>
                    </div>
                    <button
                      onClick={() => deleteSchedule(schedule.id)}
                      className="px-3 py-1 bg-red-100 text-red-700 rounded text-sm hover:bg-red-200"
                    >
                      <Icons.Trash />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function DeployPanel({ projectId, completedJobs }) {
  const [deployStatus, setDeployStatus] = useState({});

  const deploy = async (jobId, type) => {
    setDeployStatus({...deployStatus, [jobId]: { loading: true }});
    try {
      const formData = new FormData();
      formData.append('deploy_type', type);
      
      const res = await fetch(`${API_BASE}/projects/${projectId}/models/${jobId}/deploy`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      
      setDeployStatus({
        ...deployStatus, 
        [jobId]: { 
          loading: false, 
          success: data.success,
          type: type,
          ...data
        }
      });
    } catch (e) {
      setDeployStatus({
        ...deployStatus, 
        [jobId]: { loading: false, error: e.message }
      });
    }
  };

  return (
    <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
      <h3 className="font-semibold text-gray-800 mb-4">模型部署</h3>
      
      {completedJobs.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <Icons.Rocket />
          <p className="mt-2">暂无已完成的训练任务</p>
          <p className="text-sm">请先完成模型训练</p>
        </div>
      ) : (
        <div className="space-y-4">
          {completedJobs.map(job => {
            const status = deployStatus[job.id];
            return (
              <div key={job.id} className="p-4 bg-gray-50 rounded-xl">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h4 className="font-medium text-gray-800">{job.model_name}</h4>
                    <p className="text-sm text-gray-500">准确率: {(job.best_accuracy * 100).toFixed(2)}%</p>
                  </div>
                </div>
                
                {/* 部署按钮 */}
                <div className="flex gap-2 mb-3">
                  <button
                    onClick={() => deploy(job.id, 'api')}
                    disabled={status && status.loading}
                    className="px-4 py-2 bg-cyan-500 text-white rounded-lg text-sm font-medium hover:bg-cyan-600 disabled:opacity-50"
                  >
                    {status && status.loading ? <Icons.Loading /> : <Icons.Rocket />} API部署
                  </button>
                  <button
                    onClick={() => deploy(job.id, 'ollama')}
                    disabled={status && status.loading}
                    className="px-4 py-2 bg-purple-500 text-white rounded-lg text-sm font-medium hover:bg-purple-600 disabled:opacity-50"
                  >
                    <Icons.Brain /> Ollama导出
                  </button>
                </div>

                {/* 部署状态展示（非弹框） */}
                {status && !status.loading && (
                  <div className={`p-3 rounded-lg text-sm ${
                    status.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
                  }`}>
                    {status.success ? (
                      <div>
                        <div className="flex items-center gap-2 text-green-700 font-medium mb-2">
                          <Icons.Check /> 部署成功
                        </div>
                        
                        {status.type === 'api' && (
                          <div className="space-y-2">
                            <div className="text-gray-700">
                              <span className="font-medium">API端点：</span>
                              <code className="bg-white px-2 py-1 rounded text-cyan-600">{status.endpoint}</code>
                            </div>
                            <div className="text-gray-600 text-xs">
                              调用方式：POST {status.endpoint}
                            </div>
                            <div className="bg-gray-800 text-green-400 p-2 rounded text-xs font-mono overflow-x-auto">
                              {`curl -X POST "http://你的IP${status.endpoint}" \\
  -F "text=要预测的文本"`}
                            </div>
                            <div className="text-gray-500 text-xs mt-2">
                              💡 模型已加载到内存，可直接调用，响应时间 &lt;100ms
                            </div>
                            
                            {/* 在线测试 */}
                            <TestPanel endpoint={status.endpoint} />
                          </div>
                        )}
                        
                        {status.type === 'ollama' && (
                          <div className="text-purple-700">
                            <div>Ollama导出功能开发中...</div>
                            <div className="text-xs text-purple-600 mt-1">
                              模型路径：{status.model_path}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-red-700">
                        <Icons.Error /> 部署失败: {status.error}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// 训练曲线图表组件 - 支持Loss、Accuracy、学习率
function TrainingChart({ metrics, width = 600, height = 200, type = 'loss' }) {
  const canvasRef = React.useRef(null);
  
  React.useEffect(() => {
    if (!canvasRef.current || !metrics || metrics.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(dpr, dpr);
    
    // 清空画布
    ctx.clearRect(0, 0, width, height);
    
    // 边距
    const padding = { top: 25, right: 80, bottom: 30, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    // 提取数据
    const steps = metrics.map(m => m.step);
    const maxStep = Math.max(...steps);
    
    // 根据类型选择数据
    let data1 = [], data2 = [], label1 = '', label2 = '', color1 = '', color2 = '', minVal = 0, maxVal = 1;
    
    if (type === 'loss') {
      data1 = metrics.map(m => m.train_loss).filter(v => v !== null);
      data2 = metrics.map(m => m.val_loss).filter(v => v !== null);
      label1 = 'Train Loss';
      label2 = 'Val Loss';
      color1 = '#3b82f6'; // blue
      color2 = '#ef4444'; // red
      minVal = Math.min(...data1, ...data2) * 0.9;
      maxVal = Math.max(...data1, ...data2) * 1.1;
    } else if (type === 'accuracy') {
      data1 = metrics.map(m => m.train_accuracy).filter(v => v !== null);
      data2 = metrics.map(m => m.val_accuracy).filter(v => v !== null);
      label1 = 'Train Acc';
      label2 = 'Val Acc';
      color1 = '#10b981'; // green
      color2 = '#f59e0b'; // orange
      minVal = Math.min(...data1, ...data2) * 0.95;
      maxVal = Math.min(Math.max(...data1, ...data2) * 1.05, 1.0);
    } else if (type === 'lr') {
      data1 = metrics.map(m => m.learning_rate).filter(v => v !== null);
      label1 = 'Learning Rate';
      label2 = null;
      color1 = '#8b5cf6'; // purple
      minVal = 0;
      maxVal = Math.max(...data1) * 1.1;
    }
    
    if (data1.length === 0) return;
    
    // 绘制网格
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }
    
    // 绘制坐标轴
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();
    
    // 绘制主曲线
    if (data1.length > 0) {
      ctx.strokeStyle = color1;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let dataIdx = 0;
      metrics.forEach((m, i) => {
        let val = type === 'loss' ? m.train_loss : type === 'accuracy' ? m.train_accuracy : m.learning_rate;
        if (val === null) return;
        const x = padding.left + (m.step / maxStep) * chartWidth;
        const y = padding.top + chartHeight - ((val - minVal) / (maxVal - minVal)) * chartHeight;
        if (dataIdx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        dataIdx++;
      });
      ctx.stroke();
    }
    
    // 绘制第二条曲线（验证集）
    if (data2 && data2.length > 0) {
      ctx.strokeStyle = color2;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      let prevVal = null;
      metrics.forEach((m, i) => {
        let val = type === 'loss' ? m.val_loss : m.val_accuracy;
        if (val === null) { prevVal = null; return; }
        const x = padding.left + (m.step / maxStep) * chartWidth;
        const y = padding.top + chartHeight - ((val - minVal) / (maxVal - minVal)) * chartHeight;
        if (prevVal === null) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        prevVal = val;
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Y轴标签
    ctx.fillStyle = '#6b7280';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const value = minVal + (maxVal - minVal) * (1 - i / 5);
      const y = padding.top + (chartHeight / 5) * i;
      let displayVal = value;
      if (type === 'lr') {
        displayVal = value.toExponential(2);
      } else if (type === 'accuracy') {
        displayVal = (value * 100).toFixed(0) + '%';
      } else {
        displayVal = value.toFixed(3);
      }
      ctx.fillText(displayVal, padding.left - 5, y + 3);
    }
    
    // X轴标签
    ctx.textAlign = 'center';
    ctx.fillText('0', padding.left, height - padding.bottom + 15);
    ctx.fillText(maxStep.toString(), width - padding.right, height - padding.bottom + 15);
    ctx.fillText('Steps', width / 2, height - 5);
    
    // 图例
    ctx.textAlign = 'left';
    ctx.fillStyle = color1;
    ctx.fillRect(width - padding.right + 10, padding.top, 15, 3);
    ctx.fillStyle = '#374151';
    ctx.font = '11px sans-serif';
    ctx.fillText(label1, width - padding.right + 30, padding.top + 4);
    
    if (label2) {
      ctx.strokeStyle = color2;
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(width - padding.right + 10, padding.top + 18);
      ctx.lineTo(width - padding.right + 25, padding.top + 18);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#374151';
      ctx.fillText(label2, width - padding.right + 30, padding.top + 22);
    }
    
  }, [metrics, width, height, type]);
  
  return React.createElement('canvas', { ref: canvasRef, className: 'w-full' });
}

// 混淆矩阵组件
function ConfusionMatrix({ matrix, labels }) {
  if (!matrix || !labels || matrix.length === 0) return null;
  
  const maxVal = Math.max(...matrix.flat());
  const cellSize = 40;
  const headerSize = 80;
  
  return (
    <div className="overflow-auto">
      <div className="text-sm font-medium text-gray-700 mb-2">混淆矩阵</div>
      <svg width={headerSize + labels.length * cellSize} height={headerSize + labels.length * cellSize}>
        {/* 标签 */}
        {labels.map((label, i) => (
          <g key={i}>
            <text x={headerSize - 10} y={headerSize + i * cellSize + cellSize / 2 + 4} 
                  textAnchor="end" fontSize="11" fill="#6b7280">{label}</text>
            <text x={headerSize + i * cellSize + cellSize / 2} y={headerSize - 10} 
                  textAnchor="middle" fontSize="11" fill="#6b7280" transform={`rotate(-30, ${headerSize + i * cellSize + cellSize / 2}, ${headerSize - 10})`}>{label}</text>
          </g>
        ))}
        
        {/* 矩阵单元格 */}
        {matrix.map((row, i) => 
          row.map((val, j) => {
            const intensity = val / maxVal;
            const color = `rgb(${59 + (37 - 59) * intensity}, ${130 + (99 - 130) * intensity}, ${246 + (247 - 246) * intensity})`;
            return (
              <g key={`${i}-${j}`}>
                <rect x={headerSize + j * cellSize} y={headerSize + i * cellSize} 
                      width={cellSize} height={cellSize} fill={color} stroke="white" strokeWidth="1"/>
                <text x={headerSize + j * cellSize + cellSize / 2} y={headerSize + i * cellSize + cellSize / 2 + 4}
                      textAnchor="middle" fontSize="12" fill={intensity > 0.5 ? 'white' : '#374151'}>{val}</text>
              </g>
            );
          })
        )}
      </svg>
    </div>
  );
}

// 训练任务详情弹窗
function TrainingDetailModal({ projectId, job, onClose }) {
  const [metrics, setMetrics] = useState([]);
  const [logs, setLogs] = useState([]);
  const [report, setReport] = useState(null);
  const [wsStatus, setWsStatus] = useState('disconnected');
  const wsRef = useRef(null);
  
  useEffect(() => {
    if (!job) return;
    
    // 加载历史指标和日志
    loadMetrics();
    loadLogs();
    
    // 如果是训练中的任务，建立WebSocket连接
    if (job.status === 'training') {
      connectWebSocket();
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [job]);
  
  const loadMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/jobs/${job.id}/metrics`);
      const data = await res.json();
      setMetrics(data.metrics || []);
    } catch (e) {}
    
    // 加载评估报告
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/jobs/${job.id}/report`);
      const data = await res.json();
      setReport(data);
    } catch (e) {}
  };
  
  const loadLogs = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/jobs/${job.id}/logs?lines=100`);
      const data = await res.json();
      setLogs(data.logs || []);
    } catch (e) {
      setLogs([]);
    }
  };
  
  const connectWebSocket = () => {
    const wsUrl = `ws://${window.location.host}/ai-training/ws/training/${job.id}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      setWsStatus('connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'training_update') {
        // 刷新指标
        loadMetrics();
      }
    };
    
    ws.onclose = () => {
      setWsStatus('disconnected');
    };
    
    wsRef.current = ws;
  };
  
  if (!job) return null;
  
  const evalData = report && report.evaluation;
  const classReport = evalData && evalData.classification_report;
  
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-auto">
        <div className="sticky top-0 bg-white border-b p-4 flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-lg">训练详情: {job.model_name}</h3>
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <span className={`px-2 py-0.5 rounded text-xs ${
                job.status === 'completed' ? 'bg-green-100 text-green-700' :
                job.status === 'training' ? 'bg-blue-100 text-blue-700' :
                'bg-red-100 text-red-700'
              }`}>
                {job.status === 'completed' ? '已完成' : job.status === 'training' ? '训练中' : '失败'}
              </span>
              {job.status === 'training' && (
                <span className={`text-xs ${wsStatus === 'connected' ? 'text-green-600' : 'text-gray-400'}`}>
                  ● WebSocket {wsStatus === 'connected' ? '已连接' : '未连接'}
                </span>
              )}
            </div>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <Icons.Close />
          </button>
        </div>
        
        <div className="p-4 space-y-6">
          {/* 训练概览卡片 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-blue-50 rounded-xl p-3">
              <div className="text-xs text-blue-600 mb-1">最佳准确率</div>
              <div className="text-xl font-bold text-blue-800">
                {job.best_accuracy ? (job.best_accuracy * 100).toFixed(2) + '%' : '-'}
              </div>
            </div>
            <div className="bg-green-50 rounded-xl p-3">
              <div className="text-xs text-green-600 mb-1">训练轮次</div>
              <div className="text-xl font-bold text-green-800">
                {job.current_epoch}/{job.total_epochs}
              </div>
            </div>
            <div className="bg-purple-50 rounded-xl p-3">
              <div className="text-xs text-purple-600 mb-1">验证Loss</div>
              <div className="text-xl font-bold text-purple-800">
                {job.best_val_loss ? job.best_val_loss.toFixed(4) : '-'}
              </div>
            </div>
            <div className="bg-orange-50 rounded-xl p-3">
              <div className="text-xs text-orange-600 mb-1">早停状态</div>
              <div className="text-xl font-bold text-orange-800">
                {job.early_stopped ? '已触发' : '未触发'}
              </div>
            </div>
          </div>
          
          {/* 训练曲线 */}
          {metrics.length > 0 && (
            <div className="bg-gray-50 rounded-xl p-4">
              <h4 className="font-medium text-gray-800 mb-3 flex items-center gap-2">
                <Icons.TrendingUp /> 训练曲线
              </h4>
              
              {/* Loss曲线 */}
              <div className="mb-4">
                <div className="text-xs text-gray-500 mb-2">Loss 变化（越低越好）</div>
                <TrainingChart metrics={metrics} width={700} height={150} type="loss" />
              </div>
              
              {/* Accuracy曲线 */}
              {metrics.some(m => m.train_accuracy !== null) && (
                <div className="mb-4">
                  <div className="text-xs text-gray-500 mb-2">Accuracy 变化（越高越好）</div>
                  <TrainingChart metrics={metrics} width={700} height={150} type="accuracy" />
                </div>
              )}
              
              {/* 学习率曲线 */}
              {metrics.some(m => m.learning_rate !== null) && (
                <div className="mb-2">
                  <div className="text-xs text-gray-500 mb-2">学习率变化（观察调度策略）</div>
                  <TrainingChart metrics={metrics} width={700} height={120} type="lr" />
                </div>
              )}
              
              <div className="text-xs text-gray-500 mt-2">
                数据点数: {metrics.length} | 实时更新: {wsStatus === 'connected' ? '✓' : '✗'}
              </div>
            </div>
          )}
          
          {/* 评估报告 */}
          {evalData && classReport && (
            <div className="bg-gray-50 rounded-xl p-4">
              <h4 className="font-medium text-gray-800 mb-3 flex items-center gap-2">
                <Icons.Award /> 评估报告
              </h4>
              
              {/* 总体指标 */}
              <div className="grid grid-cols-4 gap-2 mb-4">
                {['accuracy', 'precision', 'recall', 'f1-score'].map(metric => {
                  const avgKey = 'weighted avg';
                  const value = classReport[avgKey] && classReport[avgKey][metric === 'accuracy' ? 'precision' : metric];
                  return (
                    <div key={metric} className="bg-white rounded-lg p-2 text-center">
                      <div className="text-xs text-gray-500 capitalize">{metric}</div>
                      <div className="text-lg font-bold text-gray-800">{(value * 100).toFixed(1)}%</div>
                    </div>
                  );
                })}
              </div>
              
              {/* 混淆矩阵 */}
              {evalData.confusion_matrix && (
                <ConfusionMatrix 
                  matrix={evalData.confusion_matrix} 
                  labels={Object.keys(classReport).filter(k => !['accuracy', 'macro avg', 'weighted avg'].includes(k))}
                />
              )}
              
              {/* 各类别详细指标 */}
              <div className="mt-4 overflow-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-100">
                    <tr>
                      <th className="px-3 py-2 text-left">类别</th>
                      <th className="px-3 py-2 text-right">Precision</th>
                      <th className="px-3 py-2 text-right">Recall</th>
                      <th className="px-3 py-2 text-right">F1-Score</th>
                      <th className="px-3 py-2 text-right">Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(classReport)
                      .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
                      .map(([label, stats]) => (
                        <tr key={label} className="border-b">
                          <td className="px-3 py-2 font-medium">{label}</td>
                          <td className="px-3 py-2 text-right">{(stats.precision * 100).toFixed(1)}%</td>
                          <td className="px-3 py-2 text-right">{(stats.recall * 100).toFixed(1)}%</td>
                          <td className="px-3 py-2 text-right">{(stats['f1-score'] * 100).toFixed(1)}%</td>
                          <td className="px-3 py-2 text-right">{stats.support}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          
          {/* 训练日志 */}
          <div className="bg-gray-900 rounded-xl p-4">
            <h4 className="font-medium text-gray-300 mb-2 flex items-center gap-2">
              <Icons.Activity /> 训练日志
            </h4>
            <div className="text-xs text-gray-400 font-mono space-y-1 max-h-60 overflow-auto">
              {logs.length > 0 ? (
                logs.map((line, i) => (
                  <div key={i} className="truncate">{line}</div>
                ))
              ) : (
                <div className="text-gray-600">暂无日志</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// 内存监控组件
function MemoryMonitor() {
  const [memoryStatus, setMemoryStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  const loadMemoryStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/system/memory`);
      const data = await res.json();
      setMemoryStatus(data);
    } catch (e) {}
  };

  const cleanupMemory = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/inference/cleanup`, { method: 'POST' });
      const data = await res.json();
      alert(`已清理 ${data.models_unloaded} 个模型，释放 ${data.memory_freed_gb.toFixed(2)} GB 内存`);
      loadMemoryStatus();
    } catch (e) {
      alert('清理失败');
    }
    setLoading(false);
  };

  useEffect(() => {
    loadMemoryStatus();
    // 每30秒刷新一次
    const interval = setInterval(loadMemoryStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  if (!memoryStatus) return null;

  const isHighMemory = memoryStatus.memory_percent > 80;
  const isCritical = memoryStatus.memory_percent > 90;

  return (
    <div className={`mb-6 p-4 rounded-xl ${
      isCritical ? 'bg-red-50 border border-red-200' :
      isHighMemory ? 'bg-orange-50 border border-orange-200' :
      'bg-blue-50 border border-blue-200'
    }`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icons.Memory />
          <span className="font-medium text-gray-800">内存状态</span>
          {isCritical && <span className="px-2 py-0.5 bg-red-500 text-white rounded text-xs">危险</span>}
          {isHighMemory && !isCritical && <span className="px-2 py-0.5 bg-orange-500 text-white rounded text-xs">告警</span>}
        </div>
        <button
          onClick={loadMemoryStatus}
          className="text-xs text-gray-500 hover:text-gray-700"
        >
          刷新
        </button>
      </div>

      {/* 内存使用进度条 */}
      <div className="mb-3">
        <div className="flex justify-between text-sm text-gray-600 mb-1">
          <span>已使用 {memoryStatus.memory_percent.toFixed(1)}%</span>
          <span>{memoryStatus.memory_used_gb.toFixed(1)} / {memoryStatus.memory_total_gb.toFixed(1)} GB</span>
        </div>
        <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              isCritical ? 'bg-red-500' :
              isHighMemory ? 'bg-orange-500' :
              'bg-green-500'
            }`}
            style={{ width: `${Math.min(memoryStatus.memory_percent, 100)}%` }}
          />
        </div>
      </div>

      {/* 已加载模型 */}
      <div className="text-sm text-gray-600 mb-3">
        已加载模型: <span className="font-medium">{memoryStatus.loaded_models_count}</span> / {memoryStatus.max_models_limit} 个
        {memoryStatus.models.length > 0 && (
          <div className="mt-2 space-y-1">
            {memoryStatus.models.map(m => (
              <div key={m.model_id} className="flex items-center justify-between text-xs bg-white/50 px-2 py-1 rounded">
                <span className="truncate max-w-[150px]">{m.model_id}</span>
                <span className="text-gray-500">
                  {m.memory_size_mb}MB · 访问{m.access_count}次 · 空闲{m.idle_seconds}s
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 自动清理说明 */}
      <div className="text-xs text-gray-500 mb-3">
        💡 自动管理: 内存&gt;80%触发清理 · 30分钟空闲自动卸载 · 最多保留5个模型
      </div>

      {/* 手动清理按钮 */}
      <button
        onClick={cleanupMemory}
        disabled={loading || memoryStatus.loaded_models_count === 0}
        className="w-full py-2 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 disabled:opacity-50"
      >
        {loading ? <Icons.Loading /> : '🧹 清理所有模型释放内存'}
      </button>
    </div>
  );
}

// 飞书通知配置Tab
function NotificationTab({ projectId }) {
  const [config, setConfig] = useState({
    enabled: false,
    webhook_url: '',
    notify_on_success: true,
    notify_on_failure: true
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testSending, setTestSending] = useState(false);

  useEffect(() => {
    loadConfig();
  }, [projectId]);

  const loadConfig = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/feishu-notification`);
      const data = await res.json();
      if (!res.ok) {
        console.error('加载配置失败:', data.detail || data.error || `请求失败 (${res.status})`);
        setLoading(false);
        return;
      }
      if (data.enabled) {
        setConfig({
          enabled: true,
          webhook_url: data.webhook_url,
          notify_on_success: data.notify_on_success,
          notify_on_failure: data.notify_on_failure
        });
      }
    } catch (e) {
      console.error('加载配置失败:', e);
    }
    setLoading(false);
  };

  const saveConfig = async () => {
    setSaving(true);
    try {
      const formData = new FormData();
      formData.append('webhook_url', config.webhook_url);
      formData.append('notify_on_success', config.notify_on_success);
      formData.append('notify_on_failure', config.notify_on_failure);

      const res = await fetch(`${API_BASE}/projects/${projectId}/feishu-notification`, {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) {
        const data = await res.json();
        alert('保存失败: ' + (data.detail || data.error || `请求失败 (${res.status})`));
        setSaving(false);
        return;
      }
      
      alert('配置已保存');
      loadConfig();
    } catch (e) {
      alert('保存失败: ' + e.message);
    }
    setSaving(false);
  };

  const deleteConfig = async () => {
    if (!confirm('确定要删除飞书通知配置吗？')) return;
    
    try {
      await fetch(`${API_BASE}/projects/${projectId}/feishu-notification`, {
        method: 'DELETE'
      });
      
      setConfig({
        enabled: false,
        webhook_url: '',
        notify_on_success: true,
        notify_on_failure: true
      });
      
      alert('配置已删除');
    } catch (e) {
      alert('删除失败');
    }
  };

  const testNotification = async () => {
    setTestSending(true);
    try {
      // 发送测试消息到飞书
      const message = {
        msg_type: "text",
        content: {
          text: `🧪 测试消息\n来自 AI模型训练工坊\n项目: ${projectId}\n时间: ${new Date().toLocaleString()}`
        }
      };
      
      const res = await fetch(config.webhook_url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(message)
      });
      
      if (res.ok) {
        alert('测试消息已发送，请检查飞书');
      } else {
        alert('发送失败，请检查Webhook地址');
      }
    } catch (e) {
      alert('发送失败: ' + e.message);
    }
    setTestSending(false);
  };

  return (
    <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
      <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
        <Icons.Bell /> 飞书通知配置
      </h3>

      {loading ? (
        <div className="text-center py-8 text-gray-500">
          <Icons.Loading />
          <p className="mt-2">加载中...</p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* 启用开关 */}
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
            <div>
              <div className="font-medium text-gray-800">启用飞书通知</div>
              <div className="text-sm text-gray-500">训练完成后自动发送消息到飞书</div>
            </div>
            <button
              onClick={() => setConfig({...config, enabled: !config.enabled})}
              className={`w-14 h-8 rounded-full transition-all relative ${
                config.enabled ? 'bg-purple-500' : 'bg-gray-300'
              }`}
            >
              <span className={`absolute top-1 w-6 h-6 bg-white rounded-full transition-all ${
                config.enabled ? 'left-7' : 'left-1'
              }`} />
            </button>
          </div>

          {config.enabled && (
            <React.Fragment>
              {/* Webhook地址 */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  飞书 Webhook 地址
                </label>
                <input
                  type="text"
                  value={config.webhook_url}
                  onChange={(e) => setConfig({...config, webhook_url: e.target.value})}
                  placeholder="https://open.feishu.cn/open-apis/bot/v2/hook/xxxx"
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none text-sm"
                />
                <p className="text-xs text-gray-500 mt-1">
                  在飞书群设置 → 群机器人 → 添加自定义机器人获取
                </p>
              </div>

              {/* 通知选项 */}
              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.notify_on_success}
                    onChange={(e) => setConfig({...config, notify_on_success: e.target.checked})}
                    className="w-4 h-4"
                  />
                  <span className="text-sm text-gray-700">训练成功时通知</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.notify_on_failure}
                    onChange={(e) => setConfig({...config, notify_on_failure: e.target.checked})}
                    className="w-4 h-4"
                  />
                  <span className="text-sm text-gray-700">训练失败时通知</span>
                </label>
              </div>

              {/* 操作按钮 */}
              <div className="flex gap-3 pt-4">
                <button
                  onClick={saveConfig}
                  disabled={saving || !config.webhook_url}
                  className="flex-1 py-2 bg-purple-600 text-white rounded-lg font-medium disabled:opacity-50"
                >
                  {saving ? <Icons.Loading /> : '保存配置'}
                </button>
                {config.enabled && (
                  <React.Fragment>
                    <button
                      onClick={testNotification}
                      disabled={testSending || !config.webhook_url}
                      className="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg font-medium disabled:opacity-50"
                    >
                      {testSending ? <Icons.Loading /> : '发送测试'}
                    </button>
                    <button
                      onClick={deleteConfig}
                      className="px-4 py-2 bg-red-100 text-red-700 rounded-lg font-medium"
                    >
                      <Icons.Trash />
                    </button>
                  </React.Fragment>
                )}
              </div>
            </React.Fragment>
          )}

          {/* 使用说明 */}
          <div className="mt-6 p-4 bg-blue-50 rounded-xl">
            <h4 className="font-medium text-blue-800 mb-2">📖 使用说明</h4>
            <ol className="text-sm text-blue-700 space-y-1 list-decimal list-inside">
              <li>在飞书群中添加自定义机器人</li>
              <li>复制机器人的 Webhook 地址</li>
              <li>粘贴到上方输入框并保存</li>
              <li>训练完成后会自动收到通知</li>
            </ol>
          </div>
        </div>
      )}
    </div>
  );
}

// 时序分析Tab
function TimeSeriesTab({ projectId, datasets }) {
  const [selectedDataset, setSelectedDataset] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [rulResult, setRulResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [rulLoading, setRulLoading] = useState(false);
  const [forecastHours, setForecastHours] = useState(24);
  const [degradationCol, setDegradationCol] = useState('');
  const [threshold, setThreshold] = useState(0.85);
  const [datasetColumns, setDatasetColumns] = useState([]);

  // 加载数据集列信息
  const loadDatasetColumns = async (datasetId) => {
    if (!datasetId) return;
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/${datasetId}/preview?limit=5`);
      const data = await res.json();
      setDatasetColumns(data.columns || []);
    } catch (e) {}
  };

  const runAnalysis = async () => {
    if (!selectedDataset) {
      alert('请先选择数据集');
      return;
    }
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('dataset_id', selectedDataset);
      formData.append('forecast_hours', forecastHours);
      
      const res = await fetch(`${API_BASE}/projects/${projectId}/time-series/analyze`, {
        method: 'POST',
        body: formData
      });
      
      const data = await res.json();
      if (!res.ok) {
        // HTTP错误状态码 (400, 500等)
        const errorMsg = data.detail || data.error || `请求失败 (${res.status})`;
        alert('分析失败: ' + errorMsg);
        setLoading(false);
        return;
      }
      if (data.success) {
        setAnalysisResult(data.analysis);
      } else if (data.error) {
        alert('分析失败: ' + data.error);
      }
    } catch (e) {
      alert('分析失败: ' + e.message);
    }
    setLoading(false);
  };

  const runRULPrediction = async () => {
    if (!selectedDataset) {
      alert('请先选择数据集');
      return;
    }
    if (!degradationCol) {
      alert('请选择劣化指标列');
      return;
    }
    
    setRulLoading(true);
    try {
      const formData = new FormData();
      formData.append('dataset_id', selectedDataset);
      formData.append('degradation_col', degradationCol);
      formData.append('threshold', threshold);
      
      const res = await fetch(`${API_BASE}/projects/${projectId}/time-series/rul`, {
        method: 'POST',
        body: formData
      });
      
      const data = await res.json();
      if (!res.ok) {
        // HTTP错误状态码 (400, 500等)
        const errorMsg = data.detail || data.error || `请求失败 (${res.status})`;
        alert('RUL预测失败: ' + errorMsg);
        setRulLoading(false);
        return;
      }
      if (data.success) {
        setRulResult(data.rul);
      } else if (data.error) {
        alert('RUL预测失败: ' + data.error);
      }
    } catch (e) {
      alert('RUL预测失败: ' + e.message);
    }
    setRulLoading(false);
  };

  return (
    <div className="space-y-4">
      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Icons.Chart /> 时序趋势分析
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">选择数据集</label>
            <select
              value={selectedDataset}
              onChange={(e) => {
                setSelectedDataset(e.target.value);
                loadDatasetColumns(e.target.value);
              }}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
            >
              <option value="">请选择</option>
              {(datasets || []).map(ds => (
                <option key={ds.id} value={ds.id}>{ds.name}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">预测时长(小时)</label>
            <select
              value={forecastHours}
              onChange={(e) => setForecastHours(parseInt(e.target.value))}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
            >
              <option value={12}>12小时</option>
              <option value={24}>24小时</option>
              <option value={48}>48小时</option>
              <option value={72}>72小时</option>
            </select>
          </div>
          
          <div className="flex items-end">
            <button
              onClick={runAnalysis}
              disabled={loading}
              className="w-full py-2 bg-purple-600 text-white rounded-lg font-medium disabled:opacity-50"
            >
              {loading ? <Icons.Loading /> : '开始分析'}
            </button>
          </div>
        </div>
      </div>

      {analysisResult && (
        <React.Fragment>
          {/* 趋势预测结果 */}
          <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
            <h3 className="font-semibold text-gray-800 mb-4">趋势预测</h3>
            
            {Object.entries(analysisResult.trend_analysis || {}).map(([metric, result]) => (
              <div key={metric} className="mb-6 p-4 bg-gray-50 rounded-xl">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium text-gray-700">{metric}</h4>
                  <span className={`px-2 py-1 rounded text-xs ${
                    result.risk_level === 'high' ? 'bg-red-100 text-red-700' :
                    result.risk_level === 'medium' ? 'bg-orange-100 text-orange-700' :
                    'bg-green-100 text-green-700'
                  }`}>
                    {result.trend_direction} · 风险{result.risk_level === 'high' ? '高' : result.risk_level === 'medium' ? '中' : '低'}
                  </span>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                  <div className="bg-white p-3 rounded-lg">
                    <div className="text-xs text-gray-500">当前值</div>
                    <div className="text-lg font-bold text-gray-800">{result.current_value}</div>
                  </div>
                  <div className="bg-white p-3 rounded-lg">
                    <div className="text-xs text-gray-500">平均值</div>
                    <div className="text-lg font-bold text-gray-800">{result.mean_value}</div>
                  </div>
                  <div className="bg-white p-3 rounded-lg">
                    <div className="text-xs text-gray-500">变化趋势</div>
                    <div className="text-lg font-bold text-gray-800">{result.value_change_percent > 0 ? '+' : ''}{result.value_change_percent}%</div>
                  </div>
                  <div className="bg-white p-3 rounded-lg">
                    <div className="text-xs text-gray-500">预测超标</div>
                    <div className="text-lg font-bold text-gray-800">{result.will_exceed_threshold ? '是' : '否'}</div>
                  </div>
                </div>
                
                {result.exceed_time && (
                  <div className="p-3 bg-red-50 rounded-lg text-red-700 text-sm">
                    ⚠️ 预计将在 <strong>{result.exceed_time}</strong> 超出阈值
                  </div>
                )}
                
                {/* 未来预测值 */}
                <div className="mt-3">
                  <div className="text-xs text-gray-500 mb-2">未来{result.forecast_hours}小时预测:</div>
                  <div className="flex gap-2 overflow-x-auto pb-2">
                    {result.future_predictions && result.future_predictions.map((p, i) => (
                      <div key={i} className="flex-shrink-0 bg-white px-3 py-2 rounded-lg text-center min-w-[80px]">
                        <div className="text-xs text-gray-500">{p.timestamp}</div>
                        <div className="font-bold text-gray-800">{p.predicted_value}</div>
                      </div>
                    ))}
                    {(!result.future_predictions || result.future_predictions.length === 0) && (
                      <div className="text-sm text-gray-500">暂无预测数据</div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* 异常模式检测 */}
          <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
            <h3 className="font-semibold text-gray-800 mb-4">异常模式检测</h3>
            
            <div className="p-4 bg-blue-50 rounded-lg mb-4">
              <p className="text-blue-800">{analysisResult.anomaly_patterns && anomaly_patterns.summary || '分析中...'}</p>
            </div>
            
            {analysisResult.anomaly_patterns && (anomaly_patterns.patterns) && anomaly_patterns.patterns.map((pattern, i) => (
              <div key={i} className={`mb-3 p-4 rounded-lg ${
                pattern.severity === 'high' ? 'bg-red-50 border border-red-200' :
                pattern.severity === 'medium' ? 'bg-orange-50 border border-orange-200' :
                'bg-gray-50'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-800">{pattern.pattern_type}</span>
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    pattern.severity === 'high' ? 'bg-red-100 text-red-700' :
                    pattern.severity === 'medium' ? 'bg-orange-100 text-orange-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {pattern.severity === 'high' ? '高风险' : pattern.severity === 'medium' ? '中风险' : '提示'}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mb-2">{pattern.description}</p>
                <p className="text-xs text-gray-500">💡 建议: {pattern.recommendation}</p>
              </div>
            ))}
          </div>

          {/* RUL 剩余使用寿命预测 */}
          <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
            <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
              <Icons.Clock /> 剩余使用寿命 (RUL) 预测
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">劣化指标列</label>
                <select
                  value={degradationCol}
                  onChange={(e) => setDegradationCol(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
                >
                  <option value="">请选择</option>
                  {(datasetColumns || []).map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-1">表示设备劣化程度的指标</p>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">故障阈值</label>
                <input
                  type="number"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
                  placeholder="例如: 0.85"
                />
                <p className="text-xs text-gray-500 mt-1">达到此值视为故障</p>
              </div>
              
              <div className="flex items-end">
                <button
                  onClick={runRULPrediction}
                  disabled={rulLoading || !selectedDataset}
                  className="w-full py-2 bg-gradient-to-r from-orange-500 to-red-500 text-white rounded-lg font-medium disabled:opacity-50"
                >
                  {rulLoading ? <Icons.Loading /> : '预测RUL'}
                </button>
              </div>
            </div>
            
            {rulResult && (
              <div className={`p-6 rounded-xl ${
                rulResult.status === 'critical' ? 'bg-red-50 border-2 border-red-200' :
                rulResult.status === 'warning' ? 'bg-orange-50 border-2 border-orange-200' :
                'bg-green-50 border-2 border-green-200'
              }`}>
                <div className="flex items-center gap-3 mb-4">
                  <span className="text-3xl">
                    {rulResult.status === 'critical' ? '🔴' : 
                     rulResult.status === 'warning' ? '🟡' : '🟢'}
                  </span>
                  <div>
                    <div className="font-bold text-lg text-gray-800">
                      {rulResult.status === 'critical' ? '紧急 - 立即检修' : 
                       rulResult.status === 'warning' ? '警告 - 计划维护' : '正常 - 持续监控'}
                    </div>
                    <div className="text-sm text-gray-600">{rulResult.message}</div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-white p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-gray-800">{rulResult.rul_hours || '-'}</div>
                    <div className="text-xs text-gray-500">剩余小时数</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-gray-800">{rulResult.rul_days || '-'}</div>
                    <div className="text-xs text-gray-500">剩余天数</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-gray-800">{rulResult.current_value || '-'}</div>
                    <div className="text-xs text-gray-500">当前劣化值</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-gray-800">{rulResult.threshold || '-'}</div>
                    <div className="text-xs text-gray-500">故障阈值</div>
                  </div>
                </div>
                
                {rulResult.estimated_failure_time && (
                  <div className="mt-4 p-3 bg-white rounded-lg text-center">
                    <span className="text-gray-600">预计故障时间: </span>
                    <span className="font-bold text-red-600">{rulResult.estimated_failure_time}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </React.Fragment>
      )}
    </div>
  );
}

// 在线学习Tab
function OnlineLearningTab({ projectId, jobs, datasets }) {
  const [selectedJob, setSelectedJob] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [learning, setLearning] = useState(false);
  const [history, setHistory] = useState([]);
  const [autoConfig, setAutoConfig] = useState({ schedule: 'daily', min_samples: 100 });

  const startOnlineLearning = async () => {
    if (!selectedJob || !selectedDataset) {
      alert('请选择模型和数据集');
      return;
    }
    
    setLearning(true);
    try {
      const formData = new FormData();
      formData.append('dataset_id', selectedDataset);
      formData.append('learning_type', 'incremental');
      
      const res = await fetch(`${API_BASE}/projects/${projectId}/models/${selectedJob}/online-learn`, {
        method: 'POST',
        body: formData
      });
      
      const data = await res.json();
      if (!res.ok) {
        alert('在线学习失败: ' + (data.detail || data.error || `请求失败 (${res.status})`));
        setLearning(false);
        return;
      }
      if (data.success) {
        alert(`增量学习完成！新增样本: ${data.result.new_samples}, 准确率: ${(data.result.accuracy * 100).toFixed(2)}%`);
        loadHistory();
      } else if (data.error) {
        alert('在线学习失败: ' + data.error);
      }
    } catch (e) {
      alert('在线学习失败: ' + e.message);
    }
    setLearning(false);
  };

  const loadHistory = async () => {
    if (!selectedJob) return;
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/models/${selectedJob}/learning-history`);
      const data = await res.json();
      if (!res.ok) {
        console.error('获取学习历史失败:', data.detail || data.error || `请求失败 (${res.status})`);
        return;
      }
      setHistory(data.history || []);
    } catch (e) {
      console.error('获取学习历史失败:', e);
    }
  };

  const setupAutoLearning = async () => {
    if (!selectedJob) {
      alert('请先选择模型');
      return;
    }
    
    try {
      const formData = new FormData();
      formData.append('schedule', autoConfig.schedule);
      formData.append('min_samples', autoConfig.min_samples);
      
      const res = await fetch(`${API_BASE}/projects/${projectId}/models/${selectedJob}/auto-learn`, {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) {
        const data = await res.json();
        alert('配置失败: ' + (data.detail || data.error || `请求失败 (${res.status})`));
        return;
      }
      
      alert('自动学习配置已保存');
    } catch (e) {
      alert('配置失败: ' + e.message);
    }
  };

  useEffect(() => {
    loadHistory();
  }, [selectedJob]);

  const completedJobs = jobs.filter(j => j.status === 'completed');

  return (
    <div className="space-y-4">
      {/* 增量学习 */}
      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Icons.Brain /> 增量学习
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">选择基础模型</label>
            <select
              value={selectedJob}
              onChange={(e) => setSelectedJob(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
            >
              <option value="">请选择</option>
              {completedJobs.map(job => (
                <option key={job.id} value={job.id}>{job.model_name} ({job.best_accuracy ? (job.best_accuracy * 100).toFixed(1) + '%' : '-'})</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">选择新数据</label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
            >
              <option value="">请选择</option>
              {datasets.map(ds => (
                <option key={ds.id} value={ds.id}>{ds.name}</option>
              ))}
            </select>
          </div>
        </div>
        
        <div className="p-4 bg-blue-50 rounded-lg mb-4">
          <h4 className="font-medium text-blue-800 mb-2">💡 增量学习说明</h4>
          <ul className="text-sm text-blue-700 space-y-1 list-disc list-inside">
            <li>用新数据继续训练现有模型，保留已有知识</li>
            <li>比全量重训练更快，适合日常更新</li>
            <li>自动评估新旧模型，保留更好的版本</li>
          </ul>
        </div>
        
        <button
          onClick={startOnlineLearning}
          disabled={learning || !selectedJob || !selectedDataset}
          className="w-full py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-semibold disabled:opacity-50"
        >
          {learning ? <Icons.Loading /> : '🚀 开始增量学习'}
        </button>
      </div>

      {/* 自动学习配置 */}
      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4">自动学习配置</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">学习频率</label>
            <select
              value={autoConfig.schedule}
              onChange={(e) => setAutoConfig({...autoConfig, schedule: e.target.value})}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
            >
              <option value="daily">每天</option>
              <option value="weekly">每周</option>
              <option value="never">关闭</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">最小新样本数</label>
            <input
              type="number"
              value={autoConfig.min_samples}
              onChange={(e) => setAutoConfig({...autoConfig, min_samples: parseInt(e.target.value)})}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
            />
          </div>
          
          <div className="flex items-end">
            <button
              onClick={setupAutoLearning}
              disabled={!selectedJob}
              className="w-full py-2 bg-cyan-500 text-white rounded-lg font-medium disabled:opacity-50"
            >
              保存配置
            </button>
          </div>
        </div>
      </div>

      {/* 学习历史 */}
      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4">学习历史</h3>
        
        {history.length === 0 ? (
          <p className="text-gray-500 text-center py-4">暂无学习记录</p>
        ) : (
          <div className="space-y-2">
            {history.map((h, i) => (
              <div key={i} className="p-3 bg-gray-50 rounded-lg flex items-center justify-between">
                <div>
                  <div className="font-medium text-gray-800">{h.learning_type === 'incremental' ? '增量学习' : '全量重训'}</div>
                  <div className="text-sm text-gray-500">
                    {new Date(h.created_at).toLocaleString()} · 
                    新增样本: {h.new_samples}
                  </div>
                </div>
                <div className="text-right">
                  <div className={`font-medium ${
                    h.accuracy_after > (h.accuracy_before || 0) ? 'text-green-600' : 'text-orange-600'
                  }`}>
                    {h.accuracy_after ? (h.accuracy_after * 100).toFixed(1) + '%' : '-'}
                  </div>
                  <div className="text-xs text-gray-500">{h.status === 'completed' ? '完成' : h.status}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// 渲染应用
// XAI 可解释性 Tab
function XAITab({ projectId, jobs, projectType }) {
  const [selectedJob, setSelectedJob] = useState('');
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  
  const completedJobs = jobs.filter(j => j.status === 'completed');
  
  // 获取特征重要性（表格数据）
  const loadFeatureImportance = async () => {
    if (!selectedJob) return;
    
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/jobs/${selectedJob}/xai/feature-importance`);
      const data = await res.json();
      if (!res.ok) {
        console.error('获取特征重要性失败:', data.detail || data.error || `请求失败 (${res.status})`);
        return;
      }
      if (data.success) {
        setFeatureImportance(data.feature_importance);
      } else if (data.error) {
        console.error('获取特征重要性失败:', data.error);
      }
    } catch (e) {
      console.error('获取特征重要性失败:', e);
    }
  };
  
  // 图像Grad-CAM解释
  const explainImage = async () => {
    if (!selectedJob || !selectedImage) {
      alert('请选择模型和图片');
      return;
    }
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('project_id', projectId);
      formData.append('job_id', selectedJob);
      formData.append('image', selectedImage);
      
      const res = await fetch(`${API_BASE}/xai/image/gradcam`, {
        method: 'POST',
        body: formData
      });
      
      const data = await res.json();
      if (!res.ok) {
        alert('解释失败: ' + (data.detail || data.error || `请求失败 (${res.status})`));
        setLoading(false);
        return;
      }
      if (data.success) {
        setExplanation(data);
      } else if (data.error) {
        alert('解释失败: ' + data.error);
      }
    } catch (e) {
      alert('解释失败: ' + e.message);
    }
    setLoading(false);
  };
  
  useEffect(() => {
    if (selectedJob && !isImage) {
      loadFeatureImportance();
    }
  }, [selectedJob]);
  
  const isImage = projectType === 'image_classification' || projectType === 'object_detection';
  const isText = projectType === 'text_classification';
  const isTabular = ['classification', 'regression', 'anomaly_detection', 'time_series'].includes(projectType);
  
  return (
    <div className="space-y-4">
      {/* 模型选择 */}
      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Icons.Help /> 模型可解释性 (XAI)
        </h3>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">选择模型</label>
          <select
            value={selectedJob}
            onChange={(e) => {
              setSelectedJob(e.target.value);
              setExplanation(null);
              setFeatureImportance(null);
            }}
            className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
          >
            <option value="">请选择训练好的模型</option>
            {completedJobs.map(job => (
              <option key={job.id} value={job.id}>
                {job.model_name || '未命名模型'} ({job.best_accuracy ? (job.best_accuracy * 100).toFixed(1) + '%' : '-'})
              </option>
            ))}
          </select>
        </div>
        
        {selectedJob && (
          <div className="p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border border-purple-200">
            <h4 className="font-medium text-purple-800 mb-2">
              {isImage ? '🖼️ 图像可解释性 (Grad-CAM)' : 
               isText ? '📝 文本可解释性 (Attention)' : 
               '📊 特征重要性分析'}
            </h4>
            <p className="text-sm text-purple-700 mb-4">
              {isImage ? 'Grad-CAM可以高亮显示模型关注的区域，帮助你理解模型为什么这样预测。' :
               isText ? 'Attention可视化可以展示模型关注的词语。' :
               '特征重要性可以告诉你哪些输入特征对预测结果影响最大。'}
            </p>
          </div>
        )}
      </div>
      
      {/* 图像解释 */}
      {isImage && selectedJob && (
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
          <h3 className="font-semibold text-gray-800 mb-4">上传图片进行解释</h3>
          
          <div className="mb-4">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setSelectedImage(e.target.files[0])}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white"
            />
          </div>
          
          <button
            onClick={explainImage}
            disabled={loading || !selectedImage}
            className="w-full py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-semibold disabled:opacity-50"
          >
            {loading ? <Icons.Loading /> : '🔍 生成Grad-CAM解释'}
          </button>
          
          {explanation && (
            <div className="mt-6 space-y-4">
              {/* 预测结果 */}
              <div className="p-4 bg-gray-50 rounded-xl">
                <div className="font-medium text-gray-800 mb-1">预测结果</div>
                <div className="text-lg">
                  <span className="font-bold text-purple-600">{explanation.prediction.class_name}</span>
                  <span className="text-gray-500 ml-2">({(explanation.prediction.confidence * 100).toFixed(1)}%)</span>
                </div>
              </div>
              
              {/* 图片对比 */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-2">原图</div>
                  <img src={explanation.original} alt="原始图片" className="w-full rounded-lg border" />
                </div>
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-2">Grad-CAM热力图</div>
                  <img src={explanation.heatmap} alt="热力图" className="w-full rounded-lg border" />
                </div>
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-2">叠加效果</div>
                  <img src={explanation.overlay} alt="叠加图" className="w-full rounded-lg border" />
                </div>
              </div>
              
              {/* 说明 */}
              <div className="p-4 bg-blue-50 rounded-xl text-sm text-blue-800">
                <strong>💡 如何解读：</strong><br/>
                红色/黄色区域表示模型关注的重点。模型主要根据这些区域做出预测。
                如果关注区域与缺陷位置一致，说明模型学对了；如果不一致，可能需要更多训练数据。
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* 表格数据特征重要性 */}
      {isTabular && featureImportance && (
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
          <h3 className="font-semibold text-gray-800 mb-4">特征重要性排行</h3>
          
          <div className="space-y-2">
            {featureImportance.slice(0, 10).map((item, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="w-8 text-center text-gray-500 font-medium">{i + 1}</div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-gray-800">{item.feature}</span>
                    <span className="text-sm text-gray-600">{(item.importance * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                      style={{width: `${item.importance * 100}%`}}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-4 p-4 bg-blue-50 rounded-xl text-sm text-blue-800">
            <strong>💡 解读：</strong><br/>
            重要性越高的特征对模型预测影响越大。可以用于特征筛选，去除低重要性特征可能提升模型性能和推理速度。
          </div>
        </div>
      )}
    </div>
  );
}

// 告警管理 Tab - 简化版
function AlertTab({ projectId }) {
  const [rules, setRules] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [templates, setTemplates] = useState([]);
  const [showCreate, setShowCreate] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [loading, setLoading] = useState(false);

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    rule_type: 'training',
    condition_field: 'best_accuracy',
    condition_operator: '<',
    condition_value: 0.8,
    severity: 'warning',
    cooldown_minutes: 60
  });

  useEffect(() => {
    loadRules();
    loadAlerts();
    loadTemplates();
  }, [projectId]);

  const loadRules = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/alert-rules`);
      const data = await res.json();
      setRules(data.rules || []);
    } catch (e) {}
  };

  const loadAlerts = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/alerts?limit=20`);
      const data = await res.json();
      setAlerts(data.alerts || []);
    } catch (e) {}
  };

  const loadTemplates = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/alert-templates`);
      const data = await res.json();
      setTemplates(data.templates || []);
    } catch (e) {}
  };

  const createRule = async () => {
    setLoading(true);
    try {
      const form = new FormData();
      Object.keys(formData).forEach(key => form.append(key, formData[key]));
      const res = await fetch(`${API_BASE}/projects/${projectId}/alert-rules`, {
        method: 'POST',
        body: form
      });
      if (res.ok) {
        setShowCreate(false);
        loadRules();
        alert('告警规则创建成功');
      }
    } catch (e) {
      alert('创建失败: ' + e.message);
    }
    setLoading(false);
  };

  const createFromTemplate = async () => {
    if (!selectedTemplate) return;
    setLoading(true);
    try {
      const form = new FormData();
      form.append('template_key', selectedTemplate);
      const res = await fetch(`${API_BASE}/projects/${projectId}/alert-rules/template`, {
        method: 'POST',
        body: form
      });
      if (res.ok) {
        setShowCreate(false);
        loadRules();
        alert('从模板创建成功');
      }
    } catch (e) {
      alert('创建失败: ' + e.message);
    }
    setLoading(false);
  };

  const deleteRule = async (ruleId) => {
    if (!confirm('确定删除此告警规则？')) return;
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/alert-rules/${ruleId}`, {
        method: 'DELETE'
      });
      if (res.ok) loadRules();
    } catch (e) {}
  };

  const acknowledgeAlert = async (alertId) => {
    try {
      await fetch(`${API_BASE}/projects/${projectId}/alerts/${alertId}/acknowledge`, {
        method: 'POST'
      });
      loadAlerts();
    } catch (e) {}
  };

  const getSeverityColor = (s) => ({info:'bg-blue-100 text-blue-800',warning:'bg-orange-100 text-orange-800',error:'bg-red-100 text-red-800',critical:'bg-red-200 text-red-900'}[s]||'bg-gray-100 text-gray-800');
  const getSeverityEmoji = (s) => ({info:'ℹ️',warning:'⚠️',error:'❌',critical:'🔴'}[s]||'⚠️');

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-gray-200/50 shadow-lg">
          <div className="text-2xl font-bold text-blue-600">{rules.filter(r=>r.enabled).length}</div>
          <div className="text-sm text-gray-600">启用规则</div>
        </div>
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-gray-200/50 shadow-lg">
          <div className="text-2xl font-bold text-orange-600">{alerts.filter(a=>a.status==='active').length}</div>
          <div className="text-sm text-gray-600">活跃告警</div>
        </div>
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-gray-200/50 shadow-lg">
          <div className="text-2xl font-bold text-red-600">{alerts.filter(a=>a.severity==='critical'&&a.status==='active').length}</div>
          <div className="text-sm text-gray-600">紧急告警</div>
        </div>
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-gray-200/50 shadow-lg">
          <div className="text-2xl font-bold text-green-600">{alerts.filter(a=>a.status==='resolved').length}</div>
          <div className="text-sm text-gray-600">已解决</div>
        </div>
      </div>

      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-800">告警规则</h3>
          <button onClick={()=>setShowCreate(!showCreate)} className="px-4 py-2 bg-purple-600 text-white rounded-lg font-medium">
            {showCreate?'取消':'+ 创建规则'}
          </button>
        </div>

        {showCreate && (
          <div className="mb-6 p-4 bg-gray-50 rounded-xl">
            <h4 className="font-medium text-gray-800 mb-3">选择创建方式</h4>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">从模板创建</label>
              <div className="flex gap-2">
                <select value={selectedTemplate} onChange={(e)=>setSelectedTemplate(e.target.value)} className="flex-1 px-3 py-2 rounded-lg border border-gray-200 bg-white">
                  <option value="">选择模板...</option>
                  {templates.map(t=><option key={t.key} value={t.key}>{t.name}</option>)}
                </select>
                <button onClick={createFromTemplate} disabled={!selectedTemplate||loading} className="px-4 py-2 bg-blue-600 text-white rounded-lg disabled:opacity-50">
                  {loading?'创建中...':'创建'}
                </button>
              </div>
            </div>
            <div className="border-t border-gray-200 my-4"></div>
            <h4 className="font-medium text-gray-800 mb-3">或自定义创建</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">规则名称</label>
                <input type="text" value={formData.name} onChange={(e)=>setFormData({...formData,name:e.target.value})} className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white" placeholder="如：训练准确率过低" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">规则类型</label>
                <select value={formData.rule_type} onChange={(e)=>setFormData({...formData,rule_type:e.target.value})} className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white">
                  <option value="training">训练相关</option>
                  <option value="rul">RUL寿命</option>
                  <option value="data_drift">数据漂移</option>
                  <option value="system">系统资源</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">监控字段</label>
                <input type="text" value={formData.condition_field} onChange={(e)=>setFormData({...formData,condition_field:e.target.value})} className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white" placeholder="如：best_accuracy" />
              </div>
              <div className="flex gap-2">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-1">运算符</label>
                  <select value={formData.condition_operator} onChange={(e)=>setFormData({...formData,condition_operator:e.target.value})} className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white">
                    <option value="<">小于</option>
                    <option value=">">大于</option>
                    <option value="<=">小于等于</option>
                    <option value=">=">大于等于</option>
                  </select>
                </div>
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-1">阈值</label>
                  <input type="number" step="0.01" value={formData.condition_value} onChange={(e)=>setFormData({...formData,condition_value:parseFloat(e.target.value)})} className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white" />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">严重程度</label>
                <select value={formData.severity} onChange={(e)=>setFormData({...formData,severity:e.target.value})} className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white">
                  <option value="info">信息</option>
                  <option value="warning">警告</option>
                  <option value="error">错误</option>
                  <option value="critical">紧急</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">冷却时间(分钟)</label>
                <input type="number" value={formData.cooldown_minutes} onChange={(e)=>setFormData({...formData,cooldown_minutes:parseInt(e.target.value)})} className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white" />
              </div>
            </div>
            <button onClick={createRule} disabled={!formData.name||loading} className="mt-4 w-full py-2 bg-purple-600 text-white rounded-lg font-medium disabled:opacity-50">
              {loading?'创建中...':'创建自定义规则'}
            </button>
          </div>
        )}

        <div className="space-y-2">
          {rules.length===0?(
            <p className="text-gray-500 text-center py-4">暂无告警规则</p>
          ):(
            rules.map(rule=>(
              <div key={rule.id} className={`p-4 rounded-xl border ${rule.enabled?'bg-white border-gray-200':'bg-gray-50 border-gray-100'}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(rule.severity)}`}>{rule.severity}</span>
                    <div>
                      <div className="font-medium text-gray-800">{rule.name}</div>
                      <div className="text-sm text-gray-500">{rule.condition_field} {rule.condition_operator} {rule.condition_value} · {rule.rule_type}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${rule.enabled?'bg-green-500':'bg-gray-300'}`}></span>
                    <button onClick={()=>deleteRule(rule.id)} className="p-2 text-red-600 hover:bg-red-50 rounded-lg">🗑️</button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4">告警历史</h3>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {alerts.length===0?(
            <p className="text-gray-500 text-center py-4">暂无告警记录</p>
          ):(
            alerts.map(alert=>(
              <div key={alert.id} className={`p-4 rounded-xl border ${alert.status==='active'?'bg-red-50 border-red-200':'bg-gray-50 border-gray-200'}`}>
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <span className="text-xl">{getSeverityEmoji(alert.severity)}</span>
                    <div>
                      <div className="font-medium text-gray-800">{alert.title}</div>
                      <div className="text-sm text-gray-600 whitespace-pre-line">{alert.message}</div>
                      <div className="text-xs text-gray-400 mt-1">{new Date(alert.created_at).toLocaleString()}{alert.actual_value&&` · 实际值: ${alert.actual_value.toFixed(3)}`}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {alert.status==='active'&&(
                      <button onClick={()=>acknowledgeAlert(alert.id)} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-lg text-sm">确认</button>
                    )}
                    <span className={`px-2 py-1 rounded text-xs ${alert.status==='active'?'bg-red-100 text-red-700':'bg-green-100 text-green-700'}`}>{alert.status==='active'?'未处理':alert.status==='acknowledged'?'已确认':'已解决'}</span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

// 测试/应用 Tab - 模型推理测试
function TestTab({ projectId, projectType, jobs }) {
  const [testJobs, setTestJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  
  useEffect(() => {
    // 获取已完成的训练任务（已部署的模型）
    const completed = jobs && jobs.filter(j => j.status === 'completed') || [];
    setTestJobs(completed);
    if (completed.length > 0 && !selectedJob) {
      setSelectedJob(completed[0]);
    }
  }, [jobs, selectedJob]);

  return (
    <div className="space-y-6">
      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4">模型测试与应用</h3>
        
        {testJobs.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Icons.Brain className="w-12 h-12 mx-auto mb-4 text-purple-300" />
            <p>暂无可用模型</p>
            <p className="text-sm">请先完成模型训练</p>
          </div>
        ) : (
          <React.Fragment>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">选择模型</label>
              <select 
                value={selectedJob && selectedJob.id || ''} 
                onChange={(e) => {
                  const job = testJobs.find(j => j.id === e.target.value);
                  setSelectedJob(job);
                }}
                className="w-full px-3 py-2 rounded-lg border border-gray-200 bg-white focus:border-purple-400 focus:outline-none"
              >
                {testJobs.map(job => (
                  <option key={job.id} value={job.id}>
                    {job.model_name} (准确率: {(job.best_accuracy * 100).toFixed(2)}%)
                  </option>
                ))}
              </select>
            </div>
            
            {selectedJob && (
              <div className="mt-6">
                <TestPanel 
                  endpoint={`/api/projects/${projectId}/models/${selectedJob.id}/predict`}
                  projectType={projectType}
                  projectId={projectId}
                />
              </div>
            )}
          </React.Fragment>
        )}
      </div>
      
      <div className="bg-gradient-to-r from-cyan-50 to-blue-50 rounded-2xl p-6 border border-cyan-200">
        <h4 className="font-medium text-cyan-800 mb-2">💡 使用说明</h4>
        <ul className="text-sm text-cyan-700 space-y-1">
          <li>• 选择已训练好的模型进行在线测试</li>
          <li>• 输入样本数据，获取模型预测结果</li>
          <li>• 支持文本分类、时序预测、目标检测等多种任务</li>
          <li>• 部署后的模型可通过 API 接口在生产环境调用</li>
        </ul>
      </div>
    </div>
  );
}

// 模型服务管理 Tab - 推理服务监控运营
function ModelServicesTab({ projectId, jobs }) {
  const [services, setServices] = useState([]);
  const [memoryStatus, setMemoryStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({ totalCalls: 0, avgLatency: 0, successRate: 100, errorCount: 0 });
  const [trends, setTrends] = useState([]);
  const [latencyDist, setLatencyDist] = useState([]);
  const [recentLogs, setRecentLogs] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');
  const [alertRules, setAlertRules] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [showCreateRule, setShowCreateRule] = useState(false);
  const [ruleForm, setRuleForm] = useState({ name: '', rule_type: 'error_rate', threshold_value: 10, time_window_minutes: 5 });

  // 加载服务列表和内存状态
  const loadServices = async () => {
    try {
      const res = await fetch(`${API_BASE}/system/memory`);
      const data = await res.json();
      setMemoryStatus(data);
      setServices(data.models || []);
    } catch (e) {
      console.error('加载服务状态失败:', e);
    }
  };

  // 加载推理统计
  const loadStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/inference/stats`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (e) {
      console.error('加载统计失败:', e);
    }
  };

  // 加载趋势数据
  const loadTrends = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/inference/trends?hours=24`);
      if (res.ok) {
        const data = await res.json();
        setTrends(data.trends || []);
      }
    } catch (e) {
      console.error('加载趋势失败:', e);
    }
  };

  // 加载延迟分布
  const loadLatencyDist = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/inference/latency-distribution`);
      if (res.ok) {
        const data = await res.json();
        setLatencyDist(data.distribution || []);
      }
    } catch (e) {
      console.error('加载延迟分布失败:', e);
    }
  };

  // 加载最近日志
  const loadRecentLogs = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/inference/recent-logs?limit=50`);
      if (res.ok) {
        const data = await res.json();
        setRecentLogs(data.logs || []);
      }
    } catch (e) {
      console.error('加载日志失败:', e);
    }
  };

  // 加载告警规则
  const loadAlertRules = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/inference/alert-rules`);
      if (res.ok) {
        const data = await res.json();
        setAlertRules(data.rules || []);
      }
    } catch (e) {
      console.error('加载告警规则失败:', e);
    }
  };

  // 加载告警记录
  const loadAlerts = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/inference/alerts?limit=20`);
      if (res.ok) {
        const data = await res.json();
        setAlerts(data.alerts || []);
      }
    } catch (e) {
      console.error('加载告警失败:', e);
    }
  };

  // 加载优化建议
  const loadRecommendations = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/inference/recommendations`);
      if (res.ok) {
        const data = await res.json();
        setRecommendations(data.recommendations || []);
      }
    } catch (e) {
      console.error('加载优化建议失败:', e);
    }
  };

  // 创建告警规则
  const createAlertRule = async () => {
    if (!ruleForm.name) {
      alert('请输入规则名称');
      return;
    }
    try {
      const form = new FormData();
      form.append('name', ruleForm.name);
      form.append('rule_type', ruleForm.rule_type);
      form.append('threshold_value', ruleForm.threshold_value);
      form.append('time_window_minutes', ruleForm.time_window_minutes);
      
      const res = await fetch(`${API_BASE}/projects/${projectId}/inference/alert-rules`, {
        method: 'POST',
        body: form
      });
      if (res.ok) {
        setShowCreateRule(false);
        setRuleForm({ name: '', rule_type: 'error_rate', threshold_value: 10, time_window_minutes: 5 });
        loadAlertRules();
      }
    } catch (e) {
      alert('创建失败: ' + e.message);
    }
  };

  // 删除告警规则
  const deleteAlertRule = async (ruleId) => {
    if (!confirm('确定要删除这个告警规则吗？')) return;
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/inference/alert-rules/${ruleId}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        loadAlertRules();
      }
    } catch (e) {
      alert('删除失败: ' + e.message);
    }
  };

  // 切换告警规则状态
  const toggleAlertRule = async (ruleId) => {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/inference/alert-rules/${ruleId}/toggle`, {
        method: 'POST'
      });
      if (res.ok) {
        loadAlertRules();
      }
    } catch (e) {
      alert('操作失败: ' + e.message);
    }
  };

  // 确认告警
  const acknowledgeAlert = async (alertId) => {
    try {
      const res = await fetch(`${API_BASE}/inference/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user: 'admin' })
      });
      if (res.ok) {
        loadAlerts();
      }
    } catch (e) {
      alert('确认失败: ' + e.message);
    }
  };

  // 手动检查告警
  const checkAlerts = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/inference/check-alerts`, {
        method: 'POST'
      });
      if (res.ok) {
        const data = await res.json();
        if (data.triggered_alerts.length > 0) {
          alert(`发现 ${data.triggered_alerts.length} 个新告警`);
        } else {
          alert('未发现异常');
        }
        loadAlerts();
      }
    } catch (e) {
      alert('检查失败: ' + e.message);
    }
    setLoading(false);
  };

  // 手动加载模型到内存
  const loadModel = async (jobId) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}/jobs/${jobId}/load`, {
        method: 'POST'
      });
      if (res.ok) {
        alert('模型已加载到内存');
        loadServices();
      } else {
        alert('加载失败');
      }
    } catch (e) {
      alert('加载失败: ' + e.message);
    }
    setLoading(false);
  };

  // 卸载模型
  const unloadModel = async (modelId) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/inference/unload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId })
      });
      if (res.ok) {
        alert('模型已卸载');
        loadServices();
      }
    } catch (e) {
      alert('卸载失败: ' + e.message);
    }
    setLoading(false);
  };

  // 清理所有模型
  const cleanupAll = async () => {
    if (!confirm('确定要清理所有已加载的模型吗？')) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/inference/cleanup`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        alert(`已清理完成，释放 ${data.memory_freed_gb?.toFixed(2) || 0} GB 内存`);
        loadServices();
      }
    } catch (e) {
      alert('清理失败: ' + e.message);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadServices();
    loadStats();
    loadTrends();
    loadLatencyDist();
    loadRecentLogs();
    loadAlertRules();
    loadAlerts();
    loadRecommendations();
    // 每10秒刷新一次
    const interval = setInterval(() => {
      loadServices();
      loadStats();
      loadTrends();
      loadRecentLogs();
      loadAlerts();
      loadRecommendations();
    }, 10000);
    return () => clearInterval(interval);
  }, [projectId]);

  const completedJobs = jobs?.filter(j => j.status === 'completed') || [];
  const isHighMemory = memoryStatus?.memory_percent > 80;
  const isCritical = memoryStatus?.memory_percent > 90;

  return (
    <div className="space-y-6">
      {/* 运营概览卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-gray-200/50 shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <Icons.Server />
            <span className="text-sm text-gray-600">活跃服务</span>
          </div>
          <div className="text-2xl font-bold text-indigo-600">{services.length}</div>
          <div className="text-xs text-gray-500">已加载模型</div>
        </div>
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-gray-200/50 shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <Icons.Activity />
            <span className="text-sm text-gray-600">总调用次数</span>
          </div>
          <div className="text-2xl font-bold text-blue-600">{stats.totalCalls}</div>
          <div className="text-xs text-gray-500">累计推理请求</div>
        </div>
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-gray-200/50 shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <Icons.Clock />
            <span className="text-sm text-gray-600">平均延迟</span>
          </div>
          <div className="text-2xl font-bold text-green-600">{stats.avgLatency}ms</div>
          <div className="text-xs text-gray-500">响应时间</div>
        </div>
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-gray-200/50 shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <Icons.Check />
            <span className="text-sm text-gray-600">成功率</span>
          </div>
          <div className="text-2xl font-bold text-purple-600">{stats.successRate}%</div>
          <div className="text-xs text-gray-500">服务可用性</div>
        </div>
      </div>

      {/* 内存状态面板 */}
      <div className={`p-4 rounded-xl border ${
        isCritical ? 'bg-red-50 border-red-200' :
        isHighMemory ? 'bg-orange-50 border-orange-200' :
        'bg-blue-50 border-blue-200'
      }`}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Icons.Memory />
            <span className="font-medium text-gray-800">系统内存状态</span>
            {isCritical && <span className="px-2 py-0.5 bg-red-500 text-white rounded text-xs">危险</span>}
            {isHighMemory && !isCritical && <span className="px-2 py-0.5 bg-orange-500 text-white rounded text-xs">告警</span>}
          </div>
          <div className="flex gap-2">
            <button onClick={loadServices} className="text-xs text-gray-500 hover:text-gray-700 px-3 py-1 bg-white rounded">
              <Icons.Refresh /> 刷新
            </button>
            <button 
              onClick={cleanupAll}
              disabled={loading || services.length === 0}
              className="text-xs px-3 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200 disabled:opacity-50"
            >
              {loading ? '处理中...' : '🧹 清理所有'}
            </button>
          </div>
        </div>

        {memoryStatus && (
          <React.Fragment>
            <div className="mb-3">
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>已使用 {memoryStatus.memory_percent?.toFixed(1)}%</span>
                <span>{memoryStatus.memory_used_gb?.toFixed(1)} / {memoryStatus.memory_total_gb?.toFixed(1)} GB</span>
              </div>
              <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    isCritical ? 'bg-red-500' :
                    isHighMemory ? 'bg-orange-500' :
                    'bg-green-500'
                  }`}
                  style={{ width: `${Math.min(memoryStatus.memory_percent || 0, 100)}%` }}
                />
              </div>
            </div>
            <div className="text-xs text-gray-500">
              💡 自动管理: 内存&gt;80%触发清理 · 30分钟空闲自动卸载 · 最多保留{memoryStatus.max_models_limit}个模型
            </div>
          </React.Fragment>
        )}
      </div>

      {/* 优化建议 */}
      {recommendations.length > 0 && (
        <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-2xl p-6 border border-amber-200">
          <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Icons.Sparkles /> 运营优化建议
          </h3>
          <div className="space-y-3">
            {recommendations.map((rec, idx) => (
              <div key={idx} className={`p-3 rounded-lg ${
                rec.priority === 'critical' ? 'bg-red-100 border border-red-200' :
                rec.priority === 'high' ? 'bg-orange-100 border border-orange-200' :
                'bg-white border border-gray-200'
              }`}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        rec.priority === 'critical' ? 'bg-red-500 text-white' :
                        rec.priority === 'high' ? 'bg-orange-500 text-white' :
                        'bg-gray-500 text-white'
                      }`}>
                        {rec.priority === 'critical' ? '紧急' : rec.priority === 'high' ? '重要' : '建议'}
                      </span>
                      <span className="font-medium text-sm">{rec.title}</span>
                    </div>
                    <p className="text-xs text-gray-600">{rec.description}</p>
                  </div>
                  <span className="text-xs text-gray-400 ml-2">{rec.action}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 活跃告警 */}
      {alerts.filter(a => a.status === 'active').length > 0 && (
        <div className="bg-red-50 rounded-2xl p-6 border border-red-200">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-red-800 flex items-center gap-2">
              <Icons.Bell /> 活跃告警 ({alerts.filter(a => a.status === 'active').length})
            </h3>
            <button 
              onClick={checkAlerts}
              disabled={loading}
              className="text-xs px-3 py-1 bg-white text-red-600 rounded border border-red-200 hover:bg-red-50"
            >
              {loading ? '检查中...' : '🔍 立即检查'}
            </button>
          </div>
          <div className="space-y-2">
            {alerts.filter(a => a.status === 'active').map((alert, idx) => (
              <div key={idx} className="p-3 bg-white rounded-lg border border-red-100">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        alert.severity === 'critical' ? 'bg-red-500 text-white' : 'bg-orange-500 text-white'
                      }`}>
                        {alert.severity === 'critical' ? '严重' : '警告'}
                      </span>
                      <span className="font-medium text-sm">{alert.title}</span>
                    </div>
                    <p className="text-xs text-gray-600">{alert.message}</p>
                    <div className="text-xs text-gray-400 mt-1">
                      当前值: {alert.metricValue?.toFixed ? alert.metricValue.toFixed(2) : alert.metricValue} / 阈值: {alert.thresholdValue}
                    </div>
                  </div>
                  <button
                    onClick={() => acknowledgeAlert(alert.id)}
                    className="text-xs px-3 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200"
                  >
                    确认
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 已加载模型列表 */}
      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Icons.Server /> 已加载模型服务
        </h3>
        
        {services.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Icons.Server className="w-12 h-12 mx-auto mb-4 text-gray-300" />
            <p>暂无已加载的模型服务</p>
            <p className="text-sm">从下方列表加载模型到内存</p>
          </div>
        ) : (
          <div className="space-y-2">
            {services.map((service, idx) => (
              <div key={idx} className="p-3 bg-gray-50 rounded-lg flex items-center justify-between">
                <div className="flex-1">
                  <div className="font-medium text-sm">{service.model_id}</div>
                  <div className="text-xs text-gray-500">
                    类型: {service.model_type} · 
                    内存: {service.memory_size_mb}MB · 
                    访问: {service.access_count}次 · 
                    空闲: {service.idle_seconds}s
                  </div>
                </div>
                <button
                  onClick={() => unloadModel(service.model_id)}
                  disabled={loading}
                  className="px-3 py-1 bg-red-100 text-red-700 rounded text-xs font-medium hover:bg-red-200 disabled:opacity-50"
                >
                  卸载
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 可用模型列表 */}
      <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
        <h3 className="font-semibold text-gray-800 mb-4">可用模型（已训练完成）</h3>
        
        {completedJobs.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Icons.Brain className="w-12 h-12 mx-auto mb-4 text-gray-300" />
            <p>暂无可用模型</p>
            <p className="text-sm">请先完成模型训练</p>
          </div>
        ) : (
          <div className="space-y-2">
            {completedJobs.map(job => {
              const isLoaded = services.some(s => s.model_id === `${projectId}/${job.id}`);
              return (
                <div key={job.id} className="p-3 bg-gray-50 rounded-lg flex items-center justify-between">
                  <div className="flex-1">
                    <div className="font-medium text-sm">{job.model_name}</div>
                    <div className="text-xs text-gray-500">
                      准确率: {(job.best_accuracy * 100).toFixed(2)}% · 
                      训练时间: {new Date(job.completed_at).toLocaleString()}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {isLoaded ? (
                      <span className="px-2 py-1 bg-green-100 text-green-700 rounded text-xs">已加载</span>
                    ) : (
                      <button
                        onClick={() => loadModel(job.id)}
                        disabled={loading || services.length >= (memoryStatus?.max_models_limit || 5)}
                        className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded text-xs font-medium hover:bg-indigo-200 disabled:opacity-50"
                      >
                        加载到内存
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* 详细统计面板 */}
      {stats.totalCalls > 0 && (
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-200/50 shadow-lg">
          <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Icons.Chart /> 推理调用分析
          </h3>
          
          {/* 子标签页 */}
          <div className="flex gap-2 mb-4 border-b">
            {[
              { id: 'trends', label: '调用趋势', icon: '📈' },
              { id: 'latency', label: '延迟分布', icon: '⏱️' },
              { id: 'logs', label: '最近日志', icon: '📋' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab.icon} {tab.label}
              </button>
            ))}
          </div>

          {/* 调用趋势 */}
          {activeTab === 'trends' && (
            <div>
              {trends.length === 0 ? (
                <p className="text-gray-500 text-center py-4">暂无趋势数据</p>
              ) : (
                <div className="space-y-3">
                  <div className="text-xs text-gray-500 mb-2">最近24小时调用趋势</div>
                  {trends.slice(0, 12).map((t, idx) => (
                    <div key={idx} className="flex items-center gap-3">
                      <div className="w-20 text-xs text-gray-600">
                        {new Date(t.hour).getHours()}:00
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <div 
                            className="h-4 bg-blue-500 rounded"
                            style={{ width: `${Math.min((t.callCount / Math.max(...trends.map(x => x.callCount))) * 100, 100)}%` }}
                          />
                          <span className="text-xs text-gray-600">{t.callCount}次</span>
                        </div>
                      </div>
                      <div className="w-20 text-xs text-gray-500">
                        {t.avgLatency}ms
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* 延迟分布 */}
          {activeTab === 'latency' && (
            <div>
              {latencyDist.length === 0 ? (
                <p className="text-gray-500 text-center py-4">暂无延迟数据</p>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {latencyDist.map((item, idx) => (
                    <div key={idx} className="p-3 bg-gray-50 rounded-lg text-center">
                      <div className="text-2xl font-bold text-indigo-600">{item.count}</div>
                      <div className="text-xs text-gray-500">{item.range}</div>
                    </div>
                  ))}
                </div>
              )}
              <div className="mt-4 p-3 bg-blue-50 rounded-lg text-xs text-blue-700">
                💡 延迟分布说明：展示成功推理请求的响应时间分布，帮助你了解模型服务的性能表现
              </div>
            </div>
          )}

          {/* 最近日志 */}
          {activeTab === 'logs' && (
            <div className="max-h-64 overflow-auto">
              {recentLogs.length === 0 ? (
                <p className="text-gray-500 text-center py-4">暂无日志记录</p>
              ) : (
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">时间</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">模型</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">延迟</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">状态</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {recentLogs.map((log, idx) => (
                      <tr key={idx} className={log.success ? '' : 'bg-red-50'}>
                        <td className="px-3 py-2 text-xs text-gray-600">
                          {new Date(log.createdAt).toLocaleTimeString()}
                        </td>
                        <td className="px-3 py-2 text-xs text-gray-800 truncate max-w-[150px]">
                          {log.modelName}
                        </td>
                        <td className="px-3 py-2 text-xs text-gray-600">
                          {log.latencyMs}ms
                        </td>
                        <td className="px-3 py-2">
                          {log.success ? (
                            <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs">成功</span>
                          ) : (
                            <span className="px-2 py-0.5 bg-red-100 text-red-700 rounded text-xs" title={log.errorMessage}>失败</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
