#!/bin/bash
# AI训练平台监控启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITORING_DIR="$SCRIPT_DIR/monitoring"
DATA_DIR="$MONITORING_DIR/data"

echo "=== AI训练平台监控启动 ==="

# 创建数据目录
mkdir -p "$DATA_DIR/prometheus"
mkdir -p "$DATA_DIR/grafana"

# 检查是否已安装 Prometheus
if ! command -v prometheus &> /dev/null; then
    echo "Prometheus 未安装，正在下载..."
    cd /tmp
    wget -q https://github.com/prometheus/prometheus/releases/download/v2.48.0/prometheus-2.48.0.linux-amd64.tar.gz
    tar -xzf prometheus-2.48.0.linux-amd64.tar.gz
    sudo mv prometheus-2.48.0.linux-amd64/prometheus /usr/local/bin/
    sudo mv prometheus-2.48.0.linux-amd64/promtool /usr/local/bin/
    rm -rf prometheus-2.48.0.linux-amd64*
    echo "Prometheus 安装完成"
fi

# 检查是否已安装 Grafana
if ! command -v grafana-server &> /dev/null; then
    echo "Grafana 未安装，正在安装..."
    sudo yum install -y https://dl.grafana.com/oss/release/grafana-10.2.3-1.x86_64.rpm 2>/dev/null || \
    sudo apt-get install -y grafana 2>/dev/null || \
    echo "请手动安装 Grafana"
fi

# 启动 Prometheus
echo "启动 Prometheus..."
nohup prometheus \
    --config.file="$MONITORING_DIR/prometheus.yml" \
    --storage.tsdb.path="$DATA_DIR/prometheus" \
    --storage.tsdb.retention.time=7d \
    --web.listen-address=:9090 \
    > "$MONITORING_DIR/prometheus.log" 2>&1 &

PROMETHEUS_PID=$!
echo $PROMETHEUS_PID > "$MONITORING_DIR/prometheus.pid"
echo "Prometheus 启动完成 (PID: $PROMETHEUS_PID)，访问 http://localhost:9090"

# 启动 Grafana (如果已安装)
if command -v grafana-server &> /dev/null; then
    echo "启动 Grafana..."
    
    # 配置 Grafana
    mkdir -p /etc/grafana/provisioning/datasources
    mkdir -p /etc/grafana/provisioning/dashboards
    
    cat > /etc/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    url: http://localhost:9090
    access: proxy
    isDefault: true
EOF

    nohup grafana-server \
        --homepath=/usr/share/grafana \
        --config=/etc/grafana/grafana.ini \
        > "$MONITORING_DIR/grafana.log" 2>&1 &
    
    GRAFANA_PID=$!
    echo $GRAFANA_PID > "$MONITORING_DIR/grafana.pid"
    echo "Grafana 启动完成 (PID: $GRAFANA_PID)，访问 http://localhost:3000"
    echo "默认账号: admin / admin"
fi

echo ""
echo "=== 监控启动完成 ==="
echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3000 (如已安装)"
echo ""
echo "查看日志:"
echo "  tail -f $MONITORING_DIR/prometheus.log"
echo "  tail -f $MONITORING_DIR/grafana.log"
