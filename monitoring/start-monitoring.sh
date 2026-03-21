#!/bin/bash
# CAD EaaS 架构 - 监控栈快速启动脚本 (Linux/Mac)
# 用于快速启动 Prometheus + Grafana 监控环境

echo "========================================"
echo "CAD EaaS 架构 - 监控栈启动脚本"
echo "========================================"
echo ""

cd "$(dirname "$0")"

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo "[错误] Docker 未运行或未安装"
    echo "请先启动 Docker 或安装 Docker"
    exit 1
fi

echo "[信息] 启动监控服务..."
echo ""

# 启动所有服务
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "[错误] 启动失败"
    exit 1
fi

echo ""
echo "========================================"
echo "服务启动成功!"
echo "========================================"
echo ""
echo "访问地址:"
echo "  - Grafana:     http://localhost:3000 (admin/admin123)"
echo "  - Prometheus:  http://localhost:9090"
echo "  - Node Exporter: http://localhost:9100"
echo ""
echo "常用命令:"
echo "  - 查看日志：docker-compose logs -f"
echo "  - 停止服务：docker-compose down"
echo "  - 重启服务：docker-compose restart"
echo ""
echo "========================================"
