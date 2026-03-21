@echo off
REM CAD EaaS 架构 - 监控栈快速启动脚本 (Windows)
REM 用于快速启动 Prometheus + Grafana 监控环境

echo ========================================
echo CAD EaaS 架构 - 监控栈启动脚本
echo ========================================
echo.

cd /d "%~dp0"

REM 检查 Docker 是否运行
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker 未运行或未安装
    echo 请先启动 Docker Desktop 或安装 Docker
    pause
    exit /b 1
)

echo [信息] 启动监控服务...
echo.

REM 启动所有服务
docker-compose up -d

if %errorlevel% neq 0 (
    echo [错误] 启动失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 服务启动成功!
echo ========================================
echo.
echo 访问地址:
echo   - Grafana:     http://localhost:3000 (admin/admin123)
echo   - Prometheus:  http://localhost:9090
echo   - Node Exporter: http://localhost:9100
echo.
echo 常用命令:
echo   - 查看日志：docker-compose logs -f
echo   - 停止服务：docker-compose down
echo   - 重启服务：docker-compose restart
echo.
echo ========================================

pause
