#!/bin/bash

# CAD 几何智能处理系统 - 前后端启动脚本
# 使用方法：./scripts/start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CAD_WEB_DIR="$PROJECT_ROOT/cad-web"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 捕获退出信号，清理进程
cleanup() {
    log_info "正在停止服务..."
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
        log_info "后端服务已停止 (PID: $BACKEND_PID)"
    fi
    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
        log_info "前端服务已停止 (PID: $FRONTEND_PID)"
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# 停止已运行的服务
stop_existing_services() {
    log_info "检查并停止已运行的服务..."
    
    # 停止后端服务 (cad-cli serve)
    backend_pids=$(pgrep -f "cad-cli.*serve" 2>/dev/null || true)
    if [ -n "$backend_pids" ]; then
        log_info "停止后端服务 (PIDs: $backend_pids)"
        echo "$backend_pids" | xargs kill 2>/dev/null || true
        sleep 1
    fi
    
    # 停止 orchestrator 服务
    orchestrator_pids=$(pgrep -f "orchestrator.*serve" 2>/dev/null || true)
    if [ -n "$orchestrator_pids" ]; then
        log_info "停止 orchestrator 服务 (PIDs: $orchestrator_pids)"
        echo "$orchestrator_pids" | xargs kill 2>/dev/null || true
        sleep 1
    fi
    
    # 停止前端服务 (vite)
    frontend_pids=$(pgrep -f "vite.*cad-web" 2>/dev/null || true)
    if [ -n "$frontend_pids" ]; then
        log_info "停止前端服务 (PIDs: $frontend_pids)"
        echo "$frontend_pids" | xargs kill 2>/dev/null || true
        sleep 1
    fi
    
    # 等待端口释放
    sleep 2
    
    # 检查端口是否仍被占用
    if lsof -i :3000 >/dev/null 2>&1; then
        log_warn "端口 3000 仍被占用，尝试强制释放..."
        lsof -ti :3000 | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    
    if lsof -i :5173 >/dev/null 2>&1; then
        log_warn "端口 5173 仍被占用，尝试强制释放..."
        lsof -ti :5173 | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    
    log_success "已清理旧服务"
}

# 启动后端服务
start_backend() {
    log_info "启动后端服务 (端口 3000)..."
    
    cd "$PROJECT_ROOT"
    
    # 使用 cad-cli serve 启动
    cargo run -p cad-cli -- serve --port 3000 &
    BACKEND_PID=$!
    
    log_info "后端服务 PID: $BACKEND_PID"
    
    # 等待后端启动
    log_info "等待后端服务启动..."
    for i in {1..30}; do
        if curl -s http://localhost:3000/health >/dev/null 2>&1; then
            log_success "后端服务已启动 (http://localhost:3000)"
            return 0
        fi
        sleep 1
    done
    
    log_error "后端服务启动超时"
    return 1
}

# 启动前端服务
start_frontend() {
    log_info "启动前端服务 (端口 5173)..."
    
    cd "$CAD_WEB_DIR"
    
    # 检查 node_modules
    if [ ! -d "node_modules" ]; then
        log_warn "node_modules 不存在，正在安装依赖..."
        pnpm install
    fi
    
    # 启动 Vite 开发服务器
    pnpm run dev &
    FRONTEND_PID=$!
    
    log_info "前端服务 PID: $FRONTEND_PID"
    
    # 等待前端启动
    log_info "等待前端服务启动..."
    for i in {1..30}; do
        if curl -s http://localhost:5173 >/dev/null 2>&1; then
            log_success "前端服务已启动 (http://localhost:5173)"
            return 0
        fi
        sleep 1
    done
    
    log_error "前端服务启动超时"
    return 1
}

# 主函数
main() {
    echo "========================================"
    echo "  CAD 几何智能处理系统 - 启动脚本"
    echo "========================================"
    echo ""
    
    # 停止旧服务
    stop_existing_services
    
    # 启动后端
    if ! start_backend; then
        log_error "后端启动失败"
        exit 1
    fi
    
    # 启动前端
    if ! start_frontend; then
        log_error "前端启动失败"
        exit 1
    fi
    
    echo ""
    echo "========================================"
    log_success "所有服务已启动!"
    echo "========================================"
    echo ""
    echo "  前端：http://localhost:5173"
    echo "  后端：http://localhost:3000"
    echo ""
    echo "  按 Ctrl+C 停止所有服务"
    echo ""
    echo "========================================"
    
    # 等待用户中断
    wait
}

# 运行主函数
main
