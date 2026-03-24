#!/bin/bash

# CAD 几何智能处理系统 - 服务管理脚本
# 使用方法：
#   ./scripts/manage.sh start    - 启动前后端服务
#   ./scripts/manage.sh stop     - 停止所有服务
#   ./scripts/manage.sh restart  - 重启所有服务
#   ./scripts/manage.sh status   - 查看服务状态
#   ./scripts/manage.sh backend  - 只启动后端
#   ./scripts/manage.sh frontend - 只启动前端

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CAD_WEB_DIR="$PROJECT_ROOT/cad-web"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查服务是否运行
check_backend() {
    if curl -s http://localhost:3000/health >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

check_frontend() {
    if curl -s http://localhost:5173 >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# 停止服务
stop_services() {
    log_info "正在停止服务..."

    # 停止后端
    backend_pids=$(pgrep -f "cad-cli.*serve" 2>/dev/null || true)
    if [ -n "$backend_pids" ]; then
        log_info "停止后端服务 (PIDs: $backend_pids)"
        echo "$backend_pids" | xargs kill 2>/dev/null || true
    fi

    orchestrator_pids=$(pgrep -f "orchestrator.*serve" 2>/dev/null || true)
    if [ -n "$orchestrator_pids" ]; then
        log_info "停止 orchestrator 服务 (PIDs: $orchestrator_pids)"
        echo "$orchestrator_pids" | xargs kill 2>/dev/null || true
    fi

    # 停止前端
    frontend_pids=$(pgrep -f "vite.*cad-web" 2>/dev/null || true)
    if [ -n "$frontend_pids" ]; then
        log_info "停止前端服务 (PIDs: $frontend_pids)"
        echo "$frontend_pids" | xargs kill 2>/dev/null || true
    fi

    sleep 2

    # 强制释放端口
    if lsof -ti :3000 >/dev/null 2>&1; then
        lsof -ti :3000 | xargs kill -9 2>/dev/null || true
    fi
    if lsof -ti :5173 >/dev/null 2>&1; then
        lsof -ti :5173 | xargs kill -9 2>/dev/null || true
    fi

    log_success "所有服务已停止"
}

# 启动后端
start_backend() {
    if check_backend; then
        log_warn "后端已在运行 (http://localhost:3000)"
        return 0
    fi

    log_info "启动后端服务 (端口 3000)..."
    cd "$PROJECT_ROOT"
    cargo run -p cad-cli -- serve --port 3000 &
    
    log_info "等待后端启动..."
    for i in {1..30}; do
        if check_backend; then
            log_success "后端服务已启动 (http://localhost:3000/health)"
            return 0
        fi
        sleep 1
    done

    log_error "后端启动超时"
    return 1
}

# 启动前端
start_frontend() {
    if check_frontend; then
        log_warn "前端已在运行 (http://localhost:5173)"
        return 0
    fi

    log_info "启动前端服务 (端口 5173)..."
    cd "$CAD_WEB_DIR"

    if [ ! -d "node_modules" ]; then
        log_info "安装依赖..."
        pnpm install
    fi

    pnpm run dev &
    
    log_info "等待前端启动..."
    for i in {1..30}; do
        if check_frontend; then
            log_success "前端服务已启动 (http://localhost:5173)"
            return 0
        fi
        sleep 1
    done

    log_error "前端启动超时"
    return 1
}

# 查看状态
show_status() {
    echo "========================================"
    echo "  CAD 系统服务状态"
    echo "========================================"
    
    if check_backend; then
        echo -e "  后端：${GREEN}● 运行中${NC} (http://localhost:3000)"
    else
        echo -e "  后端：${RED}● 已停止${NC}"
    fi
    
    if check_frontend; then
        echo -e "  前端：${GREEN}● 运行中${NC} (http://localhost:5173)"
    else
        echo -e "  前端：${RED}● 已停止${NC}"
    fi
    
    echo "========================================"
}

# 主函数
case "${1:-}" in
    start)
        log_info "启动所有服务..."
        stop_services
        start_backend
        start_frontend
        echo ""
        log_success "启动完成!"
        echo "  前端：http://localhost:5173"
        echo "  后端：http://localhost:3000"
        echo ""
        log_info "按 Ctrl+C 停止所有服务"
        wait
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        start_backend
        start_frontend
        log_success "重启完成!"
        ;;
    status)
        show_status
        ;;
    backend)
        start_backend
        wait
        ;;
    frontend)
        start_frontend
        wait
        ;;
    *)
        echo "用法：$0 {start|stop|restart|status|backend|frontend}"
        echo ""
        echo "  start    - 启动前后端服务"
        echo "  stop     - 停止所有服务"
        echo "  restart  - 重启所有服务"
        echo "  status   - 查看服务状态"
        echo "  backend  - 只启动后端"
        echo "  frontend - 只启动前端"
        exit 1
        ;;
esac
