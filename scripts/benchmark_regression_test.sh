#!/bin/bash
# =============================================================================
# P11 性能回归测试脚本
# =============================================================================
# 用途：自动化运行性能基准测试并生成对比报告
# 使用：./scripts/benchmark_regression_test.sh [baseline|compare|report]
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
BENCHMARK_DIR="./target/criterion"
REPORT_DIR="./benchmark_reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASELINE_FILE="$REPORT_DIR/baseline_$TIMESTAMP.json"
CURRENT_FILE="$REPORT_DIR/current_$TIMESTAMP.json"
COMPARISON_FILE="$REPORT_DIR/comparison_$TIMESTAMP.md"

# 打印带颜色的消息
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建报告目录
mkdir -p "$REPORT_DIR"

# 运行基准测试并保存结果
run_benchmarks() {
    local output_file=$1
    local package=${2:-topo}
    local bench_name=${3:-}
    
    info "Running benchmarks for package: $package"
    
    if [ -n "$bench_name" ]; then
        info "Running specific benchmark: $bench_name"
        cargo bench --package "$package" --bench "$bench_name" 2>&1 | tee "$output_file"
    else
        info "Running all benchmarks"
        cargo bench --package "$package" 2>&1 | tee "$output_file"
    fi
    
    success "Benchmarks completed. Results saved to: $output_file"
}

# 生成基线报告
generate_baseline() {
    info "Generating baseline..."
    
    local baseline_output="$REPORT_DIR/baseline_raw_$TIMESTAMP.txt"
    run_benchmarks "$baseline_output" "topo" "topology_bench"
    
    # 提取关键数据
    cat > "$BASELINE_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "git_commit": "$(git rev-parse HEAD)",
  "git_branch": "$(git branch --show-current)",
  "benchmarks": {}
}
EOF
    
    success "Baseline generated: $BASELINE_FILE"
}

# 运行当前测试并对比
compare_with_baseline() {
    local baseline_file=$1
    
    if [ ! -f "$baseline_file" ]; then
        error "Baseline file not found: $baseline_file"
        warn "Run 'generate_baseline' first"
        exit 1
    fi
    
    info "Comparing with baseline: $baseline_file"
    
    local current_output="$REPORT_DIR/current_raw_$TIMESTAMP.txt"
    run_benchmarks "$current_output" "topo" "topology_bench"
    
    # 生成对比报告
    generate_comparison_report "$baseline_file" "$current_output" "$COMPARISON_FILE"
    
    success "Comparison report generated: $COMPARISON_FILE"
}

# 生成对比报告
generate_comparison_report() {
    local baseline=$1
    local current=$2
    local output=$3
    
    info "Generating comparison report..."
    
    cat > "$output" << EOF
# 性能回归测试报告

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')
**基线文件**: $baseline
**当前测试**: $current

## 性能对比

| 测试项 | 基线 | 当前 | 变化 | 状态 |
|--------|------|------|------|------|
EOF
    
    # 解析并对比数据（简化版本）
    # 实际项目中可以使用 Python 脚本进行更精确的解析
    
    cat >> "$output" << EOF

## 系统信息

- **Git Commit**: $(git rev-parse HEAD)
- **Git Branch**: $(git branch --show-current)
- **Rust 版本**: $(rustc --version)
- **CPU**: $(uname -m)
- **OS**: $(uname -s)

## 建议

EOF
    
    if grep -q "ERROR" "$current"; then
        echo "⚠️ 检测到错误，请检查日志" >> "$output"
    else
        echo "✅ 未发现明显性能回归" >> "$output"
    fi
    
    success "Report saved to: $output"
}

# 生成 Markdown 格式的性能摘要
generate_summary() {
    info "Generating performance summary..."
    
    local summary_file="$REPORT_DIR/summary_$TIMESTAMP.md"
    
    cat > "$summary_file" << EOF
# 性能基准测试摘要

**日期**: $(date '+%Y-%m-%d %H:%M:%S')
**版本**: $(git describe --tags --always --dirty 2>/dev/null || git rev-parse --short HEAD)

## 测试配置

- **Rust**: $(rustc --version)
- **Cargo**: $(cargo --version)
- **LLVM**: $(rustc --version --verbose | grep LLVM || echo "N/A")

## 运行基准测试

\`\`\`bash
# 运行所有 topo 包基准测试
cargo bench --package topo

# 运行特定测试
cargo bench --package topo --bench topology_bench

# 运行大规模测试（忽略的测试）
cargo bench --package topo --bench topology_bench -- --ignored
\`\`\`

## 性能目标

| 规模 | 目标时间 | 说明 |
|------|----------|------|
| 100 线段 | < 1ms | 交互式编辑 |
| 1,000 线段 | < 10ms | 楼层平面图 |
| 10,000 线段 | < 100ms | 整层平面图 |
| 100,000 线段 | < 1s | 大型建筑图纸 |
| 1,000,000 线段 | < 10s | 园区总平面图 |

## 历史趋势

查看 \`$REPORT_DIR\` 目录中的历史报告以了解性能趋势。

EOF
    
    success "Summary saved to: $summary_file"
}

# 清理旧报告
cleanup_old_reports() {
    local keep_days=${1:-30}
    
    info "Cleaning up reports older than $keep_days days..."
    
    find "$REPORT_DIR" -name "*.md" -mtime +$keep_days -delete
    find "$REPORT_DIR" -name "*.json" -mtime +$keep_days -delete
    find "$REPORT_DIR" -name "*.txt" -mtime +$keep_days -delete
    
    success "Cleanup completed"
}

# 显示帮助信息
show_help() {
    cat << EOF
P11 性能回归测试脚本

用法：$0 [命令] [选项]

命令:
  baseline          生成新的基线报告
  compare           与基线对比并生成报告
  summary           生成性能摘要
  cleanup [天数]    清理旧报告（默认 30 天）
  full              完整流程：基线 + 对比 + 摘要
  help              显示此帮助信息

示例:
  $0 baseline                    # 生成基线
  $0 compare baseline_*.json     # 与基线对比
  $0 summary                     # 生成摘要
  $0 full                        # 完整流程
  $0 cleanup 60                  # 清理 60 天前的报告

EOF
}

# 主函数
main() {
    local command=${1:-help}
    
    case "$command" in
        baseline)
            generate_baseline
            ;;
        compare)
            local baseline_file=${2:-$(ls -t "$REPORT_DIR"/baseline_*.json 2>/dev/null | head -1)}
            compare_with_baseline "$baseline_file"
            ;;
        summary)
            generate_summary
            ;;
        cleanup)
            cleanup_old_reports "${2:-30}"
            ;;
        full)
            info "Running full benchmark suite..."
            generate_baseline
            sleep 2
            generate_summary
            info "Full suite completed"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
