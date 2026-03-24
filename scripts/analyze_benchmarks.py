#!/usr/bin/env python3
"""
P11 性能基准测试数据分析工具

用途：
1. 解析 criterion 基准测试输出
2. 生成性能对比报告
3. 检测性能回归
4. 生成可视化图表数据

使用：
    python scripts/analyze_benchmarks.py [baseline.json] [current.json]
"""

import json
import re
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """单个基准测试结果"""
    name: str
    time_ns: float  # 纳秒
    throughput: Optional[float] = None  # 每秒操作数
    change_percent: Optional[float] = None
    regression: bool = False


@dataclass
class BenchmarkReport:
    """基准测试报告"""
    timestamp: str
    git_commit: str
    git_branch: str
    rust_version: str
    benchmarks: Dict[str, BenchmarkResult]
    summary: Dict[str, float]


def parse_criterion_output(output_file: str) -> Dict[str, BenchmarkResult]:
    """
    解析 criterion 基准测试输出文件
    
    criterion 输出格式示例：
    topology_small/100 矩形
                        time:   [1.2345 ms 1.2456 ms 1.2567 ms]
                        thrpt:  [est. 800.12 Kiter/s, 810.34 Kiter/s, 820.56 Kiter/s]
    """
    benchmarks = {}
    
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 benchmark 名称和时间
    pattern = r'(\S+/[\w\d_]+\s*).*?\[(\d+\.\d+) (ms|μs|ns).*?\]'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        name = match[0].strip()
        time_value = float(match[1])
        time_unit = match[2]
        
        # 转换为纳秒
        if time_unit == 'ms':
            time_ns = time_value * 1_000_000
        elif time_unit == 'μs':
            time_ns = time_value * 1_000
        else:  # ns
            time_ns = time_value
        
        benchmarks[name] = BenchmarkResult(
            name=name,
            time_ns=time_ns
        )
    
    return benchmarks


def compare_benchmarks(
    baseline: Dict[str, BenchmarkResult],
    current: Dict[str, BenchmarkResult],
    regression_threshold: float = 10.0
) -> Dict[str, BenchmarkResult]:
    """
    对比两个基准测试结果
    
    Args:
        baseline: 基线数据
        current: 当前数据
        regression_threshold: 性能回归阈值（百分比）
    
    Returns:
        包含对比结果的字典
    """
    results = {}
    
    for name, current_result in current.items():
        if name in baseline:
            baseline_time = baseline[name].time_ns
            current_time = current_result.time_ns
            
            change = ((current_time - baseline_time) / baseline_time) * 100
            current_result.change_percent = change
            current_result.regression = change > regression_threshold
            
            results[name] = current_result
        else:
            # 新的 benchmark
            current_result.change_percent = None
            results[name] = current_result
    
    return results


def calculate_summary(benchmarks: Dict[str, BenchmarkResult]) -> Dict[str, float]:
    """计算性能摘要统计"""
    if not benchmarks:
        return {}
    
    changes = [b.change_percent for b in benchmarks.values() if b.change_percent is not None]
    regressions = sum(1 for b in benchmarks.values() if b.regression)
    
    return {
        'total_benchmarks': len(benchmarks),
        'avg_change_percent': sum(changes) / len(changes) if changes else 0,
        'max_improvement': min(changes) if changes else 0,
        'max_regression': max(changes) if changes else 0,
        'regression_count': regressions,
        'improvement_count': sum(1 for c in changes if c < 0),
    }


def get_git_info() -> Tuple[str, str]:
    """获取 Git 信息"""
    import subprocess
    
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = 'unknown'
    
    try:
        branch = subprocess.check_output(
            ['git', 'branch', '--show-current'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        branch = 'unknown'
    
    return commit, branch


def get_rust_version() -> str:
    """获取 Rust 版本"""
    import subprocess
    
    try:
        version = subprocess.check_output(
            ['rustc', '--version'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        version = 'unknown'
    
    return version


def generate_markdown_report(
    baseline_data: Dict[str, BenchmarkResult],
    current_data: Dict[str, BenchmarkResult],
    output_file: str
) -> None:
    """生成 Markdown 格式的性能对比报告"""
    
    compared = compare_benchmarks(baseline_data, current_data)
    summary = calculate_summary(compared)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    commit, branch = get_git_info()
    rust_version = get_rust_version()
    
    report = f"""# 性能基准测试对比报告

**生成时间**: {timestamp}
**Git Commit**: {commit}
**Branch**: {branch}
**Rust Version**: {rust_version}

## 摘要

| 指标 | 数值 |
|------|------|
| 总测试数 | {summary.get('total_benchmarks', 0)} |
| 平均变化 | {summary.get('avg_change_percent', 0):.2f}% |
| 性能回归数 | {summary.get('regression_count', 0)} |
| 性能提升数 | {summary.get('improvement_count', 0)} |
| 最大回归 | {summary.get('max_regression', 0):.2f}% |
| 最大提升 | {summary.get('max_improvement', 0):.2f}% |

## 详细对比

| 测试项 | 基线 (ns) | 当前 (ns) | 变化 | 状态 |
|--------|-----------|-----------|------|------|
"""
    
    # 按回归程度排序
    sorted_benchmarks = sorted(
        compared.items(),
        key=lambda x: x[1].change_percent if x[1].change_percent is not None else float('inf'),
        reverse=True
    )
    
    for name, result in sorted_benchmarks:
        baseline_time = baseline_data.get(name, BenchmarkResult(name, 0)).time_ns
        current_time = result.time_ns
        change = result.change_percent
        
        if change is None:
            change_str = "N/A"
            status = "🆕 新增"
        elif change > 10:
            change_str = f"+{change:.2f}%"
            status = "🔴 回归"
        elif change < -10:
            change_str = f"{change:.2f}%"
            status = "🟢 提升"
        else:
            change_str = f"{change:+.2f}%"
            status = "✅ 稳定"
        
        report += f"| {name} | {baseline_time:.2f} | {current_time:.2f} | {change_str} | {status} |\n"
    
    report += f"""

## 性能回归检测

**阈值**: 变化超过 10% 视为性能回归

"""
    
    regressions = [
        (name, result) for name, result in compared.items()
        if result.regression
    ]
    
    if regressions:
        report += "### 检测到的性能回归\n\n"
        for name, result in regressions:
            report += f"- **{name}**: {result.change_percent:+.2f}%\n"
    else:
        report += "✅ 未检测到性能回归\n"
    
    improvements = [
        (name, result) for name, result in compared.items()
        if result.change_percent is not None and result.change_percent < -10
    ]
    
    if improvements:
        report += "\n### 显著的性能提升\n\n"
        for name, result in improvements:
            report += f"- **{name}**: {result.change_percent:+.2f}%\n"
    
    report += f"""

## 建议

"""
    
    if summary.get('regression_count', 0) > 0:
        report += f"""⚠️ 检测到 {summary['regression_count']} 项性能回归，建议：

1. 检查最近的代码变更
2. 分析回归最严重的测试项
3. 考虑优化或回退相关改动
"""
    else:
        report += """✅ 性能表现良好，无显著回归

可以继续当前的开发节奏，保持性能监控。
"""
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已生成：{output_file}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n用法:")
        print("  python analyze_benchmarks.py <criterion_output.txt>  # 解析单个输出")
        print("  python analyze_benchmarks.py <baseline.txt> <current.txt> [output.md]  # 对比")
        sys.exit(1)
    
    report_dir = Path("benchmark_reports")
    report_dir.mkdir(exist_ok=True)
    
    if len(sys.argv) == 2:
        # 解析单个文件
        output_file = sys.argv[1]
        benchmarks = parse_criterion_output(output_file)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": get_git_info()[0],
            "git_branch": get_git_info()[1],
            "rust_version": get_rust_version(),
            "benchmarks": {
                name: asdict(result) for name, result in benchmarks.items()
            }
        }
        
        output_json = report_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"解析完成，结果保存至：{output_json}")
        print(f"共解析 {len(benchmarks)} 个基准测试")
    
    elif len(sys.argv) >= 3:
        # 对比两个文件
        baseline_file = sys.argv[1]
        current_file = sys.argv[2]
        output_md = sys.argv[3] if len(sys.argv) > 3 else report_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        baseline_data = parse_criterion_output(baseline_file)
        current_data = parse_criterion_output(current_file)
        
        generate_markdown_report(baseline_data, current_data, str(output_md))
        print(f"对比完成，报告保存至：{output_md}")


if __name__ == '__main__':
    main()
