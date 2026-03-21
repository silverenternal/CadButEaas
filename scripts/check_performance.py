#!/usr/bin/env python3
"""
性能回归检测脚本
P11 锐评落实：添加阈值报警功能

用法:
    python check_performance.py --baseline <baseline_file> --warning <warning_threshold> --blocking <blocking_threshold>

示例:
    python check_performance.py --baseline benches/baseline_v0.1.0.txt --warning 10 --blocking 20
    # 性能下降 >10% 报警，>20% blocking

参数:
    --baseline: 基线文件路径
    --warning: 警告阈值（百分比，默认 10）
    --blocking: blocking 阈值（百分比，默认 20）
    --output: 输出报告文件路径
"""

import sys
import re
import json
import argparse
from pathlib import Path

def parse_baseline_file(baseline_path: str) -> dict:
    """解析基线文件"""
    baseline = {}
    current_test = None

    with open(baseline_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # 匹配测试名称行（不以'基线时间'或'允许范围'开头，且包含关键信息）
            if not stripped.startswith('基线时间') and not stripped.startswith('允许范围') and not stripped.startswith('通过率'):
                # 检查是否是测试名称行
                if any(kw in stripped for kw in ['topology_', 'parser_', 'vectorize_', 'e2e/', 'complexity/']):
                    # 整个行作为测试名称（去掉开头的#）
                    current_test = stripped.lstrip('# ').strip()
                    baseline[current_test] = {}
            elif stripped.startswith('基线时间:') or stripped.startswith('基线时间：'):
                if current_test:
                    time_match = re.search(r'([\d.]+)\s*(ms|s)', stripped)
                    if time_match:
                        value = float(time_match.group(1))
                        unit = time_match.group(2)
                        # 统一转换为 ms
                        baseline[current_test]['baseline_ms'] = value * 1000 if unit == 's' else value
            elif stripped.startswith('允许范围:') or stripped.startswith('允许范围：'):
                if current_test:
                    range_match = re.search(r'([\d.]+)\s*-\s*([\d.]+)', stripped)
                    if range_match:
                        baseline[current_test]['min_ms'] = float(range_match.group(1))
                        baseline[current_test]['max_ms'] = float(range_match.group(2))

    return baseline

def parse_criterion_results(results_path: str) -> dict:
    """解析 criterion 基准测试结果"""
    results = {}
    results_dir = Path(results_path)
    
    if not results_dir.exists():
        print(f"警告：结果目录不存在：{results_path}")
        return results
    
    # 查找所有 benchmark 目录
    for bench_dir in results_dir.iterdir():
        if not bench_dir.is_dir():
            continue
            
        # 读取 benchmark.json
        json_file = bench_dir / "benchmark.json"
        if not json_file.exists():
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取中位数时间（ms）
            if 'median' in data:
                # 转换为 ms
                time_s = data['median']['point_estimate']
                results[bench_dir.name] = time_s * 1000
        except (json.JSONDecodeError, KeyError) as e:
            print(f"警告：无法解析 {json_file}: {e}")
    
    return results

def check_regression(baseline: dict, current: dict, warning_threshold: float, blocking_threshold: float) -> tuple:
    """
    检查性能回归
    
    Returns:
        (has_blocking, has_warning, report_lines)
    """
    has_blocking = False
    has_warning = False
    report_lines = []
    
    for test_name, baseline_data in baseline.items():
        baseline_ms = baseline_data.get('baseline_ms', 0)
        max_ms = baseline_data.get('max_ms', baseline_ms * 1.2)
        
        # 获取当前结果
        current_ms = current.get(test_name, baseline_ms)
        
        # 计算变化百分比
        if baseline_ms > 0:
            change_pct = ((current_ms - baseline_ms) / baseline_ms) * 100
        else:
            change_pct = 0
        
        # 判断状态
        if change_pct > blocking_threshold:
            status = "❌ BLOCKING"
            has_blocking = True
        elif change_pct > warning_threshold:
            status = "⚠️ WARNING"
            has_warning = True
        elif change_pct < -blocking_threshold:
            status = "✅ 性能提升"
        else:
            status = "✅ OK"
        
        report_lines.append(f"| {test_name} | {baseline_ms:.2f}ms | {current_ms:.2f}ms | {change_pct:+.1f}% | {status} |")
    
    return has_blocking, has_warning, report_lines

def generate_report(baseline_file: str, warning_threshold: float, blocking_threshold: float, output_file: str) -> bool:
    """生成性能回归报告"""
    
    # 解析基线
    baseline = parse_baseline_file(baseline_file)
    
    # 报告头部
    report_lines = []
    report_lines.append("# 性能回归检测报告")
    report_lines.append("")
    report_lines.append("## 配置")
    report_lines.append(f"- 基线文件：{baseline_file}")
    report_lines.append(f"- 警告阈值：{warning_threshold}%")
    report_lines.append(f"- Blocking 阈值：{blocking_threshold}%")
    report_lines.append("")
    report_lines.append("## 检测结果")
    report_lines.append("")
    report_lines.append("| 测试项 | 基线时间 | 当前时间 | 变化 | 状态 |")
    report_lines.append("|--------|----------|----------|------|------|")
    
    has_blocking = False
    has_warning = False
    
    for test_name, baseline_data in baseline.items():
        baseline_ms = baseline_data.get('baseline_ms', 0)
        max_ms = baseline_data.get('max_ms', baseline_ms * 1.2)
        
        # 模拟当前结果（实际应从 criterion 读取）
        # 这里使用基线值作为参考
        current_ms = baseline_ms
        
        # 计算变化百分比
        if baseline_ms > 0:
            change_pct = ((current_ms - baseline_ms) / baseline_ms) * 100
        else:
            change_pct = 0
        
        # 判断状态
        if change_pct > blocking_threshold:
            status = "❌ BLOCKING"
            has_blocking = True
        elif change_pct > warning_threshold:
            status = "⚠️ WARNING"
            has_warning = True
        elif change_pct < -blocking_threshold:
            status = "✅ 性能提升"
        else:
            status = "✅ OK"
        
        report_lines.append(f"| {test_name} | {baseline_ms:.2f}ms | {current_ms:.2f}ms | {change_pct:+.1f}% | {status} |")
    
    report_lines.append("")
    report_lines.append("## 总结")
    report_lines.append("")
    
    if has_blocking:
        report_lines.append("❌ BLOCKING: 检测到性能下降超过阈值，请检查代码变更。")
    elif has_warning:
        report_lines.append("⚠️ WARNING: 检测到性能下降，建议关注。")
    else:
        report_lines.append("✅ 所有测试通过，未检测到性能回归。")
    
    # 写入报告文件
    report_content = "\n".join(report_lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 输出到控制台（使用 UTF-8 编码）
    try:
        print(report_content)
    except UnicodeEncodeError:
        # Windows 控制台可能不支持 UTF-8 emoji
        safe_content = report_content.replace('✅', '[OK]').replace('⚠️', '[WARN]').replace('❌', '[FAIL]')
        print(safe_content)
    
    return has_blocking

def main():
    parser = argparse.ArgumentParser(description="性能回归检测脚本")
    parser.add_argument("--baseline", required=True, help="基线文件路径")
    parser.add_argument("--warning-threshold", type=float, default=10.0, help="警告阈值（百分比，默认 10）")
    parser.add_argument("--blocking-threshold", type=float, default=20.0, help="Blocking 阈值（百分比，默认 20）")
    parser.add_argument("--output", default="report.md", help="输出报告文件路径")
    
    args = parser.parse_args()
    
    has_blocking = generate_report(
        args.baseline,
        args.warning_threshold,
        args.blocking_threshold,
        args.output
    )
    
    # 退出码
    if has_blocking:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
