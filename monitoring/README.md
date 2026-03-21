# CAD EaaS 架构 - 监控仪表板说明

## 📊 快速启动

### 1. 启动监控栈

```bash
# 进入监控目录
cd monitoring

# 启动所有服务（Prometheus + Grafana + Node Exporter）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 2. 访问服务

| 服务 | URL | 账号/密码 |
|------|-----|-----------|
| Grafana | http://localhost:3000 | admin / admin123 |
| Prometheus | http://localhost:9090 | 无 |
| Node Exporter | http://localhost:9100 | 无 |

---

## 📈 Grafana 仪表板面板说明

### 面板 1: 阶段吞吐量 (5 分钟平均)
- **指标**: `rate(stage_completed_total[5m])`
- **用途**: 实时监控各阶段的处理速率
- **单位**: 请求/秒 (reqps)

### 面板 2: 阶段耗时分布 (P50/P95/P99)
- **指标**: 
  - `histogram_quantile(0.99, rate(stage_duration_seconds_bucket[5m]))` - P99
  - `histogram_quantile(0.95, rate(stage_duration_seconds_bucket[5m]))` - P95
  - `histogram_quantile(0.50, rate(stage_duration_seconds_bucket[5m]))` - P50
- **用途**: 分析阶段性能瓶颈，识别长尾延迟
- **告警阈值**: P99 > 1s 黄色，> 2s 红色

### 面板 3: 阶段超时率 (5 分钟平均)
- **指标**: `rate(stage_timeout_total[5m]) / (rate(stage_completed_total[5m]) + rate(stage_timeout_total[5m]) + rate(stage_cancel_total[5m]))`
- **用途**: 监控不稳定的阶段
- **告警阈值**: > 5% 黄色，> 10% 红色

### 面板 4: 阶段取消率 (5 分钟平均)
- **指标**: `rate(stage_cancel_total[5m]) / (rate(stage_completed_total[5m]) + rate(stage_timeout_total[5m]) + rate(stage_cancel_total[5m]))`
- **用途**: 监控快速失败频率，识别级联故障
- **告警阈值**: > 10% 黄色，> 20% 红色

### 面板 5-8: 统计卡片
- **阶段总完成数**: 累计完成的阶段执行次数
- **阶段总超时数**: 累计超时次数
- **阶段总取消数**: 累计取消次数
- **阶段平均耗时**: 5 分钟内的平均执行耗时

---

## 🚨 Prometheus 告警规则

### 告警列表

| 告警名称 | 触发条件 | 严重级别 |
|----------|----------|----------|
| HighStageTimeoutRate | 超时率 > 10% 持续 2 分钟 | Warning |
| HighStageCancelRate | 取消率 > 20% 持续 2 分钟 | Warning |
| HighStageLatencyP99 | P99 耗时 > 2s 持续 5 分钟 | Warning |
| StageStopped | 5 分钟内无完成请求 | Critical |
| ZeroCopyPerformanceDegradation | 零拷贝性能退化 > 100μs | Warning |

### 告警配置位置
- 文件：`monitoring/prometheus/alerts.yml`

---

## 🔧 自定义配置

### 修改 Prometheus 采集间隔

编辑 `prometheus/prometheus.yml`:
```yaml
global:
  scrape_interval: 5s  # 修改此值
```

### 修改 Grafana 登录密码

编辑 `docker-compose.yml`:
```yaml
environment:
  - GF_SECURITY_ADMIN_PASSWORD=your_new_password
```

### 添加新的仪表板

1. 在 Grafana UI 中创建仪表板
2. 导出为 JSON
3. 保存到 `grafana/` 目录
4. 重启 Grafana 或重新加载配置

---

## 📊 Prometheus 指标说明

### Counter 类型（只增不减）

| 指标名称 | 说明 | Labels |
|----------|------|--------|
| `stage_completed_total` | 阶段成功完成次数 | `stage_name` |
| `stage_timeout_total` | 阶段超时次数 | `stage_name` |
| `stage_cancel_total` | 阶段取消次数 | `stage_name` |

### Histogram 类型（直方图）

| 指标名称 | 说明 | Labels |
|----------|------|--------|
| `stage_duration_seconds` | 阶段执行耗时（秒） | `stage_name` |

### 常用查询示例

```promql
# 查询各阶段的完成速率
rate(stage_completed_total[5m])

# 查询 P99 耗时
histogram_quantile(0.99, rate(stage_duration_seconds_bucket[5m]))

# 查询超时率
rate(stage_timeout_total[5m]) / rate(stage_completed_total[5m])

# 查询平均耗时
rate(stage_duration_seconds_sum[5m]) / rate(stage_duration_seconds_count[5m])
```

---

## 🐛 故障排查

### Prometheus 无法采集数据

```bash
# 检查 Prometheus 日志
docker logs cad-prometheus

# 验证目标状态
curl http://localhost:9090/api/v1/targets
```

### Grafana 无法显示数据

1. 检查数据源配置：Configuration → Data Sources → Prometheus
2. 确认 URL 为 `http://cad-prometheus:9090`
3. 点击 "Save & Test"

### 指标未暴露

确保 Rust 应用已启动指标暴露：
```rust
// 在 main.rs 或 lib.rs 中
use prometheus::TextEncoder;

// 启动 HTTP 服务器暴露 /metrics 端点
```

---

## 📚 参考文档

- [Prometheus 官方文档](https://prometheus.io/docs/)
- [Grafana 官方文档](https://grafana.com/docs/)
- [Prometheus Rust 客户端](https://docs.rs/prometheus/0.13.0/prometheus/)
