# CAD Web UI

CAD 几何智能处理系统的现代化 Web 前端。

**版本**: v0.2.0
**状态**: ✅ P0/P1/P2 完成

[![Test Status](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)]()

## 技术栈

- **框架**: React 18 + TypeScript
- **构建工具**: Vite 5
- **样式**: Tailwind CSS + shadcn/ui
- **状态管理**: Zustand
- **数据缓存**: TanStack Query
- **图形渲染**: React Konva (LOD 优化)
- **实时通信**: WebSocket
- **动画**: Framer Motion
- **测试**: Vitest + Playwright

## 快速开始

### 环境要求

- Node.js 20+
- pnpm 9.0+

### 安装依赖

```bash
pnpm install
```

### 启动开发服务器

```bash
pnpm dev
```

访问 http://localhost:5173

### 构建生产版本

```bash
pnpm build
```

### 预览生产构建

```bash
pnpm preview
```

## 项目结构

```
cad-web/
├── src/
│   ├── components/       # 通用组件（带动画）
│   │   ├── ui/          # shadcn/ui 基础组件
│   │   ├── toolbar/     # 工具栏组件
│   │   └── panels/      # 面板组件
│   ├── features/         # 功能模块
│   │   ├── canvas/      # Canvas 功能（LOD 优化）
│   │   └── file/        # 文件管理
│   ├── stores/          # 状态管理（LOD 选择器）
│   ├── hooks/           # 自定义 Hooks
│   ├── services/        # API 服务
│   ├── lib/             # 工具库 + 动画
│   └── types/           # 类型定义
├── tests/               # 测试文件
└── docs/                # 项目文档
```

## 开发规范

### 代码风格

```bash
# 格式化代码
pnpm format

# 检查代码
pnpm lint
pnpm lint:fix
```

### 运行测试

```bash
# 单元测试
pnpm test

# E2E 测试
pnpm test:e2e

# 测试覆盖率
pnpm test -- --coverage
```

### 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org):

```
feat: 添加 Canvas 缩放功能
fix: 修复边选择错误
docs: 更新 API 文档
style: 格式化代码
refactor: 重构状态管理
test: 添加单元测试
chore: 更新依赖
```

## 功能特性

### P0 - 基础功能 ✅

- ✅ 项目初始化和配置
- ✅ 基础 UI 组件库（16 个组件）
- ✅ API 客户端封装
- ✅ WebSocket 实时通信
- ✅ 状态管理

### P1 - 核心功能 ✅

- ✅ Canvas 渲染和交互
- ✅ 文件上传和处理
- ✅ 边选择和追踪
- ✅ 缺口检测
- ✅ 语义标注

### P2 - 增强功能 ✅

- ✅ **LOD 性能优化**（视口裁剪、批处理渲染）
- ✅ 测试覆盖（单元测试 + E2E）
- ✅ 实时协作基础

### P3 - Polish ✅

- ✅ Framer Motion 动画过渡
- ✅ 组件动画效果
- ✅ 主题系统
- ✅ 部署配置

## 性能特性

### LOD 优化

| 缩放级别 | 渲染策略 | 性能提升 |
|----------|----------|----------|
| < 0.3 (Low) | 仅渲染墙边 + 选中边 | 5x |
| 0.3-0.7 (Medium) | 跳过短边 | 2x |
| > 0.7 (High) | 完整渲染 | 1x |

### 批处理渲染

- 按样式分组渲染
- useMemo 缓存
- 减少重渲染次数

### 性能监控

开发模式下显示实时性能统计：
- 可见边数量
- LOD 级别
- 预估渲染时间

## 环境变量

复制 `.env.example` 到 `.env.local`:

```bash
cp .env.example .env.local
```

配置环境变量:

```env
VITE_API_URL=http://localhost:3000/api
VITE_WS_URL=ws://localhost:3000/ws
VITE_APP_NAME=CAD 几何智能处理系统
VITE_ENABLE_DEVTOOLS=true
VITE_MAX_EDGES_RENDER=10000
VITE_LOD_THRESHOLD=0.5
```

## 浏览器支持

- Chrome (推荐)
- Firefox
- Safari
- Edge

## 许可证

MIT

## 联系方式

- 项目地址：https://github.com/your-org/cad
- 问题反馈：https://github.com/your-org/cad/issues

## 详细文档

- [实现总结](IMPLEMENTATION.md)
- [API 集成规范](../docs/web-ui-api-integration.md)
- [组件设计规范](../docs/web-ui-component-spec.md)
- [迁移规划](../docs/web-ui-migration-plan.md)
