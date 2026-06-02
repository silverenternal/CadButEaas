# Web UI 迁移文档索引

**版本**: v0.1.0
**创建日期**: 2026 年 3 月 21 日
**状态**: 规划阶段

---

## 📚 文档概览

本系列文档规划将当前项目的 **egui 前端**迁移到**现代化 Web UI**，采用最先进的技术栈，实现浏览器原生的 CAD 几何智能处理系统。

---

## 📋 文档列表

| 文档 | 说明 | 目标读者 | 状态 |
|------|------|----------|------|
| [web-ui-migration-plan.md](web-ui-migration-plan.md) | 迁移规划总览、技术栈选型、架构设计 | 项目经理、架构师、开发者 | ✅ 完成 |
| [web-ui-component-spec.md](web-ui-component-spec.md) | 组件设计规范、UI/UX 设计、主题定制 | 前端开发者、UI 设计师 | ✅ 完成 |
| [web-ui-api-integration.md](web-ui-api-integration.md) | API 集成规范、WebSocket、错误处理 | 后端开发者、前端开发者 | ✅ 完成 |
| **web-ui-index.md** (本文档) | 文档索引、快速导航 | 所有读者 | ✅ 完成 |

---

## 🚀 快速开始

### 我想了解迁移计划
→ 阅读 [web-ui-migration-plan.md](web-ui-migration-plan.md)

**内容**:
- 项目概述和迁移背景
- 技术栈选型（React + Vite + Tailwind + shadcn/ui）
- 系统架构设计
- 核心功能映射（egui → Web）
- 迁移计划和里程碑
- 风险与缓解措施

---

### 我想开发组件
→ 阅读 [web-ui-component-spec.md](web-ui-component-spec.md)

**内容**:
- 设计原则和组件分类
- 基础组件规范（Button、Dialog、Select 等）
- 业务组件规范（CanvasViewer、LayerPanel 等）
- 工具栏和通知系统规范
- 主题定制和响应式设计
- 可访问性规范

---

### 我想集成 API
→ 阅读 [web-ui-api-integration.md](web-ui-api-integration.md)

**内容**:
- API 客户端封装
- 类型定义和 Schema 验证
- TanStack Query 集成
- WebSocket 客户端实现
- 错误处理策略
- 测试方案

---

## 🎯 核心技术栈

```
┌─────────────────────────────────────────────────────────────┐
│                      技术栈全景图                            │
├─────────────────────────────────────────────────────────────┤
│  框架层                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   React 18  │  │ TypeScript  │  │    Vite 5   │         │
│  │  并发渲染    │  │  类型安全    │  │  极速构建    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  UI 层                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Tailwind   │  │ shadcn/ui   │  │   Radix     │         │
│  │  原子化 CSS  │  │  组件库      │  │  无头组件    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  状态管理                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Zustand    │  │ TanStack    │  │  React      │         │
│  │  轻量状态    │  │  Query      │  │  Router     │         │
│  │             │  │  数据缓存    │  │  路由        │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  图形渲染                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Konva.js   │  │  Three.js   │  │  React      │         │
│  │  2D Canvas  │  │  WebGL/3D   │  │  Konva      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  工程化                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Biome     │  │   Vitest    │  │ Playwright  │         │
│  │  代码质量    │  │  单元测试    │  │  E2E 测试     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 项目结构

```
cad-web/
├── docs/                     # 项目文档
│   ├── web-ui-index.md       # 本文档
│   ├── web-ui-migration-plan.md
│   ├── web-ui-component-spec.md
│   └── web-ui-api-integration.md
│
├── src/
│   ├── components/           # 通用组件
│   │   ├── ui/               # shadcn/ui 基础组件
│   │   ├── canvas/           # Canvas 组件
│   │   ├── toolbar/          # 工具栏组件
│   │   └── panels/           # 面板组件
│   │
│   ├── features/             # 功能模块
│   │   ├── file/             # 文件管理
│   │   ├── canvas/           # Canvas 功能
│   │   ├── interaction/      # 交互功能
│   │   └── analysis/         # 分析功能
│   │
│   ├── stores/               # 状态管理
│   ├── hooks/                # 自定义 Hooks
│   ├── services/             # API 服务
│   ├── lib/                  # 工具库
│   └── types/                # 类型定义
│
├── tests/                    # 测试文件
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── package.json
```

---

## 🗺️ 迁移路线图

```
Week 1-2: [████████████] P0 基础架构
  ├── 项目初始化
  ├── 技术栈集成
  ├── 基础组件开发
  └── 部署流程打通

Week 3-5: [████████████] P1 核心功能
  ├── Canvas 渲染
  ├── 交互功能
  └── MVP 可演示

Week 6-7: [████████████] P2 增强功能
  ├── 性能优化
  ├── 实时协作
  └── 测试覆盖

Week 8:   [████████████] P3 Polish
  ├── UI 打磨
  ├── 文档完善
  └── 正式发布
```

---

## 📊 功能对照表

| 功能模块 | egui 实现 | Web 实现 | 状态 |
|----------|----------|---------|------|
| **文件操作** | | | |
| 打开文件 | `rfd::FileDialog` | `<input type="file">` + Drag & Drop | 📋 规划 |
| 导出场景 | `rfd::FileDialog` | 浏览器下载 API | 📋 规划 |
| **Canvas 渲染** | | | |
| 线段绘制 | egui Painter | Konva.js / WebGL | 📋 规划 |
| 缩放/平移 | egui Response | Konva Transformer | 📋 规划 |
| 点选边 | 射线检测 | Konva hit detection | 📋 规划 |
| 高亮追踪 | 自定义绘制 | Konva Layer + Effects | 📋 规划 |
| **交互功能** | | | |
| 选边追踪 | HTTP/WebSocket | WebSocket + TanStack Query | 📋 规划 |
| 缺口检测 | HTTP/WebSocket | WebSocket + SSE | 📋 规划 |
| 语义标注 | ComboBox | shadcn/ui Select | 📋 规划 |
| 图层管理 | 自定义面板 | shadcn/ui Accordion | 📋 规划 |
| **视觉效果** | | | |
| 深色模式 | 自定义主题 | Tailwind dark: | 📋 规划 |
| 动画过渡 | 无 | Framer Motion | 📋 规划 |
| **通知系统** | | | |
| Toast 通知 | 自定义 | sonner / react-hot-toast | 📋 规划 |

---

## 🎓 学习资源

### 官方文档

1. [React 18](https://react.dev)
2. [TypeScript](https://www.typescriptlang.org)
3. [Vite](https://vitejs.dev)
4. [Tailwind CSS](https://tailwindcss.com)
5. [shadcn/ui](https://ui.shadcn.com)
6. [TanStack Query](https://tanstack.com/query)
7. [React Konva](https://github.com/konva-dev/react-konva)

### 教程和指南

1. [Build a CAD App in the Browser](https://example.com) - 待创建
2. [WebGL for CAD Rendering](https://example.com) - 待创建
3. [Real-time Collaboration with WebSocket](https://example.com) - 待创建

### 参考项目

1. [Excalidraw](https://excalidraw.com) - 虚拟白板
2. [tldraw](https://tldraw.com) - 轻量级绘图工具
3. [CodeSandbox](https://codesandbox.io) - Web IDE
4. [Figma](https://figma.com) - 设计协作工具

---

## 🔧 开发工具

### 必备工具

| 工具 | 用途 | 安装 |
|------|------|------|
| Node.js 20+ | 运行时环境 | [下载](https://nodejs.org) |
| pnpm 9+ | 包管理器 | `npm i -g pnpm` |
| VS Code | 编辑器 | [下载](https://code.visualstudio.com) |

### VS Code 扩展

| 扩展 | 用途 |
|------|------|
| ESLint | 代码检查 |
| Prettier | 代码格式化 |
| Tailwind CSS IntelliSense | Tailwind 智能提示 |
| TypeScript Hero | TS 导入优化 |
| Error Lens | 错误高亮 |

### 浏览器扩展

| 扩展 | 用途 |
|------|------|
| React Developer Tools | React 调试 |
| Redux DevTools | 状态调试 |

---

## 📝 开发规范

### 代码风格

```bash
# 安装依赖
pnpm install

# 格式化代码
pnpm run format

# 检查代码
pnpm run lint

# 运行测试
pnpm run test
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

### Git 工作流

```bash
# 创建功能分支
git checkout -b feature/canvas-zoom

# 开发完成后提交
git add .
git commit -m "feat: 添加 Canvas 缩放功能"

# 推送到远程
git push origin feature/canvas-zoom

# 创建 Pull Request
```

---

## 🤝 团队协作

### 角色和职责

| 角色 | 职责 | 负责人 |
|------|------|--------|
| 项目经理 | 进度跟踪、资源协调 | 待定 |
| 架构师 | 技术选型、架构设计 | 待定 |
| 前端开发 | 组件开发、功能实现 | 待定 |
| UI 设计师 | 视觉设计、交互设计 | 待定 |
| 测试工程师 | 测试用例、质量保证 | 待定 |

### 沟通渠道

- **每日站会**: 上午 10:00
- **周会**: 周一上午 11:00
- **技术评审**: 每两周一次
- **即时通讯**: [待定]

---

## 📈 进度跟踪

### 里程碑

| 里程碑 | 目标日期 | 状态 |
|--------|----------|------|
| P0 完成 | Week 2 | 📋 待开始 |
| P1 完成 | Week 5 | 📋 待开始 |
| P2 完成 | Week 7 | 📋 待开始 |
| P3 完成 | Week 8 | 📋 待开始 |
| 正式发布 | Week 8 | 📋 待开始 |

### 燃尽图

```
待办事项 ──────────────────────────────────────────
         │
         │  ████████████████████████████████
         │  ████████████████████████████████
         │  ████████████████████████████████
         │  ████████████████████████████████
         │  ████████████████████████████████
         └───────────────────────────────────────
         W1    W2    W3    W4    W5    W6    W7    W8
```

---

## ❓ 常见问题

### Q: 为什么选择 React 而不是 Vue?

A: React 拥有最大的生态系统和人才储备，对于复杂的 CAD 应用来说是最稳妥的选择。

### Q: 为什么使用 Konva 而不是 PixiJS?

A: Konva 提供了更好的 Canvas 2D API 封装，支持图层管理和事件处理，更适合 CAD 场景。

### Q: 如何保证性能？

A: 通过虚拟滚动、LOD 动态选择、Web Worker 和 WebAssembly 等技术，确保万条边流畅渲染。

### Q: 是否支持离线使用？

A: 计划支持 PWA，可以离线使用基础功能。

---

## 📞 联系方式

- **项目地址**: https://github.com/your-org/cad
- **问题反馈**: https://github.com/your-org/cad/issues
- **团队邮箱**: cad-team@example.com

---

**创建者**: CAD 团队
**审核者**: 
**最后更新**: 2026 年 3 月 21 日
