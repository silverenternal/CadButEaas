# CAD Web 前端 - Claude Code 配置

## 技术栈

- **框架**: React 18 + TypeScript
- **构建**: Vite 5
- **状态管理**: Zustand + React Query
- **3D 渲染**: Three.js + React Three Fiber + Drei
- **2D 渲染**: Konva + react-konva
- **UI 组件**: Radix UI + Tailwind CSS + lucide-react
- **表单**: React Hook Form + Zod
- **Linting/格式化**: Biome
- **测试**: Vitest + Testing Library + Playwright (E2E)
- **组件文档**: Storybook 8

## 开发命令

```bash
cd cad-web

# 安装依赖
pnpm install

# 开发服务器
pnpm dev

# 构建
pnpm build

# Lint 检查
pnpm lint

# Lint 自动修复
pnpm lint:fix

# 格式化
pnpm format

# 单元测试
pnpm test

# E2E 测试
pnpm test:e2e

# Storybook
pnpm storybook
```

## 代码风格

- 组件使用 PascalCase，文件名 kebab-case
- Hooks 和工具函数使用 camelCase
- 优先函数组件，不使用 class 组件
- 使用 Tailwind 类名，不要创建自定义 CSS（除非绝对必要）
- 组件导出使用 barrel files (index.ts)

## 项目结构

```
cad-web/src/
├── components/
│   ├── cad-viewer/      # CAD 查看器核心组件
│   ├── layout/          # 布局组件
│   ├── panels/          # 侧边栏面板（图层/属性/缺口检测）
│   ├── toolbar/         # 工具栏
│   └── ui/              # 基础 UI 组件（Radix UI 封装）
├── App.tsx              # 应用入口
└── assets/styles/       # 全局样式
```

## 注意事项

- cad-simple-viewer 是第三方 CAD 渲染库，不要修改其源码
- Three.js 渲染在 React 中通过 @react-three/fiber 管理
- WebSocket 连接到后端服务（默认 localhost:3000）
- 使用 biome 而非 eslint/prettier
