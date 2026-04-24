# 焚诀工作流 — CadButEaas 开发流程

> 你也发现了，其实AI生成的文档、代码的质量完全依赖于提示词以及我们给出指令的顺序（也就相当于我们手搓了一个工作流）。
>
> — Hugo Lee 2026.4.10

## 核心思想

AI 的输出质量 = 提示词质量 × 指令顺序。本工作流通过**一键触发 + 自动循环**，将项目质量稳步推高。

## 启动方式

### 一键触发焚诀循环（推荐）

```bash
bin/cadeaas burn          # 进入焚诀循环，自动审查→开发→审查
bin/cadeaas               # 普通会话（加载 burn 插件，可手动 /burn）
```

### 分步模式

```bash
bin/cadeaas dev           # 开发端（slave agent）
bin/cadeaas review        # 审查端（reviewer agent）
```

### 添加 alias（可选）

```bash
# 加到 ~/.zshrc
cadeaas() { claude --plugin-dir /home/hugo/codes/CadButEaas/.claude/plugins/burn-workflow "$@" }
```

## 焚诀循环流程

```
/burn 触发
    │
    ▼
Phase 0: 读取 todo.md + CLAUDE.md + ARCHITECTURE.md
    │
    ▼
Phase 1: 审查（Reviewer 角色）
    │  启动 reviewer 子 agent → 全面审查 → 输出锐评 + 优先级任务
    │
    ▼
Phase 2: 开发（Developer 角色）
    │  对每个优先级任务启动 developer 子 agent → 实现 → 测试 → 格式化
    │
    ▼
Phase 3: 更新 todo.md
    │  标记完成、调整优先级、删除冗余
    │
    ▼
Phase 4: 报告结果 + 询问是否继续
    │  用户确认 → 回到 Phase 1
    │  用户拒绝 → 停止
```

## 子 agent 角色

| Agent | 用途 | 可用工具 |
|-------|------|----------|
| **slave** | 开发执行 | Read, Write, Edit, Bash, Glob, Grep, Task*, Agent |
| **reviewer** | 审查规划 | Read, Grep, Glob, Bash, Task*, Agent |
| **researcher** | 代码探索（只读） | Read, Glob, Grep, Bash, LSP |
| **tester** | 测试验证 | Read, Write, Edit, Bash, Glob, Grep |

## 关键约定

1. **不砍文档**: 力求项目追平文档声明
2. **数据说话**: 所有评价必须有代码/测试/数据支撑
3. **零警告**: Rust clippy 必须 0 警告，biome 必须 0 错误
4. **测试不破**: 220+ 测试不容许破坏
5. **先读后改**: 修改前必须先读取理解上下文
6. **不加戏**: 不添加需求之外的功能、注释、错误处理

## 文件结构

```
.claude/
├── agents.json              ← Agent 定义（CLI --agents 参数使用）
├── QUICKREF.md              ← 快速参考
├── WORKFLOW.md              ← 本文件
├── agents/                  ← Agent 参考文档（冗余备份）
│   ├── developer.md
│   ├── reviewer.md
│   ├── researcher.md
│   └── tester.md
├── plugins/
│   └── burn-workflow/       ← 焚诀工作流插件
│       ├── .claude-plugin/
│       │   └── plugin.json
│       └── commands/
│           └── burn.md      ← /burn 命令定义
└── skills/
    └── burn/
        └── SKILL.md         ← 焚诀技能定义（冗余）
```

## Agent 定义文件

`.claude/agents.json` 包含 4 个 agent 的定义：
- **slave**: 开发执行端
- **reviewer**: 审查端
- **researcher**: 代码探索
- **tester**: 测试验证

通过 `--agents` 参数加载，配合 `--agent` 参数指定当前角色。
