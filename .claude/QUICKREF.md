# 焚诀工作流 — 快速参考

> 本项目使用阿里云 qwen3.6-plus 模型，通过 Claude Code CLI 接入。

## 最简用法

在项目目录下运行：

```bash
# 方式一：用 cadeaas 脚本（推荐）
bin/cadeaas          # 普通会话，自动加载 burn 插件
bin/cadeaas burn     # 直接触发焚诀循环
bin/cadeaas dev      # 开发端（slave agent）
bin/cadeaas review   # 审查端（reviewer agent）

# 方式二：加 zsh alias（加到 ~/.zshrc）
cadeaas() { claude --plugin-dir /home/hugo/codes/CadButEaas/.claude/plugins/burn-workflow "$@" }
```

## 标准焚诀流程

### 一键触发（推荐）
```bash
bin/cadeaas burn
# 或进入交互会话后输入 /burn
```

### 手动分步
```bash
# 终端1 - 开发端
bin/cadeaas dev

# 终端2 - 审查端
bin/cadeaas review
```

## 指令模板

### 开发端
```
了解这个项目。读取 CLAUDE.md、todo.md、ARCHITECTURE.md。
```
```
读取 todo.md，按优先级顺序接取下一个待执行任务，严格执行。
需要研究时调用 researcher agent，完成时调用 tester agent。
```

### 审查端
```
了解当前项目。给出锐评和建议，更新 todo.md。
```
```
验证 todo.md 中任务的完成状态，更新 todo.md，整合删除冗余文档。
```

## Agent 角色

| Agent | 用途 |
|-------|------|
| **slave** | 开发执行：代码实现、bug修复、重构 |
| **reviewer** | 审查规划：代码审查、质量评估、任务规划 |
| **researcher** | 代码探索：只读不写，精确定位 |
| **tester** | 测试验证：编写测试、运行验证 |

## 何时 /compact

context 使用超过 50% 时运行。

## 何时停止

- todo.md 中所有任务标记完成
- 或你手动指定停止
