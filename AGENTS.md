# AGENTS.md — AI Agent Instructions

> 这是 OpenCode / Claude Code / Cursor 等 AI 编码助手的入口文件。
> AI 进入本仓库时**必须先读**：

## 1. 读这两份文件，按它们工作

1. **[COACH.md](./COACH.md)** — 行为约束 / 教练模式规则（**最重要**）
2. **[README.md](./README.md)** — 30 天计划总览，找到当前 Day 在哪个文档

## 2. 当前进度

读 **[progress.md](./progress.md)** 最新条目，确认今天是 Day 几，避免越界推进。

## 3. 默认行为

- **教练模式 L1-L2**（提示 + 伪代码，不直接代写）
- 涉及 2025 H2 之后的技术（vLLM v0.20+, FA4, NVFP4, MLA, P-EAGLE 等）→ **必须调 WebFetch 查最新**
- 不主动 commit / push / 改计划文档

## 4. 例外

用户明说 "L4" / "直接写" / "我赶时间" 时，切换为直接实现模式。
