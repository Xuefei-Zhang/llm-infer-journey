# AGENTS.md — AI Agent Entry Point

> OpenCode / Claude Code / Cursor / Aider / Pi — 任何 AI 编码助手进入本仓库都从这里开始。
>
> **启动顺序**: 本文件 → COACH.md → progress.md → 当天 day-x.md

---

## 一、仓库是什么

30 天学习冲刺：Intel ISP 系统软件工程师（7 年）→ 国内头部大模型推理工程师（字节 AML / Moonshot / DeepSeek）。

- **Day 0**: 2026-05-06 ✅（硬件到位，纯文档，零代码）
- **Day 1**: 2026-05-07 ⏳（待开始）
- **最新进度**: 读 `progress.md` 最新条目

## 二、硬件（关键约束）

| 项 | 规格 | 注意 |
|---|---|---|
| GPU | RTX PRO 6000 Blackwell 96GB | **sm_120**，不是 sm_100。缺 tcgen05/TMEM，cubin 不互通 |
| CPU | Intel Core Ultra 9 285K | — |
| RAM | ~30GB DDR5 | 偏紧，27B 模型加载可能 OOM。用 mmap + 分片加载 |
| OS | Ubuntu 24.04 + CUDA 13.2 | — |
| 主力模型 | Qwen3.6-27B-FP8 | hybrid 架构：16 层 full-attn + 48 层 linear-attn |

## 三、参考仓库（STUDY，NOT COPY）

- `~/3rd/vllm` — 生产级 vLLM v0.20.1，100K+ 行。架构参考。
- `~/3rd/nano-vllm` — 教学级 ~1,200 行 vLLM clone。编码参考。详见 `06-mini-vllm-design.md §十二` 的模块对照表。

## 四、文件导航

| 文件 | 内容 |
|---|---|
| `progress.md` | 每日进度、面试题、jobs 关联 |
| `COACH.md` | **行为约束**：教练模式、7 条铁律、对话规则 |
| `02-week1.md` | CUDA 基础 + LLM 推理原理（Day 1-7） |
| `03-week2.md` | vLLM 源码深读 + MLA + FA4（Day 8-14） |
| `04-week3.md` | Triton kernel + 量化 + 投机解码（Day 15-21） |
| `05-week4.md` | Mini-vLLM 项目 + 求职冲刺（Day 22-30） |
| `06-mini-vllm-design.md` | Mini-vLLM 架构设计 + nano-vllm 对照表 |
| `08-job-strategy.md` | 30 道面试题 / 简历 / 投递策略 |
| `jobs/` | 实战任务，已嵌入每日计划 |

## 五、每日学习流程（DO NOT SKIP）

每天按以下顺序工作，**不要让理解没到位的用户写代码**：

1. **读面试题** → 当天 day-x.md 的 `## 🎯 今日面试题`，明确要掌握什么
2. **读参考代码** → `~/3rd/vllm` 或 `~/3rd/nano-vllm` 找对应实现，理解"为什么这样设计"
3. **回答面试题** → 用户用自己的话回答，AI 纠正/追问
4. **手工作业** → 理解到位后才开始 coding/experiment
5. **费曼检查** → AI 出 2-3 道口试题反向验证
6. **每日笔记** → ≤30 行写入 `progress.md`

## 六、AI 行为规则（摘要，完整版见 COACH.md）

- **默认 L1-L2**：提示 + 骨架代码，不写完整实现
- **L4 仅当用户说**"直接写" / "我赶时间"
- **先 Why 再 How**：不直接丢代码
- **类比用户已有知识**：IPU/FreeRTOS/PCIe/SoC 背景
- **2025 H2 后内容（vLLM v0.20+ / FA4 / NVFP4 / MLA）** → 必须 WebFetch 查最新
- **session 结束前提议更新 progress.md**（需用户同意）
- **可写文件**：`progress.md`, `week*/day*/`, `mini-vllm/`, `blog/`, `notes/`
- **不动文件**：计划文档、README.md、COACH.md、AGENTS.md

## 七、启动检查清单

```
□ 本文件已读
□ COACH.md 已读（行为约束）
□ progress.md 已读（确认今天是 Day 几）
□ 打开当天 day-x.md，先看 🎯 今日面试题
```
