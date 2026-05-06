# jobs/ — 真实大厂 LLM 推理工程师工作模拟 backlog

> 30 天计划是**必修课**。这里是**选修 + 课后实战题**,模拟入职后的日常 backlog。
> 让你提前进入"工程师在干什么"的语境,而不是"学生在学什么"。

## 怎么用

1. 读 `jobs_taxonomy.md`,理解大厂推理工程师的能力地图(15 个工作域)。
2. 读 `jobs_backlog.md`,挑你 30 天计划当天的"自由时间"想做的任务。
3. 认领一条 → 在 `tasks/` 目录复制 `_template.md` → 命名 `T-XXX-<slug>.md` → 开干。
4. 每周一,我跑一次"雷达扫描" → 输出到 `weekly-radar/<ISO-week>.md`,新发现的任务由你点头才进 backlog。
5. 真出事故/故意制造事故 → `postmortems/` 写五问复盘。

## 与 30 天主计划的关系

| 阶段 | 主计划状态 | jobs/ 用法 |
|------|-----------|-----------|
| Day 1-7 (Week 1) | 课程紧,8h 满载 | 不碰 jobs/,先把 Week 1 走完 |
| Day 8-21 (Week 2-3) | 课程 + 实战 | 周末选 1 条 L1-L2 任务热身 |
| Day 22-28 (Week 4) | mini-vLLM 项目 | 不碰,集中精力 |
| Day 29-30 (Week 5 头) | 总结 + 投递 | 用 backlog 完成度回答面试题 |
| Day 31+ (Week 5+ 空窗) | 主活动 = jobs/ | 每周 3-5 条,持续做到入职 |

## 任务难度

- **L1**: 1-3h,纯调试/配置,产出 1 个 progress 记录
- **L2**: 4-8h,跨 1-2 个工具,需要 1 张 profiling 图 + 1 个数据表
- **L3**: 1-3 天,跨 3+ 工具,需要博客文 + benchmark + 至少 1 处源码改动

## 文件清单

| 文件 | 用途 | 谁更新 |
|------|------|--------|
| `README.md` | 本文档 | 极少更新 |
| `jobs_taxonomy.md` | 15 个工作域定义 + 公司映射 | 月度 |
| `jobs_backlog.md` | 主任务列表(`[ ]` checkbox) | 周度 |
| `tasks/_template.md` | 单任务记录模板 | 极少 |
| `tasks/T-XXX-*.md` | 你认领的具体任务日志 | 你做时 |
| `weekly-radar/<week>.md` | 每周搜出的新任务 | 周一 |
| `postmortems/<slug>.md` | 事故复盘 | 触发时 |
| `_raw/` | librarian agent 原始转储,**不直接看** | 归档 |

## 规则

- 这里所有内容**不能违反 COACH.md §四 4.1**(01-09 主计划文档不动)。
- 任务**不能引入计划之外的硬件依赖**(没有云 GPU、没有多机)。
- 单条任务**必须能在 PRO 6000 96GB 单机本地复现**,否则归类 `[lab-only-impossible]`。
- 来源**必须真实**(JD URL / blog URL / GitHub issue URL),不允许编造。

## 当前版本

- v0.1 (2026-05-06): 骨架建立,backlog 基于 1 个 librarian agent 真实素材池(`_raw/librarian-jd-dump-2026-W19.md`,196 条候选)初步精选 30-50 条进 v0.1 backlog。
