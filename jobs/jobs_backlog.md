# jobs_backlog.md — v0.1

> 精选 30 条任务,全部能在你的 PRO 6000 96GB 单机上完成。
> 来源:`_raw/librarian-jd-dump-2026-W19.md`(196 条候选)+ 我的二次筛选(去掉云依赖、去掉多机依赖、去掉空泛任务)。
> 字段:`[ ] T-XXX [Lx] <标题>` → 域 / 估时 / Done definition / 来源 / 关联 Day。

## 怎么挑

任务已直接嵌入每日计划，无需单独挑选。在对应 Day 的 `0{week}-week*.md` 中查找 `[jobs]` 标记即可。

| 任务 | 嵌入位置 | 触发 Day |
|---|---|---|
| T-001 (vLLM 冷启动) | 02-week1.md Day 6 | Day 6 |
| T-002 (bench TTFT vs TPS) | 02-week1.md Day 6 | Day 6 |
| T-003 (gpu-memory 调优) | 02-week1.md Day 6 | Day 6 |
| T-004 (prefix caching 对比) | 03-week2.md Day 11 | Day 11 |
| T-005 (Nsight profiling) | 03-week2.md Day 11 | Day 11 |
| T-010 (MTP 投机解码) | 04-week3.md Day 20 | Day 20 |
| T-015 (Prometheus/Grafana) | 05-week4.md Day 26 | Day 26 |
| T-041 (故意 OOM 复盘) | 03-week2.md Day 13 | Day 13 |

---

## L1 区 — 调试 / 配置 / 1-3h(快速热身用)

- [ ] **T-001 [L1] 跑通 vLLM serve Qwen3.6-27B-FP8,记录冷启动时间分布**
  - 域:1 部署
  - 估时:2h
  - Done:`uv run vllm serve` 启动成功,记录 ckpt 加载耗时、CUDA graph capture 耗时、warmup 耗时三段,贴进 progress.md
  - 来源:字节 AML JD「冷启动优化」+ 主计划 Day 13
  - 关联:Day 13(可直接当 Day 13 产出)

- [ ] **T-002 [L1] 用 vllm bench serve 压一组 (concurrency 1/4/8/16),画 TTFT vs TPS 曲线**
  - 域:2 优化
  - 估时:3h
  - Done:1 张 PNG(matplotlib),concurrency 增加后的拐点写出来
  - 来源:vLLM 官方 bench 文档
  - 关联:Day 13-14

- [ ] **T-003 [L1] 把 `--gpu-memory-utilization` 从 0.85 调到 0.95,观察 max_num_seqs 与 OOM 概率**
  - 域:1 部署
  - 估时:2h
  - Done:3 个值各跑 200 个请求,记录 OOM 发生数 + 实际 KV pages 占用
  - 来源:vLLM 配置文档常见 issue
  - 关联:Day 14

- [ ] **T-004 [L1] 启用 `--enable-prefix-caching`,对比命中率高/低两种 workload 的 TTFT 差**
  - 域:2 优化
  - 估时:3h
  - Done:构造 2 组 prompt(共享前缀 vs 完全独立),报 prefix cache hit rate + TTFT
  - 来源:vLLM blog 多次提及
  - 关联:Day 14

- [ ] **T-005 [L1] 用 `nvidia-smi dmon -s pucvmet` + `nsys profile` 抓一次 prefill+decode 全过程**
  - 域:2 优化
  - 估时:3h
  - Done:1 张 nsys 时间线截图,标出 prefill/decode/sampling/idle 各占比
  - 来源:Nsight 官方手册 + 主计划 Day 11
  - 关联:Day 11-12

## L2 区 — 跨工具 / 4-8h(实战级)

- [ ] **T-010 [L2] 给 Qwen3.6-27B-FP8 启用 `--speculative-config` MTP,实测加速比与精度漂移**
  - 域:7 投机解码
  - 估时:6h
  - Done:对比 baseline / MTP-on,跑 humaneval 子集 50 题,acceptance rate + speedup + pass@1 差
  - 来源:Moonshot Kimi blog + 主计划 Day 17 已经预演
  - 关联:Day 17

- [ ] **T-011 [L2] 做 FP8 KV cache 精度兜底实验**
  - 域:8 长 context
  - 估时:6h
  - Done:在 8K / 32K / 128K context 下,FP16 KV vs FP8 KV 输出 BLEU/ROUGE 对比 + KV 显存节省数字
  - 来源:阿里 JD「长 context 显存优化」
  - 关联:Day 17-18

- [ ] **T-012 [L2] vLLM 升级:把当前 v0.20.x 升到 v0.21.x(或最新 patch),跑全套回归 + diff 配置变更**
  - 域:11 框架
  - 估时:6h
  - Done:升级前后 TTFT/TPS 不下降 5% 以上;若有 regression,定位 root cause 并写 progress
  - 来源:每个推理工程师每月都干一次
  - 关联:Day 13 部署完之后

- [ ] **T-013 [L2] 用 `torch.compile` mode='reduce-overhead' 包一段自定义 attention,跑前后对比**
  - 域:4 kernel
  - 估时:6h
  - Done:1 个最小可复现 .py + benchmark 表 + 编译日志中观察到的 graph break 列表
  - 来源:PyTorch 官方 + GPU MODE
  - 关联:Day 15-16(Triton 之后)

- [ ] **T-014 [L2] chunked-prefill token budget 自适应:写一个 watchdog 调整 `--max-num-batched-tokens`**
  - 域:2 优化 + 8 长 context
  - 估时:8h
  - Done:写一个 `monitor.py`,根据 GPU 利用率 + 等待队列长度,自动调 budget,配 1 张曲线图
  - 来源:vLLM blog 2025 chunked-prefill 系列
  - 关联:Day 14 之后

- [ ] **T-015 [L2] 给 vLLM 服务加 Prometheus 指标暴露,Grafana 出 1 张看板**
  - 域:12 监控
  - 估时:6h
  - Done:监控含 TTFT P50/P99 / running req / waiting req / KV usage / TPS / OOM count;screenshot 进 progress
  - 来源:每家公司必备
  - 关联:Week 5+ 选修

- [ ] **T-016 [L2] 用 `py-spy` 抓 vLLM scheduler 一次 OOM 之前的 30s,定位 hot path**
  - 域:13 oncall
  - 估时:5h
  - Done:py-spy svg + 5 行总结,把 hot 函数与 vLLM scheduler 源码对应
  - 来源:vLLM issue 区高频问题
  - 关联:Week 5+

- [ ] **T-017 [L2] 测试 `--enforce-eager` 关闭后(启用 CUDA Graph)的延迟改善 + capture 失败排查路径**
  - 域:11 框架 + 13 oncall
  - 估时:5h
  - Done:2 组数据(eager / graph),如果 graph capture 报错,记录错误信息 + 解决思路
  - 来源:vLLM 高频 issue
  - 关联:Day 14

- [ ] **T-018 [L2] structured output(JSON / regex)对 TPS 的影响实测**
  - 域:2 优化
  - 估时:5h
  - Done:同一 prompt,自由生成 vs guided JSON,TPS 衰减比例 + decode 路径里多了哪一步
  - 来源:SGLang / outlines 文档
  - 关联:Week 5+

- [ ] **T-019 [L2] 给 mini-vLLM 项目接一个 OpenAI 兼容的 streaming API**
  - 域:1 部署
  - 估时:6h
  - Done:可以用 `openai` SDK 直接调用,SSE 流式返回正确;附 1 个 curl 测试日志
  - 来源:主计划 Day 22-28 mini-vLLM 项目
  - 关联:Day 22-28

## L3 区 — 1-3 天(项目级,Week 5+ 才碰)

- [ ] **T-030 [L3] 从 vLLM 找一个 `good first issue`,提一个 PR(从 fork 到 merge 全流程)**
  - 域:11 框架 + 15 协作
  - 估时:2-3 天
  - Done:PR URL + reviewer 反馈处理记录,无论 merge 与否都进 progress
  - 来源:GitHub vllm-project/vllm
  - 关联:Week 5+

- [ ] **T-031 [L3] 复现 1 篇 2025 H2 推理论文(EAGLE-3 / Medusa-2 / Lookahead 任选)的最小 demo**
  - 域:14 论文
  - 估时:3 天
  - Done:1 篇 progress,含 paper 摘要 + 代码 + 1 张实测对比图 + 评估能否上生产
  - 来源:arxiv 月度扫描
  - 关联:Week 5+

- [ ] **T-032 [L3] 在单机用 TP=2 模拟"P/D 分离":2 个 vLLM 实例(一个只 prefill 一个只 decode),NVLink 传 KV**
  - 域:6 P/D 分离
  - 估时:3 天
  - Done:写一个 router,measure 双实例方案 vs 单实例的 throughput 差;如果不可行(单卡 96GB 跑两个 27B 装不下),用 Qwen2.5-7B
  - 来源:NVIDIA TRT-LLM / vLLM disaggregated serving 文档
  - 关联:Week 5+

- [ ] **T-033 [L3] 给 mini-vLLM 加 prefix cache(radix tree),benchmark vs 无 cache**
  - 域:4 kernel + 8 长 context
  - 估时:3 天
  - Done:radix tree 实现 + 命中率统计 + benchmark 表
  - 来源:SGLang RadixAttention paper + 主计划 mini-vLLM 扩展
  - 关联:Day 22-28 之后

## L2 区(W19 雷达 librarian B 增量)

> 来源:`_raw/librarian-blogs-dump-2026-W19.md`(60 个候选 story)。
> ⚠️ 60 story 中很多公司归属与博客不可证实(如"Anthropic LatencySpikeIncident"、"OpenAI PostgreSQL Logging"等无对应公开博客),按 COACH §七 已舍弃。
> 仅保留 URL/主题在 vLLM/SGLang/DeepSeek/FlashAttention 公开仓库可证实的项,共 **5 条**新任务:

- [ ] **T-050 [L2] 读 vLLM v0.20 release notes,把 PagedAttention 升级点列出来,并在你的部署上验证 1 个**
  - 域:11 框架
  - 估时:5h
  - Done:release notes 摘要 + 1 项升级点(如新增配置项/API 改动)在你环境实测的 before/after 数据
  - 来源:https://github.com/vllm-project/vllm/releases (W20 用 gh CLI 实证哪个 tag 是 v0.20)
  - 关联:T-001 / T-012 之后

- [ ] **T-051 [L2] 读 DeepSeek-V3 / R1 inference report,挑 1 个工程优化点(如 MTP / FP8 / EP routing)在 Qwen3.6 上能复用的尝试**
  - 域:7 投机解码 + 9 MoE
  - 估时:8h
  - Done:1 篇 progress 记录论文哪一节 + 你的复用方案 + 1 张 benchmark 对比
  - 来源:DeepSeek arxiv + GitHub deepseek-ai/DeepSeek-V3
  - 关联:Day 17 (MTP) / Week 5+

- [ ] **T-052 [L2] FlashAttention release tag 演进史:从 FA2→FA3→FA4 的 API/dtype 支持变化梳理(只读 README + release notes)**
  - 域:4 kernel
  - 估时:4h
  - Done:1 张 markdown 对照表,标出哪个 FA 版本支持你的 sm_120 + FP8/NVFP4
  - 来源:https://github.com/Dao-AILab/flash-attention/releases
  - 关联:Day 15-16 / Day 17-18 (NVFP4)

- [ ] **T-053 [L2] SGLang RadixAttention 论文 + 仓库 README 精读,与 vLLM prefix-cache 做对照**
  - 域:8 长 context + 11 框架
  - 估时:5h
  - Done:1 篇 progress,列 5 个差异点(数据结构/淘汰策略/lookup 复杂度/兼容性等)
  - 来源:SGLang 仓库 + arxiv RadixAttention paper
  - 关联:T-004 之后

- [ ] **T-054 [L2] 给本地部署画一张"成本-延迟-精度"三维取舍表,5 种配置(eager/cuda-graph/MTP/FP8-KV/Spec)各打分**
  - 域:12 监控 + 2 优化
  - 估时:5h
  - Done:1 张表 + 推荐"日常 / 高峰 / 长 context"3 种 SLO 场景的最优配置
  - 来源:综合 vLLM/SGLang blog 的 trade-off 思路
  - 关联:Week 3 全部完成后

## 灵活 / 触发型

- [ ] **T-040 [L1] 每周一花 30min 浏览 vLLM blog + DeepSeek/Moonshot 工程博客上周新文章,挑 1 篇精读**
  - 域:14 论文
  - 估时:每周 1h
  - Done:在 `weekly-radar/` 写一段 200 字摘要
  - 关联:Week 5+ 起每周

- [ ] **T-041 [L2] 故意制造一次 OOM(把 `--gpu-memory-utilization 0.99` + 高并发),完整复盘**
  - 域:13 oncall
  - 估时:4h
  - Done:在 `postmortems/` 写五问复盘(发现-止损-定位-修复-预防)
  - 关联:Week 5+

- [ ] **T-042 [L2] 做一次 silent 模型输出漂移检测:同 prompt 同 seed,vLLM 升级前后输出 token-by-token diff**
  - 域:13 oncall + 11 框架
  - 估时:5h
  - Done:diff 脚本 + 50 个 prompt 的样本对比,有差就定位到哪一层(sampling / kernel / config)
  - 来源:vLLM 历史 regression 多次出现
  - 关联:T-012 之后

---

## 完成统计(自动维护)

- 当前 backlog 总数:**29 条**(原 24 + W19 雷达 librarian B 增量 5)
- L1: 8 / L2: 17 / L3: 4
- 已完成:0
- 进行中:0

## 版本

- v0.1(2026-05-06 早):初稿,librarian A(JD)真实素材池筛选 24 条。
- v0.2(2026-05-06 晚):W19 雷达 librarian B(engineering blogs)增量 5 条(T-050~T-054),60 候选中 55 条因 URL/归属不可证实被舍弃。
- 待补:已弃 bg_481cf1bb(GitHub issues)的内容,在 W20 用 `gh` CLI 手补。
