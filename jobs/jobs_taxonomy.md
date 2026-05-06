# LLM 推理工程师能力地图(15 个工作域)

> 来源:1 个 librarian agent 扫描 2025 H2 - 2026 真实 JD(字节 AML、阿里 PAI、DeepSeek、华为昇腾、NVIDIA、OpenAI 等)。
> 原始素材池:`_raw/librarian-jd-dump-2026-W19.md`(196 条候选任务)。

每个域都有:**它做什么** / **典型 JD 引用** / **公司频次**(★ = 出现频率)/ **本仓库哪些 Day 触及**。

---

## 1. 模型部署与服务化(Serving)

- **做什么**:把训练好的 ckpt 跑成 HTTP/gRPC 服务,处理流量、多模型路由、灰度发布、容量规划。
- **典型 JD 引用**(字节 AML):"负责火山引擎大模型推理系统的研发与性能优化,包括分布式大模型推理系统、大规模推理流量调度。"
- **公司频次**:★★★★★(每家都有)
- **本仓库**:Week 2 Day 13-14(vLLM serve)、Day 9(局域网部署)。

## 2. 推理性能优化(Latency/Throughput)

- **做什么**:profile → 找瓶颈 → 改 batch/dtype/chunk/调度 → 压回归 → 灰度。日常 70% 时间。
- **典型 JD 引用**(阿里通义千问):"负责大模型推理性能优化,涵盖 attention、MoE、量化、长 context 等关键算子。"
- **公司频次**:★★★★★
- **本仓库**:Week 2 Day 11-12(profiling)、Day 13-14(吞吐压测)。

## 3. 量化与压缩(Quant/Compression)

- **做什么**:FP8/INT4/NVFP4 部署,精度保护,kernel 兼容性验证。
- **典型 JD 引用**(NVIDIA):"Develop and optimize quantization techniques (W8A8, FP8, NVFP4) for production LLM serving."
- **公司频次**:★★★★★
- **本仓库**:Week 3 Day 17-18(W4A8/FP8/NVFP4)。

## 4. Kernel 开发(CUDA/Triton/CUTLASS)

- **做什么**:写或改 attention/GEMM/sampling kernel,适配新 SM,benchmark 对齐 cuBLAS/FA。
- **典型 JD 引用**(DeepSeek):"系统研发工程师,要求 CUDA/GPU 经验,能编写或优化高性能算子。"
- **公司频次**:★★★★(对学历/经验门槛高)
- **本仓库**:Week 3 Day 15-16(Triton attention 入门)、Week 4 mini-vLLM 部分 kernel。

## 5. 多机分布式推理(TP/PP/EP/SP)

- **做什么**:NCCL 拓扑、TP/PP/EP 切分、alltoall 调优、容灾。
- **典型 JD 引用**(字节 AML):"千卡集群调优、弹性调度。"
- **公司频次**:★★★★★(大模型团队必备)
- **本仓库**:Week 3 Day 19-20(理论 + 单机 TP=2 模拟)。**真实多机不在本仓库**。

## 6. P/D 分离 & KV 路由(Disaggregated Serving)

- **做什么**:Prefill/Decode 拆开部署,KV 通过 NVLink/RDMA 传输,提高 throughput。
- **典型 JD 引用**(NVIDIA):"prefill-decode separation, KV cache transfer optimization."
- **公司频次**:★★★★(2025 H2 起爆发)
- **本仓库**:Day 20 概念介绍。深度实现属 jobs/ 选修。

## 7. 投机解码(Speculative Decoding / EAGLE / MTP)

- **做什么**:部署 draft model,调 verify step,精度 vs 加速 tradeoff。
- **典型 JD 引用**(Moonshot 公开博客):"K1.5 部署使用 P-EAGLE 减少 decode 延迟。"
- **公司频次**:★★★★
- **本仓库**:Week 3 Day 17(MTP 体验,Qwen3.5-MTP)。

## 8. 长 Context & KV cache 管理

- **做什么**:128K-1M context,chunked-prefill,KV 量化,内存换算法。
- **典型 JD 引用**(阿里通义千问):"长 context 场景下的 TTFT 与显存优化。"
- **公司频次**:★★★★
- **本仓库**:Week 3 Day 17(256K 实测)。

## 9. MoE 推理优化

- **做什么**:Expert 路由、负载均衡、EP 切分、激活专家缓存。
- **典型 JD 引用**(DeepSeek/字节):"DeepSeek-V3 / V4 EP 推理优化,alltoall 性能调优。"
- **公司频次**:★★★★(MoE 团队专项)
- **本仓库**:Week 3 Day 19-20 概念。**真实 MoE 部署不在本仓库**。

## 10. 多模态推理(VLM/Audio/Video)

- **做什么**:vision encoder + LLM 联合推理,token merging,kvcache 协同。
- **典型 JD 引用**(阿里):"通义千问 VL 系列推理优化。"
- **公司频次**:★★★(子团队)
- **本仓库**:Week 5+ 选修(Qwen3.6-27B 是 VL,可探)。

## 11. 框架与编译工具链

- **做什么**:vLLM/SGLang/TRT-LLM/MindIE 升级、torch.compile、CUDA Graph、TVM/MLIR。
- **典型 JD 引用**(华为昇腾):"MindIE/MindSpore 部署优化。"
- **公司频次**:★★★★
- **本仓库**:Week 2-4 全程使用 vLLM。

## 12. 监控、告警、SLO、成本

- **做什么**:Prometheus/Grafana 看板、TTFT/TPS/E2E 告警、$/Mtoken 优化。
- **典型 JD 引用**(OpenAI):"optimize FLOPs and GB GPU RAM per request."
- **公司频次**:★★★★
- **本仓库**:Week 5+ 选修。

## 13. On-Call & 故障排查

- **做什么**:NCCL hang、OOM 暴涨、cuda graph capture 失败、模型输出漂移。
- **典型 JD 引用**(字节 AML):"线上故障定位、root cause 分析、复盘文档。"
- **公司频次**:★★★★★(每个 oncall 工程师都干)
- **本仓库**:`postmortems/` 子目录专项训练。

## 14. 论文跟踪与原型实现

- **做什么**:每周扫 arxiv,选 1 篇复现 demo,评估能否上生产。
- **典型 JD 引用**(NVIDIA):"keep up with state-of-the-art inference research, prototype promising ideas."
- **公司频次**:★★★(senior 责任)
- **本仓库**:Week 5+ 选修,与 `weekly-radar/` 联动。

## 15. 跨团队协作

- **做什么**:与算法对齐精度、与产品定 SLO、与硬件团队 debug 驱动 bug、code review。
- **典型 JD 引用**:几乎所有 JD 末尾都有"良好沟通能力"。
- **公司频次**:★★★★★
- **本仓库**:无单独练习,通过 PR/issue 参与开源时积累。

---

## 公司画像速查

| 公司 | 主流框架 | 主力硬件 | 招聘最热 3 个域 |
|------|---------|---------|----------------|
| 字节 AML / 火山 | vLLM 改 + 自研 | H100/H800 + 自研芯片 | 1, 2, 5 |
| 阿里 PAI / 通义 | RTP-LLM / vLLM | H100/H20/PPU | 2, 3, 8 |
| DeepSeek | 自研 + vLLM | H800 + H20 | 4, 5, 9 |
| Moonshot | vLLM 改 | H800 | 2, 7, 8 |
| 华为昇腾 | MindIE | Ascend 910B/910C | 1, 3, 11 |
| NVIDIA | TRT-LLM / vLLM | 自家全栈 | 4, 6, 11 |
| OpenAI | 自研 | Azure H100 fleet | 2, 12 |

数据来源 commit 时刻:`_raw/librarian-jd-dump-2026-W19.md`(2026-05-06 编译)。
