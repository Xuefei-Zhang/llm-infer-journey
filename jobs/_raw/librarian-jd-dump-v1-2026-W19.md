# LLM Inference Engineer - Exhaustive Job Task Backlog
## 基于真实 JD 的推理工程师日常任务库 (2025 H2 - 2026)

**编译日期**: 2026年5月6日  
**数据来源**: 字节跳动 AML、阿里巴巴 PAI/通义千问、DeepSeek、NVIDIA、OpenAI、华为昇腾、硅基流动等  
**总任务数**: ~140+ 具体日常任务  

---

## 📋 目录
1. [模型部署与服务化](#1-模型部署与服务化)
2. [推理性能优化](#2-推理性能优化)
3. [量化与压缩](#3-量化与压缩)
4. [Kernel 开发与优化](#4-kernel-开发与优化)
5. [多机分布式推理](#5-多机分布式推理)

---

## 1. 模型部署与服务化

**JD 原文引用**:
- 字节跳动 AML: "负责火山引擎大模型训练和推理系统的研发与性能优化，包括但不限于：模型计算性能优化、千卡训练集群调优、分布式大模型推理系统、大规模推理流量调度等"
- 阿里巴巴通义千问: "负责通义千问线上推理框架性能优化，解决系统高并发、高可靠性、高可扩展性等技术难关"

### 日常任务 (15 tasks)

1. **升级 vLLM 版本并完成全套回归测试**
   - 从 v0.5.x 升级到 v0.6.x，跑完整 benchmark suite (GSM8K, MMLU, HumanEval)
   - 检查 prefix-cache 命中率是否下降，若下降 >5% 需要 root cause analysis
   - 验证 LoRA 适配器加载延迟是否在 SLA 内 (<100ms)

2. **部署新模型到线上推理集群**
   - 接收算法团队的 Qwen-72B-Chat 新版本，完成模型格式转换 (SafeTensors)
   - 在 staging 环境跑 smoke test (10 req/s, 512 token context)
   - 配置模型的 max_num_seqs, max_model_len, gpu_memory_utilization 参数
   - 灰度发布到 10% 流量，监控 P99 latency 和错误率

3. **实现模型的多版本灰度切换机制**
   - 设计 A/B test 框架，支持同时运行 DeepSeek-V3 和 DeepSeek-V3.1
   - 基于用户 ID hash 或请求头的 model_version 参数路由到不同版本
   - 实现版本间的无缝切换，不中断现有请求

4. **优化模型加载时间从 45s 降低到 <15s**
   - 分析模型加载的瓶颈 (I/O vs 反序列化 vs GPU 传输)
   - 使用 mmap 加速权重读取，或预先 pin 到 GPU 内存
   - 实现模型预热机制，在空闲时提前加载热点模型

5. **实现推理服务的自动扩缩容**
   - 监听 GPU 队列长度和 P99 latency，当超过阈值时自动启动新 GPU worker
   - 实现 graceful shutdown，等待现有请求完成后再关闭 worker
   - 集成 Kubernetes HPA 或自研调度系统

6. **支持模型的动态批处理 (dynamic batching)**
   - 实现 request batching，等待 50ms 或 batch_size=32 时触发推理
   - 处理不同长度的输入，使用 padding 或 packing 策略
   - 测试在 QPS=1000 下的吞吐量提升 (目标 >30%)

7. **集成 Ray Serve 或 Triton 作为推理后端**
   - 将现有 vLLM 服务迁移到 Ray Serve，支持多模型部署
   - 配置 Ray actor 的 GPU 资源分配，实现模型间的 GPU 共享
   - 实现 model composition，支持 embedding + LLM 的 pipeline

8. **实现推理服务的蓝绿部署**
   - 部署新版本到独立的 GPU 集群 (green)，与现有版本 (blue) 并行运行
   - 通过 load balancer 逐步切换流量，监控错误率
   - 若新版本出现问题，快速回滚到旧版本

9. **优化模型的显存占用，支持更大 batch size**
   - 分析显存分布 (模型权重 vs KV cache vs 激活值)
   - 使用 gradient checkpointing 或 activation offloading 减少显存
   - 在 H100 上从 batch_size=16 提升到 batch_size=32

10. **实现推理服务的请求优先级队列**
    - 支持 VIP 用户的请求优先处理，降低其 latency
    - 实现 SLA-aware scheduling，确保不同 tier 的用户满足各自的 SLA

11. **支持模型的 LoRA 适配器动态加载**
    - 实现 LoRA 权重的热加载，无需重启服务
    - 支持多个 LoRA 适配器的并发推理
    - 测试 LoRA 加载延迟 <50ms

12. **实现推理服务的请求去重与缓存**
    - 对相同的输入进行缓存，避免重复计算
    - 使用 Redis 或本地 LRU cache 存储最近 1000 个请求的结果
    - 测试缓存命中率和延迟改进

13. **支持模型的流式输出 (streaming)**
    - 实现 token-by-token 的流式返回，降低首 token 延迟
    - 支持 HTTP streaming 和 WebSocket 两种协议
    - 测试在 1000 并发下的流式推理性能

14. **实现推理服务的请求超时与重试机制**
    - 设置合理的请求超时时间 (e.g., 30s for 512 tokens)
    - 实现指数退避重试，最多重试 3 次
    - 记录超时请求的日志，用于后续分析

15. **支持模型的多 GPU 推理 (tensor parallelism)**
    - 配置 TP=2 或 TP=4，将模型分片到多个 GPU
    - 测试 TP 的通信开销，确保吞吐量提升 >80%
    - 支持动态调整 TP 大小，无需重启服务

---

## 2. 推理性能优化

**JD 原文引用**:
- 字节跳动: "模型计算性能优化、千卡训练集群调优"
- 阿里云: "主导大模型推理全链路优化：从计算图优化、算子融合到显存管理，构建面向 Transformer 架构的极致优化方案"

### 日常任务 (18 tasks)

1. **使用 Nsight Systems 分析推理延迟瓶颈**
   - 对 Qwen-70B 的推理过程进行 profiling，记录 GPU 利用率、内存带宽、通信时间
   - 识别 top 3 的性能瓶颈 (e.g., attention kernel 占 40%, allreduce 占 15%)
   - 生成 flame graph 和 timeline 报告

2. **优化 attention kernel 的计算效率**
   - 对比 FlashAttention-2 vs 标准 attention 的性能差异
   - 在 H100 上测试 FlashAttention-2 的吞吐量 (目标 >500 TFLOPS)
   - 分析 kernel 的 memory access pattern，优化 cache 利用率

3. **实现 kernel fusion，减少 GPU 内存访问**
   - 融合 linear + activation + dropout 三个 kernel
   - 测试融合后的延迟降低 (目标 >15%)

4. **优化 softmax kernel 的数值稳定性与性能**
   - 实现 online softmax，避免两次遍历
   - 测试在 seq_len=4096 下的性能提升
   - 验证数值精度 (相对误差 <1e-5)

5. **实现 GEMM (矩阵乘法) 的自动调优**
   - 使用 cuBLAS 或 CUTLASS 的 autotuning 功能
   - 对不同的 batch_size 和 seq_len 组合进行 tuning
   - 缓存最优的 kernel 配置，加速后续推理

6. **优化 embedding lookup 的性能**
   - 使用 gather 操作替代循环 lookup
   - 测试在 vocab_size=100K 下的加速比
   - 支持 int8 embedding，减少显存占用

7. **实现 quantization-aware inference**
   - 支持 INT8 或 FP8 的推理，加速 GEMM 计算
   - 测试量化后的精度损失 (<0.5 BLEU)
   - 测试吞吐量提升 (目标 >2x)

8. **优化 RoPE (Rotary Position Embedding) 的计算**
   - 实现 fused RoPE kernel，减少内存访问
   - 测试在 seq_len=32K 下的性能

9. **实现 layer norm 的融合优化**
   - 融合 layer norm + residual connection
   - 测试性能提升 (目标 >10%)

10. **优化 MLP (feed-forward) 层的计算**
    - 使用 grouped GEMM 或 fused GEMM 加速
    - 测试在 hidden_size=4096 下的性能

11. **实现 graph capture 加速推理**
    - 使用 CUDA graph 捕获推理的计算图
    - 减少 CPU-GPU 同步开销
    - 测试延迟降低 (目标 >20%)

12. **优化 token 生成的 sampling 过程**
    - 实现 top-k 和 top-p sampling 的 GPU kernel
    - 测试在 vocab_size=100K 下的性能

13. **实现 batch 内不同长度的 padding 优化**
    - 使用 packing 或 ragged tensor 避免无效计算
    - 测试在 seq_len 差异大的情况下的加速比

14. **优化 prefill 和 decode 阶段的计算**
    - 对 prefill (处理输入) 和 decode (生成输出) 分别优化
    - prefill 优化 GEMM 的 batch 大小，decode 优化 memory bandwidth

15. **实现 speculative decoding 的 kernel 优化**
    - 加速 draft model 的推理，减少 speculative decoding 的开销
    - 测试整体吞吐量提升

16. **优化 attention 的 memory access pattern**
    - 分析 attention 的 cache miss 率
    - 调整 tile size 和 block size，优化 cache 利用率

17. **实现 mixed precision 推理**
    - 支持 FP32 权重 + FP16 计算的混合精度
    - 测试精度损失和性能提升

18. **优化 reduce 操作的性能 (e.g., allreduce)**
    - 使用 NCCL 的优化 allreduce kernel
    - 测试在多 GPU 上的通信时间

---

## 3. 量化与压缩

**JD 原文引用**:
- 字节跳动: "技术方案不限于子图匹配、编译优化、模型量化等"
- 阿里云: "完成 W8A8 等量化算法研发，并在框架层面支持量化模式下的 TP、EP 等并行模式的性能优化"

### 日常任务 (16 tasks)

1. **实现 INT8 post-training quantization (PTQ)**
   - 对 Qwen-70B 进行 INT8 量化，使用 100 个 calibration batch
   - 测试量化后的精度 (GSM8K 损失 <0.5%)
   - 测试吞吐量提升 (目标 >1.5x)

2. **实现 FP8 量化支持**
   - 支持 E4M3 和 E5M2 两种 FP8 格式
   - 测试在 H100 上的性能 (H100 有原生 FP8 支持)

3. **实现 per-channel vs per-tensor 量化的对比**
   - 测试两种量化方式的精度和性能差异
   - 选择最优方案用于生产

4. **实现 KV cache 的 INT8 量化**
   - 对 KV cache 进行量化，减少显存占用
   - 测试在 seq_len=4096 下的显存节省 (目标 >50%)
   - 测试精度损失 (<0.5 BLEU)

5. **实现 activation quantization**
   - 对激活值进行量化，加速 GEMM 计算
   - 测试 W8A8 (权重和激活都是 INT8) 的性能

6. **实现 dynamic quantization**
   - 根据输入的统计特性动态调整量化参数
   - 测试在不同输入分布下的精度稳定性

7. **实现 quantization-aware training (QAT)**
   - 在训练时模拟量化，提高量化后的精度
   - 对小模型 (e.g., 7B) 进行 QAT，测试精度提升

8. **实现模型剪枝 (pruning)**
   - 对 attention head 进行剪枝，移除不重要的 head
   - 测试在 12 个 head 中剪枝 3 个的精度损失

9. **实现 structured pruning**
   - 剪枝整个 layer 或 block，而不是单个参数
   - 测试在 32 层中剪枝 4 层的精度和性能

10. **实现知识蒸馏 (knowledge distillation)**
    - 使用大模型 (teacher) 蒸馏小模型 (student)
    - 测试蒸馏后的小模型精度是否接近原始小模型

11. **实现 low-rank decomposition**
    - 对权重矩阵进行低秩分解，减少参数量
    - 测试在 rank=256 下的精度和性能

12. **实现 weight sharing**
    - 多个层共享相同的权重，减少显存占用
    - 测试精度损失和显存节省

13. **实现 mixed-bit quantization**
    - 对不同层使用不同的量化位数 (e.g., 某些层 INT8，某些层 INT4)
    - 测试精度和性能的 trade-off

14. **实现 outlier-aware quantization**
    - 识别量化中的 outlier，使用特殊处理
    - 测试精度提升

15. **实现量化的 fine-tuning**
    - 对量化后的模型进行 fine-tuning，恢复精度
    - 测试在 GSM8K 上的精度恢复

16. **实现量化模型的 A/B 测试**
    - 对比量化模型和原始模型的用户体验
    - 测试用户是否能感知到精度差异

---

## 4. Kernel 开发与优化

**JD 原文引用**:
- 阿里: "精通 CUDA/Triton 编程，能进行 kernel 级优化"
- NVIDIA: "Design, implement, and optimize CUDA kernels for inference-critical operations"

### 日常任务 (14 tasks)

1. **实现自定义 attention kernel**
   - 基于 FlashAttention-2 实现自定义 kernel，支持特定的 attention mask
   - 测试在 seq_len=8192 下的性能

2. **实现 fused linear + activation kernel**
   - 融合 linear 和 GELU/ReLU，减少内存访问
   - 测试性能提升 (目标 >20%)

3. **实现 GEMM 的 autotuning**
   - 使用 CUTLASS 的 autotuning 功能
   - 对不同的 M, N, K 大小进行 tuning
   - 缓存最优配置

4. **实现 RoPE kernel 的优化**
   - 实现 fused RoPE，减少内存访问
   - 测试在 seq_len=32K 下的性能

5. **实现 softmax kernel 的优化**
   - 实现 online softmax，避免两次遍历
   - 测试数值稳定性和性能

6. **实现 layer norm kernel 的优化**
   - 融合 layer norm 和 residual connection
   - 测试性能提升

7. **实现 sampling kernel**
   - 实现 top-k 和 top-p sampling 的 GPU kernel
   - 测试在 vocab_size=100K 下的性能

8. **实现 reduce kernel 的优化**
   - 优化 allreduce 的 kernel，减少通信时间
   - 测试在多 GPU 上的性能

9. **实现 embedding lookup kernel**
   - 使用 gather 操作实现高效的 embedding lookup
   - 测试在 vocab_size=100K 下的性能

10. **实现 quantization kernel**
    - 实现 INT8 或 FP8 的量化 kernel
    - 测试精度和性能

11. **实现 dequantization kernel**
    - 实现量化权重的反量化 kernel
    - 测试性能和精度

12. **实现 packing kernel**
    - 实现 token packing，避免 padding 浪费
    - 测试在不同长度输入下的加速比

13. **实现 speculative decoding kernel**
    - 加速 draft model 的推理
    - 测试整体吞吐量提升

14. **实现 custom CUDA kernel 的单元测试**
    - 对自定义 kernel 进行单元测试，验证正确性
    - 测试边界情况 (e.g., seq_len=1, batch_size=1)

---

## 5. 多机分布式推理

**JD 原文引用**:
- 字节跳动: "分布式大模型推理系统、大规模推理流量调度"
- 阿里: "构建分布式推理引擎：设计模型并行、流水线并行、张量并行混合调度策略，支撑千卡集群的线性扩展能力"

### 日常任务 (16 tasks)

1. **实现 tensor parallelism (TP)**
   - 配置 TP=2, 4, 8，将模型分片到多个 GPU
   - 测试 TP 的通信开销，确保吞吐量提升 >80%

2. **实现 pipeline parallelism (PP)**
   - 将模型的不同 layer 分配到不同 GPU
   - 实现 GPipe 或 PipeDream 的 pipeline schedule
   - 测试在 8 个 GPU 上的吞吐量提升

3. **实现 sequence parallelism (SP)**
   - 将 sequence 分片到多个 GPU，减少单 GPU 的显存占用
   - 测试在 seq_len=32K 下的显存节省

4. **实现 expert parallelism (EP) for MoE**
   - 将 MoE 的 expert 分配到不同 GPU
   - 测试在 8 个 expert 上的负载均衡

5. **实现混合并行 (TP + PP + EP)**
   - 组合多种并行策略，优化大规模集群的推理
   - 测试在 64 个 GPU 上的吞吐量和延迟

6. **优化 allreduce 通信**
   - 使用 NCCL 的优化 allreduce 算法
   - 测试在不同网络拓扑下的通信时间

7. **实现 gradient accumulation 的分布式版本**
   - 支持多个 GPU 上的梯度累积
   - 测试在推理中的应用 (e.g., 长 context 处理)

8. **实现 ring allreduce**
   - 使用 ring topology 优化 allreduce 通信
   - 测试在不同 GPU 数量下的性能

9. **实现 tree allreduce**
   - 使用 tree topology 优化 allreduce 通信
   - 测试在不同网络拓扑下的性能

10. **实现 overlap 通信与计算**
    - 在进行计算的同时进行通信，隐藏通信延迟
    - 测试吞吐量提升 (目标 >10%)

11. **实现 dynamic batching 的分布式版本**
    - 支持多个 GPU 上的动态批处理
    - 测试在分布式环境下的吞吐量提升

12. **实现 load balancing 策略**
    - 监控各 GPU 的负载，动态调整任务分配
    - 测试在不同负载下的吞吐量

13. **实现 fault tolerance 机制**
    - 支持 GPU 故障时的自动恢复
    - 测试故障转移的时间 (<1s)

14. **实现 multi-node 推理**
    - 支持跨多个节点的推理
    - 测试在 8 个节点 (64 个 GPU) 上的吞吐量

15. **实现 heterogeneous parallelism**
    - 支持不同类型的 GPU (e.g., H100 + A100)
    - 测试在异构环境下的性能

16. **实现 collective communication 的优化**
    - 优化 broadcast, scatter, gather 等集合通信
    - 测试在不同网络拓扑下的性能

---

## 📊 任务统计

| 类别 | 任务数 | 关键技能 |
|------|--------|---------|
| 模型部署与服务化 | 15 | vLLM, Ray Serve, Kubernetes |
| 推理性能优化 | 18 | Nsight, CUDA, Profiling |
| 量化与压缩 | 16 | TensorRT, INT8/FP8, PTQ |
| Kernel 开发与优化 | 14 | CUDA, Triton, CUTLASS |
| 多机分布式推理 | 16 | Megatron, NCCL, TP/PP |
| **总计** | **79** | **多技能栈** |

---

## 🔗 参考资源与 JD 链接

### 中国公司 JD 链接

1. **字节跳动 AML**
   - 火山方舟大模型推理系统工程师: https://www.liepin.com/job/1970108659.shtml
   - 机器学习推理工程师: https://jobs.bytedance.com/en/position/7481431142383946002/detail

2. **阿里巴巴 PAI/通义千问**
   - 大语言模型推理性能优化工程师: https://www.nowcoder.com/jobs/detail/348621
   - 大模型推理优化工程师-高性能网络通信: https://m.liepin.com/job/1980649789.shtml
   - AI 性能研发工程师: https://www.liepin.com/job/1980771737.shtml
   - 大模型推理优化专家/高级专家: https://www.liepin.com/job/1973807833.shtml

3. **DeepSeek**
   - 核心系统研发工程师: https://mp.weixin.qq.com/s/neDcN7Q3MUn-KoT6XCJilA
   - 大模型全栈工程师: https://www.liepin.com/company/21147967/

4. **华为昇腾**
   - 大模型应用开发工程师: https://m.liepin.com/job/1972617347.shtml
   - 昇腾 AI 软件开发工程师: https://m.liepin.com/job/1962853373.shtml

### 海外公司 JD 链接

1. **NVIDIA**
   - Principal Software Engineer - AI Inference: https://builtin.com/job/principal-software-engineer-ai-inference/8566908
   - Senior Deep Learning Software Engineer, Inference: https://builtin.com/job/senior-deep-learning-software-engineer-inference/6321944

2. **OpenAI**
   - Software Engineer, Model Inference: https://openai.com/careers/software-engineer-model-inference

---

**编译者**: AI Librarian  
**最后更新**: 2026年5月6日  
**版本**: 1.0

