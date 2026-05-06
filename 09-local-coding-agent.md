# 本地 Coding Agent 部署：替代 GitHub Copilot

> **目标**：用本地 RTX PRO 6000 Blackwell 96GB + Qwen3.6-27B-FP8 跑 OpenCode / Cursor / Continue，**完全替代到期的 GitHub Copilot**，且体验达到 90-95%。
>
> **顺带做求职简历加分项**：写一篇 "本地 Coding Agent 部署调优" 博客。
>
> **当前环境**：Ubuntu 24.04 + Driver 595.58.03 + CUDA 13.2 + vLLM v0.20.1。

---

## 一、Coding Agent 的真实硬件需求

### 1.1 与"聊天 LLM"的差异

| 维度 | 聊天 LLM | Coding Agent |
|---|---|---|
| 平均 Context | 2-8K | **30-150K** |
| 峰值 Context | 32K | **256K-1M** |
| 并发 | 1 | **2-4**（多 agent / 多对话）|
| 期望 Decode TPS | ≥30 | **≥40**（流式更敏感）|
| 期望 TTFT | ≤2s | **≤1s**（agent 多次往返）|
| 模型尺寸下限 | 7B 够用 | **≥27B**（代码能力质变阈值）|
| Prefix Cache 命中率 | 低 | **极高**（system prompt + tools 固定）|

### 1.2 Qwen3.6-27B-FP8 的合理性

```mermaid
graph LR
    M14B[Qwen3-14B / 32B<br/>速度快但代码弱 or 显存吃紧] -->|质量/容量不平衡| X1[❌ 不推荐]
    M27B[Qwen3.6-27B-FP8 Hybrid<br/>16 full-attn + 48 linear-attn<br/>FP8 ≈ 27GB] -->|甜点| OK[✅ 首选]
    M70B[Qwen3 70B+ Dense<br/>FP8 仍 ~70GB + KV 紧张] -->|显存压力| X2[⚠️ 可玩但 Coding Agent 体验下降]
    MoE[Qwen3-235B-A22B / -A3B<br/>稀疏激活] -->|总权重 200GB+| X3[❌ 单卡跑不了完整模型]

    style OK fill:#d4edda
```

**确认信息（2026-05 已验证）：**
- `Qwen/Qwen3.6-27B`（bf16 多模态 VL，hybrid 架构 + MTP）和 `Qwen/Qwen3.6-27B-FP8` 均已上 ModelScope / HuggingFace
- vLLM v0.20.1 原生支持 `qwen3_5` + `qwen3_5_mtp.py`（即 Qwen3.6 系列）
- Coding Agent 主线选 FP8 版（非 VL，纯文本/代码）

---

## 二、单卡 PRO 6000 96GB 部署画像

```mermaid
graph TD
    P6[PRO 6000 Blackwell Workstation 96GB<br/>1792 GB/s · sm_120 · 600W]

    P6 --> Plan1[甜点配置<br/>Qwen3.6-27B-FP8]
    Plan1 --> P1A[模型 ~27 GB<br/>剩余 ~65 GB 给 KV + activation]
    P1A --> P1B[KV cache: 256K context<br/>FP8 KV，hybrid 下仅 ~8.4 GB<br/>+ 4 路并发足够]
    P1B --> P1C[Decode 50-66 tps 理论<br/>实测带 prefix cache 可飙到 80+ tps<br/>TTFT < 1s]
    P1C --> P1D[体验 vs Copilot: 90-95%]

    P6 --> Plan2[豪华配置<br/>Qwen3.6-27B bf16 VL 满血]
    Plan2 --> P2A[权重 ~54 GB → 适合做多模态实验]
    P2A --> P2B[Coding Agent 主线不用，仅供研究]

    P6 --> Plan3[激进配置<br/>NVFP4 量化版 (Day 17 自产)]
    Plan3 --> P3A[~14 GB 权重 → 80GB 给 KV<br/>但需自验 Blackwell sm_120 NVFP4 兼容性<br/>sm_120 走 mma.sync.block_scale，非 sm_100 tcgen05]

    style Plan1 fill:#d4edda
```

> 已退役内容：v2.0 之前的"PRO 5000 48GB 方案"已删除。当前只维护单卡 PRO 6000 96GB 主线。

---

## 三、推荐部署方案（vLLM + OpenAI 兼容 API）

### 3.1 一键启动脚本

**`~/serve-coding.sh`：**

```bash
#!/usr/bin/env bash
# Coding Agent 专用 vLLM 启动脚本（PRO 6000 Blackwell 96GB · vLLM v0.20.1）

set -e

HOST=0.0.0.0
PORT=8000
LOG_DIR=$HOME/logs/vllm
mkdir -p "$LOG_DIR"

MODEL=/home/xuefeiz2/models/Qwen3.6-27B-FP8
MAX_LEN=262144      # 256K context（hybrid 架构 + FP8 KV，实际 KV 仅 ~8.4 GB）
GPU_UTIL=0.92
KV_DTYPE=fp8

# 启动 vLLM v0.20.1
exec vllm serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-dtype "$KV_DTYPE" \
    --enable-prefix-caching \
    --prefix-caching-hash-algo builtin \
    --enable-chunked-prefill \
    --max-num-seqs 8 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --served-model-name local-coder \
    2>&1 | tee "$LOG_DIR/$(date +%Y%m%d-%H%M%S).log"
```

> ⚠️ `--tool-call-parser` / `--reasoning-parser` 的可用值随 vLLM 版本变化。先 `vllm serve --help | grep -E 'tool-call-parser|reasoning-parser'` 确认 v0.20.1 实际接受的字符串再硬编码。

**用法：**
```bash
chmod +x ~/serve-coding.sh
~/serve-coding.sh

# 验证
curl http://localhost:8000/v1/models
```

### 3.2 systemd 服务化（开机自启）

**`/etc/systemd/system/vllm-coder.service`：**

```ini
[Unit]
Description=vLLM Coding Agent
After=network.target

[Service]
Type=simple
User=xuefeiz2
WorkingDirectory=/home/xuefeiz2
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/home/xuefeiz2/serve-coding.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now vllm-coder
sudo systemctl status vllm-coder
journalctl -u vllm-coder -f
```

---

## 四、客户端集成

### 4.1 OpenCode（你正在用）

**`~/.opencode/config.json`：**

```json
{
  "model": "local-coder",
  "provider": {
    "name": "openai",
    "baseURL": "http://localhost:8000/v1",
    "apiKey": "dummy"
  },
  "maxTokens": 8192,
  "temperature": 0.2
}
```

### 4.2 Cursor

Settings → Models → Add Custom Model：
- Provider: OpenAI Compatible
- Base URL: `http://localhost:8000/v1`
- Model: `local-coder`
- API Key: 任意字符串

### 4.3 Continue (VSCode/JetBrains)

`.continue/config.yaml`：

```yaml
models:
  - title: Local Qwen3.6 Coder
    provider: openai
    model: local-coder
    apiBase: http://localhost:8000/v1
    apiKey: dummy
    contextLength: 131072
    completionOptions:
      temperature: 0.2
      maxTokens: 8192
```

### 4.4 Claude Code (官方 CLI 不支持自定义 endpoint)

**workaround：用 LiteLLM 做代理**

```bash
uv pip install litellm
litellm --model openai/local-coder \
  --api_base http://localhost:8000/v1 \
  --api_key dummy \
  --port 4000

# Claude Code 走 ANTHROPIC_BASE_URL=http://localhost:4000 ...
```

---

## 五、性能调优清单

```mermaid
graph TD
    Start[默认部署] --> O1[1. 开启 prefix caching<br/>system prompt 命中率 90%+]
    O1 --> O2[2. 开启 chunked prefill<br/>长 context TTFT 平滑]
    O2 --> O3[3. KV cache → FP8<br/>2× context 容量]
    O3 --> O4[4. 启用 P-EAGLE 投机解码<br/>Decode 2-3×]
    O4 --> O5[5. 启用 CUDA Graph<br/>Decode latency -15%]
    O5 --> O6[6. 调 max_num_seqs<br/>多 agent 平衡]
    O6 --> O7[7. 监控 + tune]
```

### 5.1 Prefix Caching 是杀手锏

**典型 Coding Agent 请求结构：**

```
[system_prompt: 5K tokens]   ← 每次都一样，必命中
[tool_definitions: 8K tokens] ← 每次都一样，必命中
[file_context: 50K tokens]    ← 跨请求复用
[conversation: 10K tokens]    ← 同 session 累积复用
[current question: 200 tokens] ← 真正新增
```

**命中率 90% 时**：
- TTFT 从 1500ms → 80ms（**18× 加速**）
- Prefill 算力节省 90%+

**vLLM 启用方法**（已在脚本里）：
```bash
--enable-prefix-caching --prefix-caching-hash-algo builtin
```

### 5.2 投机解码（Coding Agent 最佳场景）

**为什么 Coding Agent 是投机解码完美场景：**
- 并发低（2-4），GPU 没饱和，可以"浪费"算力换延迟
- 代码 token 高度可预测（关键字、缩进、函数名）
- Draft 模型接受率高（60-80%）

**两条路径（按可用性优先级）：**

```bash
# 路径 A：MTP（Qwen3.6-27B-FP8 自带，最简单）
--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'

# 路径 B：P-EAGLE（需外部 EAGLE draft 权重）
# 先在 HF / ModelScope 搜 "Qwen3.6 EAGLE"，确认有官方/社区 draft 再用
# 找不到就用 MTP，不要硬编造模型名
```

### 5.3 max_num_seqs 调优

| 你的并发场景 | max_num_seqs 推荐 |
|---|---|
| 单人单 agent | 2 |
| 单人 + 多 agent (OpenCode 多任务) | 4 |
| 你 + 家人共用 | 8 |

太大 → KV 抢占多，TTFT 不稳；太小 → 并发吞吐浪费

### 5.4 配套监控

```bash
# 装 Grafana + Prometheus
docker compose -f vllm/examples/online_serving/prometheus_grafana/docker-compose.yaml up

# 关键指标
- vllm:num_requests_running
- vllm:gpu_cache_usage_perc
- vllm:time_to_first_token_seconds
- vllm:time_per_output_token_seconds
- vllm:prefix_cache_hit_rate
```

---

## 六、与 Copilot 的真实对比

### 6.1 体验对比表（基于本地 PRO 6000 96GB + Qwen3.6-27B-FP8）

| 维度 | GitHub Copilot | 本地 Qwen3.6 27B | 评估 |
|---|---|---|---|
| 自动补全速度 | 即时 | 60-100ms 延迟 | ⚠️ 略慢 |
| 长 context（整 repo）| 64-128K | **256K** ✅ | 本地胜 |
| 代码质量 | Claude/GPT-4o | Qwen3.6 中等 | Copilot 略胜 |
| Agent 模式（多步推理）| ✅ | ✅ | 平 |
| Tool calling | ✅ | ✅（Qwen3 原生） | 平 |
| 24h 无限调用 | ❌ 配额限制 | **✅** | 本地胜 |
| 隐私（公司代码不外传）| ❌ | **✅** | 本地胜 |
| 月成本 | $20/月 | 电费 ~¥80/月 | 平（但前期投资）|
| 离线可用 | ❌ | **✅** | 本地胜 |

### 6.2 Decode 速度实测（理论 + 实际）

```
配置：PRO 6000 96GB + Qwen3.6-27B-FP8 + 32K KV cache (FP8)
模型显存读取量 ≈ 27 GB（FP8 权重）+ KV 部分（hybrid 下 32K 仅 ~1 GB）
理论 Decode TPS ≈ 1792 / 28 ≈ 64 tps
实测目标（含 prefix cache 命中）= 60-80 tps（Coding Agent 流畅）

启用 MTP / P-EAGLE 后：
有效 TPS 视接受率而定，目标 ×1.5-3，实测以 progress.md 记录为准（不要照抄）
```

---

## 七、调试技巧（必装工具）

| 工具 | 用途 | 安装 |
|---|---|---|
| `nvtop` | 实时 GPU 使用率/显存 | `apt install nvtop` |
| `nvidia-smi dmon` | 时间序列监控 | 自带 |
| `gpustat -i 1` | 美化版 nvidia-smi | `pip install gpustat` |
| `glances` | 系统总览 | `pip install glances` |
| Grafana 面板 | vLLM metrics 可视化 | 上面 docker compose |
| `nsys profile` | 单次 forward 性能剖析 | CUDA toolkit 自带 |

---

## 八、典型问题 & 解决

### 8.1 OOM (显存不足)

```bash
# 症状：vLLM 启动时 RuntimeError: CUDA out of memory
# 解决顺序：
1. 降低 --max-model-len（256K → 128K）
2. 降低 --max-num-seqs（8 → 4）
3. 降低 --gpu-memory-utilization (0.92 → 0.85)
4. 切换更激进量化（FP8 → NVFP4）
5. 启用 --kv-cache-dtype int4 / int2_turbo
```

### 8.2 Prefix Cache 命中率低

```bash
# 检查
curl http://localhost:8000/metrics | grep prefix_cache

# 常见原因
- system prompt 每次微调（时间戳/随机数），破坏 hash
- block_size 太大（默认 16 已经够小）
- 多 client 用不同 system prompt → 拆服务
```

### 8.3 TTFT 突然飙高

```bash
# 通常是 prefix cache 被踢
# 监控：vllm:gpu_cache_usage_perc
# 解决：
- 增加 --num-gpu-blocks-override（手动加 cache 池）
- 降低 max_num_seqs，少抢占
```

### 8.4 Decode 速度慢于理论值

```bash
# 排查 4 步
1. CUDA Graph 是否启用：日志看 "Capturing CUDA graph"
2. 是不是被其他进程抢 GPU：nvidia-smi 看 process
3. PCIe 带宽：nvidia-smi --query-gpu=pcie.link.gen.current
4. 温度降频：nvidia-smi --query-gpu=temperature.gpu (>83°C 会降频)
```

---

## 九、5 月份成本估算

| 项目 | 成本 |
|---|---|
| 电费（PRO 6000 24h × 30 天 × 0.5kW 平均 × ¥0.6/kWh，含整机） | ~¥216 |
| GitHub Copilot 节省 | -$20 ≈ -¥145 |
| **PRO 6000 净月成本** | **~¥70/月**（远低于 cloud GPU 成本，且数据全本地） |

> 注：电费按整机平均 ~0.5kW 估算（GPU 推理负载下不会持续 600W 满载，但配合 CPU/系统额外耗电）。实际跑一周记录到 progress.md 后再校准。

---

## 十、写成博客的角度

**博客标题（求职 + 流量双赢）：**
> 《不再续费 GitHub Copilot：用 RTX PRO 6000 Blackwell 96GB + vLLM v0.20.1 + Qwen3.6-27B-FP8 自建 Coding Agent 的 30 天实战》

**大纲：**
1. 为什么放弃 Copilot（隐私、配额、模型不可控）
2. 硬件选型（为什么是单卡 PRO 6000 96GB 而不是双 4090 / cloud）
3. vLLM v0.20.1 + Qwen3.6 hybrid 架构部署细节
4. Prefix Caching 让 Coding Agent 起飞
5. MTP / P-EAGLE 投机解码：实测吐字速度提升
6. OpenCode/Cursor/Continue 配置
7. 30 天体验对比 Copilot：90-95% 区间
8. 成本核算 + Bonus：求职 portfolio 加分

**这篇博客双重价值：**
- 技术圈传播（流量）
- 简历加分（"我能从硬件到应用栈打通的 LLM 工程师"）

---

## 十一、与 30 天计划的衔接

```mermaid
graph LR
    Day7[Day 7 (2026-05-13)<br/>vLLM hello world<br/>Qwen3.6-27B-FP8 跑通] --> Day17[Day 17 (2026-05-23)<br/>NVFP4 量化模型<br/>同步部署到 Coding Agent (如成功)]
    Day17 --> Day20[Day 20 (2026-05-26)<br/>MTP / P-EAGLE<br/>同步开启]
    Day20 --> Day27[Day 27 (2026-06-02)<br/>稳定运行 1 周]
    Day27 --> Blog[Day 30+ 博客发布<br/>简历加分]

    style Blog fill:#d4edda
```

**关键节点：**
- Day 7 后：本地 vLLM + Qwen3.6-27B-FP8 baseline 跑通
- Day 17 后：本地 Coding Agent 完全替代 Copilot（FP8 baseline；NVFP4 视 Day 17 是否成功）
- Day 21 后：开 MTP / P-EAGLE，速度起飞
- Day 30 后：稳定使用，可写博客

---

## 下一步

返回 [README.md](./README.md) 查看总目录，或开始 [02-week1.md](./02-week1.md) 的 Day 1 任务。
