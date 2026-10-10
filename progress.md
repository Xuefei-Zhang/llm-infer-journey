# 30 天进度跟踪

> 每天结束时记录：完成项 / 卡点 / 明日计划 / 心得

---

## Day 0 - 2026-05-06（周三，硬件落地日）

### 完成
- [x] 30 天计划 v2.0 文档体系（10 个文件）
- [x] 仓库初始化 `~/self/llm-infer-journey/`
- [x] 装机：Intel Core Ultra 9 285K + MSI PRO B860-P WIFI + RTX PRO 6000 Blackwell 96GB
- [x] 系统：Ubuntu 24.04 LTS + Driver 595.58.03 + CUDA 13.2（`nvidia-smi` / `nvcc -V` 验证通过）
- [x] 模型下载：`Qwen/Qwen3.6-27B`（bf16 VL，多模态版）已开始拉取至 `~/models/Qwen3.6-27B/`
- [x] 文档 v2.1 改写：删除 AutoDL/双 GPU/PRO 5000 分支，全量校准为本地 96GB 单卡环境

### 待办（Day 1 之前必须做完）
- [ ] `Qwen/Qwen3.6-27B` 下载收尾（确认 `model.safetensors.index.json` 落盘）
- [ ] 下载 FP8 主力模型：
  ```bash
  modelscope download --model Qwen/Qwen3.6-27B-FP8 --local_dir ~/models/Qwen3.6-27B-FP8
  ```
- [ ] 安装 vLLM v0.20.1（`uv venv --python 3.12 && uv pip install "vllm==0.20.1"`，CUDA 13 wheel）
- [ ] 安装 Nsight Systems（`sudo apt install nsight-systems`）
- [ ] DDR5 内存升级：当前 ~30GB 偏紧（HF safetensors 反序列化峰值 + vLLM CPU offload 需 ≥64GB），建议加到 64GB 或 96GB

### 风险
- 系统内存只有 30GB：Week 4 mini-vLLM 测 27B 模型时若开 CPU KV offload 会 OOM；Week 1 之前最好补内存条
- Blackwell sm_120 在 vLLM quant 表里尚无独立列（最新表列到 Hopper），FP8 W8A8 / NVFP4 实际能否跑、性能多少 → Week 3 Day 17-18 自验

### 心得
- 五天硬件折腾完，正式开始。计划改回**本地单卡 96GB** 主线，所有云上/过渡内容删除。

### 文档勘误（v2.1 → v2.1.1）
- **错误**：v2.1 全篇把 PRO 6000 Workstation 写成 SM 10.0（24 处）。
- **正确**：`nvidia-smi` 实测 `compute_cap=12.0`，NVIDIA 官方表确认 PRO 6000 Workstation = **sm_120**；sm_100 是 B100/B200/GB200（数据中心 Blackwell）。
- **影响**：sm_120 与 sm_100 同 Blackwell 家族但**不同 SM 主版本**：sm_100 独占 tcgen05/TMEM/5 代 NVLink/INT4 TC/228KB shared mem；sm_120 走 mma.sync + WGMMA。二进制 cubin 不互通，baseline PTX `compute_100` 向前兼容到 sm_120。
- **简历改动**：删除"100% 指令集兼容"；改为"sm_120 实战 + 熟悉 sm_100 移植路径"。
- **修正**：COACH/README/01-09 共 24 处 SM10 → sm_120 + 4 处兼容性断言重写。

---

## Re-baseline - 2026-05-07 未启动，2026-10-10（周六）全面重校准

### 背景
Day 1 原定 2026-05-07，实际 5 个月未启动。今日全面重调研环境，计划文档已同步更新：

- **日期重排**：Day 1 = 2026-10-10 ~ Day 30 = 2026-11-08（30 天不变）
- **主力模型**：Qwen3.6-27B-FP8 → **Qwen3.8-27B-FP8**（29GB，64 层 hybrid：16 full-attn + 48 linear-attn，`model_type: qwen3_5`，带 MTP head；API served-name 故意保留 `Qwen3.6-27B-FP8` 作兼容别名）
- **vLLM**：v0.20.1 → **v0.21.0**（`~/deploy/vllm-prod/.venv`，torch 2.11+cu130）；生产 @ 9070：256K ctx / FP8 KV / prefix caching / chunked prefill / tool-call+reasoning parser，2026-08 至今稳定，即 Claude Code 后端
- **源码参考**：`~/3rd/llm_inference/vllm`（含 v0.21.0 tag，HEAD 为 v0.23.1rc0 系自编）；`~/3rd/vllm` 实为 KVCache 学习笔记目录
- **NVFP4**：本地 `Qwen3.8-27B-NVFP4`（22GB，官方下载非自产）**已知故障：GPU hang / 系统 crash，未修复，暂不使用**；Day 17 改为“定位根因 + 自产对照”（自产需 bf16 源模型，本地暂无）
- **已提前完成项**（8 月实战，非本计划产出但可回填）：Day 6 的部署+benchmark 已有实战数据（2026-08-26，TTFT ~32ms / 裸引擎解码 ~50.8 tok/s；⚠️ Qwen3.8 默认 xhigh thinking 开启时**体感仅 18~22 tok/s**，关 thinking 才 50.8，见 `~/deploy/vllm-profiling/comparison-analysis.md`；另注意 API 名陷阱：served-model-name 是旧名 `Qwen3.6-27B-FP8`，用 3.8 名调用 404）。另有 vLLM 源码自编 16 个 .so、SGLang 对照部署、3.6 vs 3.8 对比分析。Day 5 公式部分（Roofline/显存口算）仍需打卡

### 卡点（部分，实测已降级为非阻塞）
- **驱动 NVML mismatch**：今日 09:15 apt 升级 nvidia 用户态库 595.71.05→595.91.07，内核模块仍 595.71.05（`/var/run/reboot-required` 已标记）。**实测分界线**：❌ 只有走 NVML 的工具挂（`nvidia-smi`、vLLM CLI 平台探测）；✅ **CUDA 计算栈完全正常**（torch `cuda.is_available()=True`、nvcc 编译的 kernel 同步后正确返回、nsys 能出 trace）。**结论：Day 1 的 CUDA kernel 学习不受阻塞**，重启仅为让 `nvidia-smi` 监控工具恢复（可选，非必须；生产服务也不受影响）。

### 今日计划（Day 1 = 今天，2026-10-10 周六）
- ~~重启修驱动~~ → **不必重启**：CUDA 计算栈实测正常，Day 1 直接开工（nvidia-smi 监控工具报 mismatch 无碍，要修可稍后重启）
- 环境校准：`torch.cuda.get_device_capability()`=(12,0) ✅ 已验 / Nsight ✅ 已验 / nvcc 13.2 ✅ 已验
- vec_add.cu 第一个 CUDA kernel + nsys
- 面试题 Q23（C.23 Warp divergence）

---

## Day 1 - 2026-05-07（周四）

> ⚠️ 本模板日期已过期，Day 1 实际重排为 **2026-10-10（周六）**，见 [02-week1.md](./02-week1.md#day-1)。本节保留为打卡模板。

### 计划：[02-week1.md Day 1](./02-week1.md#day-1)

### 面试题
- 今日：C.23（Warp divergence）
- 自答摘要：

### 完成
- [ ]

### [jobs] 完成的任务
- （如无 jobs 任务留空）

### [jobs] 发现的阻塞
- （如无阻塞留空）

### 卡点
-

### 明日计划
-

---

<!-- 后续每天追加 -->
