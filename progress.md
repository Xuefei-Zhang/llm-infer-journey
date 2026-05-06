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

## Day 1 - 2026-05-07（周四）

### 计划：[02-week1.md Day 1](./02-week1.md#day-1)

### 完成
- [ ]

### 卡点
-

### 明日计划
-

---

<!-- 后续每天追加 -->
