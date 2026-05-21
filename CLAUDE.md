# CLAUDE.md — Repo Context for Claude Code

## What This Is

A 30-day learning sprint: Intel ISP systems engineer (7 years) transitioning to LLM inference engineer at top Chinese AI companies (ByteDance AML / Moonshot / DeepSeek / MiniMax).

- **Day 0**: 2026-05-06 (done — hardware + docs only, zero code)
- **Day 1**: 2026-05-07 (pending)
- **Today**: Check `progress.md` for latest status.

## Hardware

- **GPU**: RTX PRO 6000 Blackwell Workstation 96GB — **sm_120** (NOT sm_100)
- **CPU**: Intel Core Ultra 9 285K, ~30GB DDR5 (tight for 27B models)
- **OS**: Ubuntu 24.04 + CUDA 13.2
- **Main model**: Qwen3.6-27B-FP8 (hybrid: 16 full-attn + 48 linear-attn layers)

## Reference Repos (STUDY, NOT COPY)

- `~/3rd/vllm` — Production vLLM v0.20.1, 100K+ lines. Architecture reference.
- `~/3rd/nano-vllm` — Minimal ~1,200-line vLLM clone. Coding reference for mini-vLLM. See `06-mini-vllm-design.md §十二` for module-by-module mapping.

## File Navigation

| File | What |
|---|---|
| `progress.md` | Current day, daily checkpoints, jobs linkage |
| `COACH.md` | AI assistant rules (coaching mode L1-L2) |
| `02-week1.md` | CUDA basics + LLM inference fundamentals (Day 1-7) |
| `03-week2.md` | vLLM v0.20 source deep read + MLA + FA4 (Day 8-14) |
| `04-week3.md` | Triton kernels + quantization + speculative decoding (Day 15-21) |
| `05-week4.md` | Mini-vLLM project + job hunting sprint (Day 22-30) |
| `06-mini-vllm-design.md` | Mini-vLLM architecture design + nano-vllm mapping |
| `08-job-strategy.md` | Resume / 30 interview Qs / salary negotiation |
| `jobs/` | Job readiness tasks embedded in daily plans |

## Daily Learning Flow (DO NOT SKIP)

1. **Read interview questions** → check today's `## Day N` section for `## 🎯 今日面试题`
2. **Read reference code** → `~/3rd/vllm` or `~/3rd/nano-vllm`, find code matching today's concepts
3. **Answer interview questions** → in user's own words, AI corrects/probes
4. **Handwork/coding** → only after understanding demonstrated
5. **Feynman check** → AI quizzes user (2-3 oral questions)
6. **Daily note** → ≤30 lines in `progress.md`

## AI Behavior Rules (from COACH.md)

- **Coaching mode L1-L2 by default**: hints + skeleton code, NOT full implementations
- **L4 only when user says** "直接写" / "我赶时间"
- **Always explain Why before How**
- **Connect new concepts to user's Intel ISP/FreeRTOS/PCIe background** (e.g., "PagedAttention block table ≈ MMU page table")
- **Never rush to code** — understanding first, building second
- **User answers interview questions in own words** — don't write answers for them
- **Honesty on knowledge boundaries** — vLLM v0.20+ / Blackwell post-cutoff, use WebFetch for latest
- **Update `progress.md` at session end** (with user consent)
- **Write to**: `progress.md`, `week*/day*/`, `mini-vllm/`, `blog/`, `notes/`
- **Do NOT touch**: plan docs, README.md, COACH.md, AGENTS.md unless explicitly asked

## Key Context

- **sm_120 ≠ sm_100**: PRO 6000 Workstation shares Blackwell family but lacks tcgen05/TMEM/5th-gen NVLink. Baseline PTX forward-compatible, but cubin binaries not interchangeable.
- **Qwen3.6-27B is hybrid**: 16 full-attention layers + 48 linear-attention layers. KV cache only on 16 full-attn layers, making it much more memory-efficient than pure Transformer.
- **30GB RAM is a bottleneck** for 27B model loading. Use mmap + chunked loading, or benchmark with smaller models (Qwen2.5-1.5B).
- **Mini-vLLM target**: 5 days, reach 30-50% of vLLM performance, cover PagedAttention / Continuous Batching / FA4 / Triton.

## Interview Q Bank

30 questions mapped across 27 days in `08-job-strategy.md §4.2`. Three categories:
- A (1-10): LLM inference fundamentals
- B (11-20): vLLM / system design
- C (21-30): CUDA / performance optimization
