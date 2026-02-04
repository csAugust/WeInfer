# WeInfer Benchmark Results

## Setup

- **Model**: SmolLM-135M-Instruct-q0f32-MLC (0.3GB) / SmolLM2-135M-Instruct-q0f32-MLC for WebLLM Latest
- **Prompt**: "Please introduce the Peking University in detail to me."
- **Max new tokens**: 32
- **xInterval (x)**: 4
- **Browser**: Chromium (Playwright)
- **Platform**: macOS (Darwin 25.2.0)

## Decode Speed (ms/token)

| Run | WebLLM Latest (v0.2.80) | WebLLM +Cache | WebLLM +Cache+xInterval |
|-----|-------------------------|---------------|-------------------------|
| 1   | 19.49                   | 15.84         | 11.78                   |
| 2   | 19.73                   | 16.82         | 9.82                    |
| 3   | 19.46                   | 16.04         | 9.65                    |
| 4   | 22.64                   | 16.25         | 9.77                    |
| 5   | 20.87                   | 16.70         | 10.38                   |
| 6   | 19.47                   | 16.77         | 10.27                   |
| 7   | 21.45                   | 16.43         | 11.45                   |
| 8   | 19.14                   | 17.34         | 11.71                   |
| 9   | 18.71                   | 17.24         | 11.57                   |
| 10  | 21.44                   | 17.35         | 12.13                   |
| **Avg** | **20.24**           | **16.68**     | **10.85**               |

## Summary

| Config                   | Avg (ms/token) | Speedup vs Latest |
|--------------------------|----------------|-------------------|
| WebLLM Latest (v0.2.80)  | 20.24          | 1.00x             |
| WebLLM +Cache            | 16.68          | 1.21x             |
| WebLLM +Cache+xInterval  | 10.85          | 1.86x             |

## Analysis

### WebLLM Latest (v0.2.80)

Updated from the previous pre-built v0.2.46 artifact to the latest npm release (v0.2.80). Uses the OpenAI-compatible `chat.completions.create()` streaming API instead of the deprecated `generate()` method. Model changed from SmolLM v1 to SmolLM2-135M-Instruct-q0f32-MLC (loaded from HuggingFace prebuilt config).

Each decode step follows a synchronous flow:

```
forward() → GPU sync → allocate new buffer → sample token → process → repeat
```

Two bottlenecks limit performance:

1. **Per-token GPU buffer allocation**: `GPUSampler` creates a fresh bind group and uniform buffer for every token sample.
2. **Per-token GPU/CPU synchronization**: The CPU blocks waiting for the GPU after each token.

### WebLLM +Cache (1.21x speedup)

Introduces **buffer reuse** via the `GPUSamplerFixed2Dst` class:

- Pre-allocates GPU buffers once and reuses them across all decode steps.
- Eliminates per-token buffer allocation overhead.
- Still performs GPU/CPU sync after every single token.

The ~21% speedup comes purely from removing repeated GPU memory allocation.

### WebLLM +Cache+xInterval (1.86x speedup)

Adds **asynchronous pipelining** on top of buffer reuse. With `x=4` (default):

```
Step 1: forward → sample → store in slot[0] → NO SYNC
Step 2: forward → sample → store in slot[1] → NO SYNC
Step 3: forward → sample → store in slot[2] → NO SYNC
Step 4: forward → sample → store in slot[3] → SYNC (copy all 4 tokens at once)
```

Instead of blocking the CPU after every token, it only syncs every `x` tokens. This:

- **Reduces GPU/CPU synchronization overhead by 4x** (with x=4).
- **Keeps the GPU pipeline saturated** — the GPU doesn't idle waiting for CPU acknowledgment.
- **Batches the expensive `device.sync()` call** across multiple tokens.

### Optimization Impact

| Optimization             | What It Eliminates                | Speedup |
|--------------------------|-----------------------------------|---------|
| Buffer reuse (+Cache)    | Per-token GPU buffer allocation   | 1.21x   |
| Async pipeline (+xInterval) | Per-token GPU/CPU synchronization | 1.86x   |

The dominant bottleneck is **GPU/CPU synchronization**, not memory allocation. This is why the async pipeline — reducing sync frequency from every token to every 4 tokens — provides a much larger improvement than buffer reuse alone.
