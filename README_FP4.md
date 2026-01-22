
# 4-bit Quantization & NVIDIA Blackwell Support

This repository includes advanced memory optimization using 4-bit quantization, specifically tailored for NVIDIA GPUs.

## Features

### 1. 4-bit Storage (VRAM Reduction)
- **Problem**: 16-bit models (FP16/BF16) consume ~12.7GB VRAM (Allocated) for the 3B model, often causing OOM on 12GB cards or limiting multitasking on 16GB cards.
- **Solution**: We compress model weights into 4-bit format using `bitsandbytes`.
- **Benefit**: Reduces Allocated VRAM usage to **~8.3 GB** (approx 35% saving).

### 2. Auto-Detecting Blackwell Support (RTX 50-series)
Our implementation automatically detects if you are running on an NVIDIA Blackwell GPU (Compute Capability 10.0+).

- **Legacy GPUs (RTX 30/40)**: Uses `nf4` (NormalFloat4) quantization.
- **Blackwell GPUs (RTX 50+)**: Switches to **`fp4`** (Floating Point 4) quantization.

## "True" FP4 Compute?

**Current Status**:
- **Storage**: **YES**. Weights are effectively stored in the Blackwell-native FP4 format in memory (saving VRAM).
- **Compute**: **Partial**. As of early 2026, the library performs "on-the-fly dequantization". Weights are FP4, but temporarily converted back to BF16/FP16 for the actual Matrix Multiplication.
- **Impact**: 
    - **Pros**: Massive VRAM savings (4.5GB+).
    - **Cons**: Slight generation speed decrease (~15% slower) due to dequantization overhead.

**Benchmarks (RTX 5060 Ti 16GB, 20s Audio):**
- **Original (BF16)**: ~44s Generation | ~12.7 GB Allocated
- **Quantized (FP4)**: ~51s Generation | ~8.3 GB Allocated

**Future Proofing**:
The code is architected to switch to native FP4 compute kernels automatically as soon as the underlying libraries (`bitsandbytes` / `TorchAO`) stabilize their support for Blackwell Tensor Cores.
