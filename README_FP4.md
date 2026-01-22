
# 4-bit Quantization & NVIDIA Blackwell Support

This repository includes advanced memory optimization using 4-bit quantization, specifically tailored for NVIDIA GPUs.

## Features

### 1. 4-bit Storage (VRAM Reduction)
- **Problem**: 16-bit models (FP16/BF16) consume ~12-14GB VRAM for the 3B model, leaving little room for other nodes.
- **Solution**: We compress model weights into 4-bit format.
- **Benefit**: Reduces VRAM usage to **~8GB** (approx 35% saving), allowing comfortable usage on 12GB/16GB cards.

### 2. Auto-Detecting Blackwell Support (RTX 50-series)
Our implementation automatically detects if you are running on an NVIDIA Blackwell GPU (Compute Capability 10.0+).

- **Legacy GPUs (RTX 30/40)**: Uses `nf4` (NormalFloat4) quantization.
- **Blackwell GPUs (RTX 50)**: Switches to `fp4` (Floating Point 4) quantization.

## "True" FP4 Compute?

**Current Status**:
- **Storage**: **YES**. Weights are effectively stored in the Blackwell-native FP4 format in memory.
- **Compute**: **Partial**. As of early 2026, the `bitsandbytes` library performs "on-the-fly dequantization". This means even though weights are FP4, they are temporarily converted back to BF16 for the actual calculation.
- **Impact**: You get the full VRAM savings, but inference speed is slightly slower than pure FP16 (due to the conversion overhead).

**Future Proofing**:
The code is written such that as soon as `bitsandbytes` or PyTorch enables native FP4 compute kernels, your hardware will be ready to use them without further code changes.
