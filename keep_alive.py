#!/usr/bin/env python3
"""
HPC Session Keeper (~35% resource usage)

GPU: 25GB per GPU (A100 80GB × 31%)
RAM: 70GB (200GB × 35%)
CPU: periodic numpy ops across multiple threads

Usage:
    python scripts/keep_alive.py
    # chain: bash sweep.sh; python scripts/keep_alive.py
    # stop: Ctrl+C
"""

import time, signal, sys
import torch
import numpy as np

signal.signal(signal.SIGINT, lambda *_: (print("\nStopped."), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda *_: (print("\nStopped."), sys.exit(0)))

GPU_MEM_GB = 25
RAM_GB = 70
INTERVAL = 60
N_GPUS = torch.cuda.device_count()  # respects CUDA_VISIBLE_DEVICES

print("=== Session Keeper (~35%) ===")

# GPU: allocate on each device
gpu_tensors = []
for i in range(N_GPUS):
    try:
        dev = torch.device(f"cuda:{i}")
        size = int((GPU_MEM_GB * 1e9 / 4) ** 0.5)
        t = torch.randn(size, size, device=dev)
        gpu_tensors.append((dev, t))
        print(f"  GPU {i}: {torch.cuda.memory_allocated(dev)/1e9:.1f}GB")
    except Exception as e:
        print(f"  GPU {i}: unavailable ({e})")

# RAM: numpy array
ram = np.random.randn(int(RAM_GB * 1e9 / 8))
print(f"  RAM: {ram.nbytes/1e9:.1f}GB")
print(f"  Heartbeat every {INTERVAL}s. Ctrl+C to stop.\n")

step = 0
while True:
    step += 1

    # GPU: 512×512 matmul (light but visible)
    for dev, t in gpu_tensors:
        _ = torch.mm(t[:512, :512], t[:512, :512])
        torch.cuda.synchronize(dev)

    # CPU: 10M element operation (uses a few cores briefly)
    _ = np.dot(ram[:10_000_000], ram[:10_000_000])

    if step % 10 == 0:
        print(f"  [alive] {step * INTERVAL // 60}min")

    time.sleep(INTERVAL)