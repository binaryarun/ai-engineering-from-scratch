import time, torch

size = 5000
a, b = torch.randn(size, size), torch.randn(size, size)

t = time.time(); a @ b; cpu = time.time() - t
print(f"CPU: {cpu:.3f}s")

if torch.backends.mps.is_available():
    ag, bg = a.to("mps"), b.to("mps")
    ag @ bg                      # warm-up run
    torch.mps.synchronize()
    t = time.time(); ag @ bg; torch.mps.synchronize()
    gpu = time.time() - t
    print(f"MPS: {gpu:.3f}s  speedup: {cpu/gpu:.0f}x")
