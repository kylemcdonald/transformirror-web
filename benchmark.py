import torch, time

x = torch.randn(4, 4, 4096, 4096, device="cuda")
conv = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1).cuda()

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    conv(x)
torch.cuda.synchronize()
print("Time per step:", (time.time() - start) / 100 * 1000, "ms")