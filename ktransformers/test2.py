import time
import torch
from torch.utils.data import DataLoader

N = 50000
torch.cuda.empty_cache()
x = torch.UntypedStorage(N * N * 2).pin_memory("cuda")
x2 = torch.UntypedStorage(N * N * 2).pin_memory("cuda")
x3 = torch.UntypedStorage(N * N * 2).pin_memory("cuda")

y = torch.randn(N // 5 , N // 5, device="cuda")

tmp = torch.UntypedStorage(N * N * 2, device="cuda")
tmp2 = torch.UntypedStorage(N * N * 2, device="cuda")
tmp3 = torch.UntypedStorage(N * N * 2, device="cuda")
tmp_view = torch.as_tensor(tmp, dtype=torch.float16, device="cuda").view(N, N)
tmp_view2 = torch.as_tensor(tmp2, dtype=torch.float16, device="cuda").view(N, N)
tmp_view3 = torch.as_tensor(tmp3, dtype=torch.float16, device="cuda").view(N, N)
tmp_view = torch.randn(N, N)
tmp_view2 = torch.randn(N, N)
tmp_view3 = torch.randn(N, N)
x.copy_(tmp, non_blocking=True)
x2.copy_(tmp2, non_blocking=True)
x3.copy_(tmp3, non_blocking=True)

stream = torch.cuda.Stream()
event = torch.cuda.Event()
event2 = torch.cuda.Event()
event3 = torch.cuda.Event()

with torch.cuda.stream(stream):
    tmp.copy_(x, non_blocking=True)
    event.record()
    tmp2.copy_(x2, non_blocking=True)
    event2.record()
    tmp2.copy_(x2, non_blocking=True)
    event2.record()
event.wait()
z = y @ y
event2.wait()
z = z @ y
event3.wait()
z = z @ y
