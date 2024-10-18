import time
import torch

N = 10000
offload_size = 5


class LinearCache:
    def __init__(self):
        self.device = "cuda"

        self.size = N * N * 2
        w1 = torch.UntypedStorage(self.size * offload_size).pin_memory()
        self.w1_offloaded_storages = [
            w1[i * self.size : (i + 1) * self.size] for i in range(offload_size)
        ]
        w2 = torch.UntypedStorage(self.size * offload_size).pin_memory()

        self.w2_offloaded_storages = [
            w2[i * self.size : (i + 1) * self.size] for i in range(offload_size)
        ]
        w3 = torch.UntypedStorage(self.size * offload_size).pin_memory()
        self.w3_offloaded_storages = [
            w3[i * self.size : (i + 1) * self.size] for i in range(offload_size)
        ]

        tmp = torch.UntypedStorage(self.size, device="cuda")
        tmp_view = torch.as_tensor(tmp, dtype=torch.float16, device="cuda").view(N, N)
        ttt = torch.randn(N, N)
        tmp_view.copy_(ttt)
        for i in range(offload_size):
            self.w1_offloaded_storages[i].copy_(tmp, non_blocking=True)
            self.w2_offloaded_storages[i].copy_(tmp, non_blocking=True)
            self.w3_offloaded_storages[i].copy_(tmp, non_blocking=True)

        self.copy_stream = torch.cuda.Stream()

        def storage():
            storage_size = N * N * 2
            storage = torch.UntypedStorage(storage_size, device=self.device)
            a_view = torch.as_tensor(
                storage, dtype=torch.float16, device=self.device
            ).view(N, N)
            ttt = torch.randn(N, N)
            a_view.copy_(ttt)
            assert a_view.data_ptr() == storage.data_ptr()
            return storage, a_view

        self.storage1, self.a1 = storage()
        self.storage2, self.a2 = storage()
        self.storage3, self.a3 = storage()
        self.w1_event = torch.cuda.Event()
        self.w2_event = torch.cuda.Event()
        self.w3_event = torch.cuda.Event()

    def _load(self, index):
        with torch.cuda.stream(self.copy_stream):
            self.storage1.copy_(self.w1_offloaded_storages[index], non_blocking=True)
            self.w1_event.record()
            self.storage2.copy_(self.w3_offloaded_storages[index], non_blocking=True)
            self.w3_event.record()
            self.storage3.copy_(self.w2_offloaded_storages[index], non_blocking=True)
            self.w2_event.record()
        self.w1_event.wait()
        current_hidden_states = self.a1 @ self.a1
        self.w3_event.wait()
        current_hidden_states = current_hidden_states @ self.a3
        self.w2_event.wait()
        current_hidden_states = current_hidden_states @ self.a2


linear = LinearCache()
for i in range(offload_size):
    linear._load(i)
    print(i)
