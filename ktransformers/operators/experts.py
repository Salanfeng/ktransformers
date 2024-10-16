#!/usr/bin/env python
# coding=utf-8
"""
Description  :  
Author       : Azure-Tang, Boxin Zhang, chenht2022
Date         : 2024-07-25 11:25:24
Version      : 0.1.0
LastEditors  : Azure 
LastEditTime : 2024-08-15 02:36:29
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import json
import threading
from typing import Any, Union
import numpy as np
import numpy.typing as npt
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import sys, os
from ktransformers.operators.base_operator import BaseInjectedModule

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build")
)
sys.path.append(
    os.path.join(
        os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Release"
    )
)
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Debug")
)
import cpuinfer_ext
from cpuinfer_ext.moe import MOEConfig, MOE
import ctypes
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.util.utils import InferenceState
from ktransformers.server.config.config import Config
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from abc import ABC, abstractmethod
from ktransformers.operators.linear import (
    KLinearMarlin,
    KLinearTorch,
    KTransformersLinear,
)
import time
from ktransformers.operators.cpuinfer import CPUInfer


# class Base(BaseInjectedModule, ABC):
class KExpertsBase(ABC):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        device: str = "cuda",
        **kwargs,
    ):
        # super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.key = key
        self.gguf_loader = gguf_loader
        self.config = config
        self.device = device

    @abstractmethod
    def forward(self, input_tensor, expert_ids, weights):
        pass

    @abstractmethod
    def load(
        self,
        w: dict | nn.Parameter | tuple | None = None,
        device: str = "cpu",
        warmup: bool = False,
    ):
        pass

    @abstractmethod
    def unload():
        pass

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                targets = [
                    ".ffn_gate_exps.weight",
                    ".ffn_up_exps.weight",
                    ".ffn_down_exps.weight",
                ]
                tensors = self.load_multi(key, targets, device=device)
                gate = tensors[".ffn_gate_exps.weight"]
                up = tensors[".ffn_up_exps.weight"]
                down = tensors[".ffn_down_exps.weight"]
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"][
                    "ggml_type"
                ]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"][
                    "ggml_type"
                ]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"][
                    "ggml_type"
                ]
            elif key + ".ffn_down.0.weight" in self.gguf_loader.tensor_info:
                # for supporting  Mixtral-8x7B-Instuct
                gate = []
                up = []
                down = []
                for i in range(8):
                    gatei, upi, downi = (
                        f".ffn_gate.{i}.weight",
                        f".ffn_up.{i}.weight",
                        f".ffn_down.{i}.weight",
                    )
                    targets = [gatei, upi, downi]
                    tensors = self.load_multi(key, targets, device=device)
                    gate_it, up_it, down_it = (
                        tensors[gatei],
                        tensors[upi],
                        tensors[downi],
                    )
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = torch.stack(gate)
                up = torch.stack(up)
                down = torch.stack(down)
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate.0.weight"][
                    "ggml_type"
                ]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up.0.weight"][
                    "ggml_type"
                ]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down.0.weight"][
                    "ggml_type"
                ]
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {
                key: {
                    "gate": gate,
                    "up": up,
                    "down": down,
                    "gate_type": gate_type,
                    "up_type": up_type,
                    "down_type": down_type,
                }
            }
        return res

    def load_multi(self, key: str, keys: list[str], device: str = "cpu"):
        tensors = {}
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + k, device=device)
        return tensors


class KExpertsCPU(KExpertsBase):
    input_tensor_cpu: Tensor = None
    expert_ids_cpu: Tensor = None
    weights_cpu: Tensor = None
    output_cpu: Tensor = None
    output_gpu_map: dict = {}  # Manage output tensor buffer on different gpu
    # stream_map:dict = {} # Manage cuda stream on different gpu
    CPU_INFER = CPUInfer(Config().cpu_infer)

    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        out_device: str = "cuda",  # this device mean which device the output should on. TODO: support cpu.
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        assert device.lower() == "cpu", "KExpertsCPU can only be loaded on CPU"
        self.n_routed_experts = n_routed_experts
        self.out_device = out_device

    def load(
        self,
        w: dict | nn.Parameter | tuple | None = None,
        device: str | None = None,
        warmup: bool = False,
    ):
        if device:
            assert (
                device.lower() == "cpu"
            ), 'KExpertsCPU can only be loaded on CPU, Parameter "device" can be cpu or None.'
        if w is None:
            w = self.load_weights()[self.key]
        self.gate = w["gate"]
        self.up = w["up"]
        self.down = w["down"]
        self.gate_type = w["gate_type"]
        self.up_type = w["up_type"]
        self.down_type = w["down_type"]
        gate_ptr = ctypes.addressof(
            ctypes.cast(self.gate.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        up_ptr = ctypes.addressof(
            ctypes.cast(self.up.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        down_ptr = ctypes.addressof(
            ctypes.cast(self.down.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        # print(self.gate_qtype, self.up_qtype, self.down_qtype)
        n_routed_experts = self.n_routed_experts
        # n_routed_experts = len(self.orig_module)
        moe_config = MOEConfig(
            n_routed_experts,
            self.config.num_experts_per_tok,
            self.config.hidden_size,
            self.config.moe_intermediate_size,
            64,
            10,
            1024,
            gate_ptr,
            up_ptr,
            down_ptr,
            self.gate_type,
            self.up_type,
            self.down_type,
            30,  # TODO: get from model.dtype
        )
        # print(n_routed_experts, hidden_size, moe_intermediate_size)
        num_experts_per_tok = self.config.num_experts_per_tok
        self.moe = MOE(moe_config)
        self.cpu_infer = KExpertsCPU.CPU_INFER
        if warmup:
            self.cpu_infer.submit(self.moe.warm_up())
            self.cpu_infer.sync()
        if self.out_device not in KExpertsCPU.output_gpu_map:
            KExpertsCPU.output_gpu_map[self.out_device] = torch.zeros(
                (self.config.hidden_size), device=self.out_device
            )
        if KExpertsCPU.input_tensor_cpu == None:
            KExpertsCPU.input_tensor_cpu = torch.zeros(
                (self.config.hidden_size), device="cpu", pin_memory=True
            )
            KExpertsCPU.expert_ids_cpu = torch.zeros(
                (num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True
            )
            KExpertsCPU.weights_cpu = torch.zeros(
                (num_experts_per_tok),
                device="cpu",
                dtype=torch.float32,
                pin_memory=True,
            )
            KExpertsCPU.output_cpu = torch.zeros(
                (self.config.hidden_size),
                device="cpu",
                pin_memory=True,
                dtype=torch.bfloat16,
            )

    def submit_for_one_decode(self, input_tensor, expert_ids, weights):
        KExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
        KExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
        KExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
        self.cpu_infer.submit_with_cuda_stream(
            torch.cuda.current_stream(self.out_device).cuda_stream,
            self.moe.forward(
                1,
                expert_ids.size(0),
                KExpertsCPU.expert_ids_cpu.data_ptr(),
                KExpertsCPU.weights_cpu.data_ptr(),
                KExpertsCPU.input_tensor_cpu.data_ptr(),
                KExpertsCPU.output_cpu.data_ptr(),
            ),
        )

    def sync_for_one_decode(self):
        self.cpu_infer.sync_with_cuda_stream(
            torch.cuda.current_stream(self.out_device).cuda_stream
        )
        KExpertsCPU.output_gpu_map[self.out_device].copy_(
            KExpertsCPU.output_cpu, non_blocking=True
        )
        return KExpertsCPU.output_gpu_map[self.out_device]

    def forward(self, input_tensor, expert_ids, weights):
        if not os.path.exists("data.pt"):
            torch.save(
                {
                    "input_tensor": input_tensor,
                    "expert_ids": expert_ids,
                    "weights": weights,
                },
                "data.pt",
            )
        # generate, capture and run cuda graph
        # print(expert_ids)
        # start = time.time()
        if input_tensor.size(0) == 1:
            # TODO: this branch is unreachable, but the shape of input_tensor([1,hidden_size]) and input_tensor_cpu([hidden_size]) is not compatible
            # print("capturing experts")
            KExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
            KExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
            KExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(
                torch.cuda.current_stream().cuda_stream,
                self.moe.forward(
                    1,
                    expert_ids.size(1),
                    KExpertsCPU.expert_ids_cpu.data_ptr(),
                    KExpertsCPU.weights_cpu.data_ptr(),
                    KExpertsCPU.input_tensor_cpu.data_ptr(),
                    KExpertsCPU.output_cpu.data_ptr(),
                ),
            )
            self.cpu_infer.sync_with_cuda_stream(
                torch.cuda.current_stream().cuda_stream
            )
            KExpertsCPU.output_gpu_map[self.out_device].copy_(
                KExpertsCPU.output_cpu, non_blocking=True
            )
            return KExpertsCPU.output_gpu_map[self.out_device]
        else:
            input_tensor = input_tensor.contiguous().cpu()
            expert_ids = expert_ids.contiguous().cpu()
            weights = weights.contiguous().to(torch.float32).cpu()
            output = torch.empty_like(input_tensor).contiguous()
            self.cpu_infer.submit(
                self.moe.forward(
                    expert_ids.size(0),
                    expert_ids.size(1),
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_tensor.data_ptr(),
                    output.data_ptr(),
                )
            )
            self.cpu_infer.sync()
            # end = time.time()
            # print(f"KExpertsCPU forward time: {end-start}")
            return output.to(device=object.__getattribute__(self, "out_device"))

    def unload(self):
        return

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"][
                    "ggml_type"
                ]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"][
                    "ggml_type"
                ]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"][
                    "ggml_type"
                ]
            elif key + ".ffn_down.0.weight" in self.gguf_loader.tensor_info:
                # for supporting  Mixtral-8x7B-Instuct
                gate = []
                up = []
                down = []
                for i in range(8):
                    gate_it = self.gguf_loader.get_mmap_tensor(
                        f"{key}.ffn_gate.{i}.weight"
                    )
                    up_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_up.{i}.weight")
                    down_it = self.gguf_loader.get_mmap_tensor(
                        f"{key}.ffn_down.{i}.weight"
                    )
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = np.stack(gate)
                up = np.stack(up)
                down = np.stack(down)
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate.0.weight"][
                    "ggml_type"
                ]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up.0.weight"][
                    "ggml_type"
                ]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down.0.weight"][
                    "ggml_type"
                ]
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {
                key: {
                    "gate": gate,
                    "up": up,
                    "down": down,
                    "gate_type": gate_type,
                    "up_type": up_type,
                    "down_type": down_type,
                }
            }
        return res


class KExpertsMarlin(KExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]

    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.expert_num = n_routed_experts
        self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        self.device = device
        # create empty marlin experts according to the number of experts per token
        # up
        self.up_projs = [
            KLinearMarlin(key + "." + "ffn_up_exps", gguf_loader, config, device=device)
            for i in range(self.expert_num)
        ]
        # gate
        self.gate_projs = [
            KLinearMarlin(
                key + "." + "ffn_gate_exps", gguf_loader, config, device=device
            )
            for i in range(self.expert_num)
        ]
        # down
        self.down_projs = [
            KLinearMarlin(
                key + "." + "ffn_down_exps", gguf_loader, config, device=device
            )
            for i in range(self.expert_num)
        ]

    def load(
        self,
        w: dict | nn.Parameter | tuple | None = None,
        device: str | None = None,
        warmup: bool = False,
    ):
        if device is None:
            device = self.device
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        if w is None:
            w = self.load_weights()[self.key]

        if isinstance(w, dict):
            self.gate = nn.Parameter(torch.from_numpy(w["gate"]))
            self.up = nn.Parameter(torch.from_numpy(w["up"]))
            self.down = nn.Parameter(torch.from_numpy(w["down"]))
            for i in range(self.expert_num):
                self.up_projs[i].load(self.up[i, ...], device=device)
                self.gate_projs[i].load(self.gate[i, ...], device=device)
                self.down_projs[i].load(self.down[i, ...], device=device)
                self.loaded_experts_idx.append(i)
        return

    def unload(self):
        for i in self.loaded_experts_idx:
            self.up_projs[i].unload()
            self.gate_projs[i].unload()
            self.down_projs[i].unload()
        self.loaded_experts_idx = []

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.load_gguf_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.load_gguf_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.load_gguf_tensor(key + ".ffn_down_exps.weight")
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"][
                    "ggml_type"
                ]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"][
                    "ggml_type"
                ]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"][
                    "ggml_type"
                ]
                # tensors = self.load_multi(key, [".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight"])
            res = {
                key: {
                    "gate": gate,
                    "up": up,
                    "down": down,
                    "gate_type": gate_type,
                    "up_type": up_type,
                    "down_type": down_type,
                }
            }
        return res

    def forward(self, input_tensor: torch.Tensor, expert_ids, weights):
        # forward
        device = input_tensor.device
        input_tensor = input_tensor.to("cuda")
        outs = torch.zeros_like(input_tensor)
        for expert_idx in range(expert_ids.size(0)):
            down_proj = self.down_projs[expert_idx]
            gate_proj = self.gate_projs[expert_idx]
            up_proj = self.up_projs[expert_idx]

            outs += (
                down_proj(self.act_fn(gate_proj(input_tensor)) * up_proj(input_tensor))
                * weights[expert_idx]
            )
        outs = outs.to(device)
        return outs


class KExpertsTorchBackup(KExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.layer_idx = int(key.split(".")[1])
        self.expert_num = n_routed_experts
        # self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        self.device = device
        self.gate = None
        # gate.size() = 64, 1408, 1024
        self.up = None
        self.down = None
        self.dtype = torch.get_default_dtype()

    def load(
        self,
        w: dict | nn.Parameter | tuple | None = None,
        device: str | None = None,
        warmup: bool = False,
    ):
        if device is None:
            device = self.device
        if w is None:
            w = self.load_weights(device=device)[self.key]

        if isinstance(w, dict):
            self.gate = w["gate"].to(device=device, dtype=self.dtype)
            self.up = w["up"].to(device=device, dtype=self.dtype)
            self.down = w["down"].to(device=device, dtype=self.dtype)

    def unload(self):
        if self.gate is not None:
            self.gate = None
            self.up = None
            self.down = None

    def forward(
        self,
        hidden_states_cpu: torch.Tensor,
        selected_experts_cpu: torch.Tensor,
        routing_weights_cpu: torch.Tensor,
    ) -> torch.Tensor:
        t1 = time.time()
        org_device = hidden_states_cpu.device
        hidden_states_cpu = hidden_states_cpu.to(self.device)
        selected_experts_cpu = selected_experts_cpu.to(self.device)
        routing_weights_cpu = routing_weights_cpu.to(self.device)

        batch_sequence_length, hidden_dim = hidden_states_cpu.size()

        final_hidden_states = torch.zeros(
            (batch_sequence_length, hidden_dim),
            dtype=self.gate.dtype,
            device=hidden_states_cpu.device,
        )
        org_dtype = hidden_states_cpu.dtype
        hidden_states_cpu = hidden_states_cpu.to(self.gate.dtype)
        routing_weights_cpu = routing_weights_cpu.to(self.gate.dtype)
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts_cpu, num_classes=self.expert_num
        ).permute(2, 1, 0)
        t2 = time.time()

        # 只遍历selected_experts_cpu
        unique_selected_experts = torch.unique(selected_experts_cpu)
        for expert_idx in unique_selected_experts:
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            G = current_state @ self.gate[expert_idx, ...].T
            A = self.act_fn(G)
            U = current_state @ self.up[expert_idx, ...].T
            H = A * U
            current_hidden_states = (
                H @ self.down[expert_idx, ...].T * routing_weights_cpu[top_x, idx, None]
            )
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        t3 = time.time()
        # 遍历所有expert
        # for expert_idx in range(self.expert_num):
        #     # start_time = time.time()  # 记录开始时间
        #     idx, top_x = torch.where(expert_mask[expert_idx])
        #     # Index the correct hidden states and compute the expert hidden state for
        #     # the current expert. We need to make sure to multiply the output hidden
        #     # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        #     current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
        #     G = current_state @ self.gate[expert_idx,...].T
        #     A = self.act_fn(G)
        #     U = current_state @ self.up[expert_idx,...].T
        #     H = A * U  # Element-wise multiplication
        #     current_hidden_states = H @ self.down[expert_idx,...].T * routing_weights_cpu[top_x, idx, None]
        #     # However `index_add_` only support torch tensors for indexing so we'll use
        #     # the `top_x` tensor here.
        #     final_hidden_states.index_add_(0, top_x, current_hidden_states)
        # print(f"KExpertsTorch forward time: {t2-t1} + {t3-t2}")
        return final_hidden_states.to(device=org_device, dtype=org_dtype)


class KExpertsTorch(KExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.layer_idx = int(key.split(".")[1])
        self.expert_num = n_routed_experts
        self.act_fn = ACT2FN[config.hidden_act]
        self.device = device
        self.dtype = torch.get_default_dtype()
        self.cache = KExpertsCache(
            config=config,
            dtype=self.dtype,
        )
        self.loading_lock = [
            [torch.cuda.Event() for _ in range(3)] for _ in range(self.expert_num)
        ]

    def load(
        self,
        w: dict | nn.Parameter | tuple | None = None,
        device: str | None = None,
        warmup: bool = False,
    ):
        if device is None:
            device = self.device
        if w is None:
            w = self.load_weights(device=device)[self.key]
        if isinstance(w, dict):
            for expert_idx in range(self.expert_num):
                self.cache.load_weights_to_storage(
                    w["gate"][expert_idx],
                    w["up"][expert_idx],
                    w["down"][expert_idx],
                    expert_idx + (self.layer_idx - 1) * self.expert_num,
                    dtype=self.dtype,
                )

    def unload(self):
        for expert_idx in range(self.expert_num):
            self.cache.unload_expert_weights(
                expert_idx=expert_idx + (self.layer_idx - 1) * self.expert_num,
            )

    def forward(
        self,
        hidden_states_cpu: torch.Tensor,
        selected_experts_cpu: torch.Tensor,
        routing_weights_cpu: torch.Tensor,
    ) -> torch.Tensor:
        org_device = hidden_states_cpu.device
        hidden_states_cpu = hidden_states_cpu.to(self.device)
        selected_experts_cpu = selected_experts_cpu.to(self.device)
        routing_weights_cpu = routing_weights_cpu.to(self.device)

        batch_sequence_length, hidden_dim = hidden_states_cpu.size()

        final_hidden_states = torch.zeros(
            (batch_sequence_length, hidden_dim),
            dtype=self.dtype,
            device=hidden_states_cpu.device,
        )
        org_dtype = hidden_states_cpu.dtype
        hidden_states_cpu = hidden_states_cpu.to(self.dtype)
        routing_weights_cpu = routing_weights_cpu.to(self.dtype)

        # One hot encode the selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(
            selected_experts_cpu, num_classes=self.expert_num
        ).permute(2, 1, 0)

        # 只遍历selected_experts_cpu
        unique_selected_experts = torch.unique(selected_experts_cpu).tolist()
        # 提前加载需要使用的专家权重
        self.cache.wait_prefetch()
        expert_weights = self.cache.get_experts_weights(
            unique_selected_experts, self.layer_idx, self.loading_lock
        )
        # self.cache.prefetch_expert(layer_idx=self.layer_idx)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in unique_selected_experts:
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)

            gate_weight, up_weight, down_weight = expert_weights[expert_idx]
            self.loading_lock[expert_idx][0].wait()
            G = current_state @ gate_weight.T
            A = self.act_fn(G)
            self.loading_lock[expert_idx][1].wait()
            U = current_state @ up_weight.T
            H = A * U
            self.loading_lock[expert_idx][2].wait()
            current_hidden_states = (
                H @ down_weight.T * routing_weights_cpu[top_x, idx, None]
            )
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        return final_hidden_states.to(device=org_device, dtype=org_dtype)


class KExpertsCache:
    _instance = None

    # 单例模式
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KExpertsCache, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        config: PretrainedConfig,
        load_size=None,
        dtype=torch.get_default_dtype(),
        devices_usage=None,
    ):
        if not hasattr(self, "initialized"):
            # 如果config有n_routed_experts
            if load_size is None:
                from math import sqrt

                self.load_size = [int(sqrt(config.n_routed_experts))] * (
                    config.num_hidden_layers
                )
                self.load_size[0] = 0
            else:
                assert len(load_size) == config.num_hidden_layers
                self.load_size = load_size
            self.dtype = dtype
            if hasattr(config, "n_routed_experts"):
                self.expert_num_per_layer = config.n_routed_experts
                self.total_expert_num = config.n_routed_experts * (
                    config.num_hidden_layers - 1
                )
                self.hidden_size = config.hidden_size
                self.moe_intermediate_size = config.moe_intermediate_size
            elif hasattr(config, "num_local_experts"):
                self.expert_num_per_layer = config.num_local_experts
                self.total_expert_num = config.num_local_experts * (
                    config.num_hidden_layers - 1
                )
                self.hidden_size = config.hidden_size
                self.moe_intermediate_size = config.intermediate_size
            elif hasattr(config, "num_experts"):
                self.expert_num_per_layer = config.num_experts
                self.total_expert_num = config.num_experts * (
                    config.num_hidden_layers - 1
                )
                self.hidden_size = config.hidden_size
                self.moe_intermediate_size = config.intermediate_size
            else:
                raise ValueError(
                    "config error, n_routed_experts or num_local_experts or num_experts not found"
                )

            self.gate_shape = torch.Size([self.moe_intermediate_size, self.hidden_size])
            self.up_shape = torch.Size([self.moe_intermediate_size, self.hidden_size])
            self.down_shape = torch.Size([self.hidden_size, self.moe_intermediate_size])
            # 如果没有指定设备，默认使用 cuda 0 load专家
            if devices_usage is None:
                devices_usage = {}
                devices_usage["cuda:0"] = [
                    i for i in range(1, config.num_hidden_layers)
                ]
                # devices存的是每个设备上的层索引
            self.devices_usage = devices_usage
            self.layer2device = {}
            for device, layers in devices_usage.items():
                for layer in layers:
                    self.layer2device[layer] = device
            # 内存，存储所有专家
            size_per_expert = (
                dtype.itemsize * self.moe_intermediate_size * self.hidden_size
            )
            gate_large = torch.UntypedStorage(
                size_per_expert * self.total_expert_num
            ).pin_memory()
            up_large = torch.UntypedStorage(
                size_per_expert * self.total_expert_num
            ).pin_memory()
            down_large = torch.UntypedStorage(
                size_per_expert * self.total_expert_num
            ).pin_memory()

            self.gate_storage = [
                gate_large[i * size_per_expert : (i + 1) * size_per_expert]
                for i in range(self.total_expert_num)
            ]
            self.up_storage = [
                up_large[i * size_per_expert : (i + 1) * size_per_expert]
                for i in range(self.total_expert_num)
            ]
            self.down_storage = [
                down_large[i * size_per_expert : (i + 1) * size_per_expert]
                for i in range(self.total_expert_num)
            ]

            # 显存
            self.gate_memory = {}
            self.up_memory = {}
            self.down_memory = {}
            self.gate_views = {}
            self.up_views = {}
            self.down_views = {}
            for device in devices_usage:
                load_experts_num = sum(
                    self.load_size[layer_idx]
                    for layer_idx in self.devices_usage[device]
                )
                total_size = load_experts_num * size_per_expert
                total_gate_memory = torch.UntypedStorage(total_size, device=device)
                # 将gate_memory 切分为load_experts_num个专家
                self.gate_memory[device] = [
                    total_gate_memory[i * size_per_expert : (i + 1) * size_per_expert]
                    for i in range(load_experts_num)
                ]
                self.gate_views[device] = [
                    torch.as_tensor(storage, dtype=dtype, device=device).view(
                        self.gate_shape
                    )
                    for storage in self.gate_memory[device]
                ]
                total_up_memory = torch.UntypedStorage(total_size, device=device)
                self.up_memory[device] = [
                    total_up_memory[i * size_per_expert : (i + 1) * size_per_expert]
                    for i in range(load_experts_num)
                ]
                self.up_views[device] = [
                    torch.as_tensor(storage, dtype=dtype, device=device).view(
                        self.up_shape
                    )
                    for storage in self.up_memory[device]
                ]
                total_down_memory = torch.UntypedStorage(total_size, device=device)
                self.down_memory[device] = [
                    total_down_memory[i * size_per_expert : (i + 1) * size_per_expert]
                    for i in range(load_experts_num)
                ]
                self.down_views[device] = [
                    torch.as_tensor(storage, dtype=dtype, device=device).view(
                        self.down_shape
                    )
                    for storage in self.down_memory[device]
                ]

            # 当前加载到显存中的专家索引和位置
            self.loaded_experts_idx = {}
            for device, layers in self.devices_usage.items():
                self.loaded_experts_idx[device] = {}
                for layer in layers:
                    self.loaded_experts_idx[device][layer] = {}
            self.free_memory_slots = self.initialize_free_memory_slots()

            # CUDA Stream
            self.copy_stream = torch.cuda.Stream()
            # 专家使用次数
            self.usage_count = []
            for layer in range(0, config.num_hidden_layers):
                self.usage_count.append([0] * self.expert_num_per_layer)
                self.usage_count[layer][0] = -1
            # with open("usage.json", "r") as f:
            #     self.usage_count = json.load(f)
            self.prefetch_size = 1
            self.prefetch_lock = [
                [torch.cuda.Event() for _ in range(3)]
                for _ in range(self.prefetch_size)
            ]
            self.prefetching = False
            self.hits = 0
            self.misses = 0
            self.initialized = True

    def initialize_free_memory_slots(self):
        # 每个device有连续的索引，按照专家数量依次分配
        free_memory_slots = {}
        for device, layers in self.devices_usage.items():
            free_memory_slots[device] = {}
            current_index = 0
            for layer in self.devices_usage[device]:
                next_index = current_index + self.load_size[layer]
                free_memory_slots[device][layer] = list(
                    range(current_index, next_index)
                )
                current_index = next_index
        return free_memory_slots

    def weight_to_storage(self, weight, dtype=torch.float16, device="cpu"):
        weight = weight.to(dtype)

        storage_size = weight.nbytes
        storage = torch.UntypedStorage(storage_size, device=device)

        a_view = torch.as_tensor(storage, dtype=dtype, device=device).view(weight.shape)
        a_view.copy_(weight)
        assert a_view.data_ptr() == storage.data_ptr()
        return storage

    def storage_to_weight(self, storage, shape, dtype=torch.float16, device="cuda"):
        weight = torch.as_tensor(storage, dtype=dtype, device=device).view(shape)
        return weight

    def load_expert_weights(
        self, expert_idx, init=False, non_blocking=True, loading_lock=None
    ):
        # 找位置
        layer_idx = expert_idx // self.expert_num_per_layer
        device = self.layer2device[layer_idx]
        if not self.free_memory_slots[device][layer_idx]:
            if init:
                return
            # 如果没有空闲位置，卸载一个已有的专家权重(用LRU)
            # TODO better LRU
            offload_expert = next(iter(self.loaded_experts_idx[device][layer_idx]))
            self.unload_expert_weights(
                offload_expert,
                device,
                layer_idx,
            )

        # 空闲位置
        memory_slot = self.free_memory_slots[device][layer_idx].pop(0)
        # 用 CUDA Stream
        with torch.cuda.stream(self.copy_stream):
            self.gate_memory[device][memory_slot].copy_(
                self.gate_storage[expert_idx],
                non_blocking=non_blocking,
            )
            loading_lock[0].record()
            self.up_memory[device][memory_slot].copy_(
                self.up_storage[expert_idx],
                non_blocking=non_blocking,
            )
            loading_lock[1].record()
            self.down_memory[device][memory_slot].copy_(
                self.down_storage[expert_idx],
                non_blocking=non_blocking,
            )
            loading_lock[2].record()
        # 记录加载的专家索引和位置
        self.loaded_experts_idx[device][layer_idx][expert_idx] = memory_slot

    def unload_expert_weights(self, expert_idx, device=None, layer_idx=None):
        # 获取专家在显存中的位置
        if device is None:
            layer_idx = expert_idx // self.expert_num_per_layer
            device = self.layer2device[layer_idx]
        if expert_idx not in self.loaded_experts_idx[device][layer_idx]:
            return
        memory_slot = self.loaded_experts_idx[device][layer_idx][expert_idx]
        self.free_memory_slots[device][layer_idx].append(memory_slot)
        self.loaded_experts_idx[device][layer_idx].pop(expert_idx)

    def get_experts_weights(self, expert_idxs, layer_idx, loading_lock):
        # prefetch_experts = sorted(
        #     enumerate(self.usage_count[layer_idx]),
        #     key=lambda x: x[1],
        #     reverse=True,
        # )[:self.prefetch_size]
        # for expert_idx, _ in prefetch_experts:
        #     if expert_idx in expert_idxs:
        #         self.hits += 1
        #     else:
        #         self.misses += 1
        experts = {}
        if len(expert_idxs) > self.load_size[layer_idx]:
            deepcopy = True
        else:
            deepcopy = False
        # load_idxs = []
        # unload_idxs = []
        # for expert_idx in expert_idxs:
        #     self.usage_count[layer_idx][expert_idx] += 1
        #     device = self.layer2device[layer_idx]
        #     if expert_idx in self.loaded_experts_idx[device][layer_idx].keys():
        #         load_idxs.append(expert_idx)
        #     else:
        #         unload_idxs.append(expert_idx)

        for expert_idx in expert_idxs:
            experts[expert_idx] = self.get_expert_weights(
                expert_idx + (layer_idx - 1) * self.expert_num_per_layer,
                deepcopy=deepcopy,
                loading_lock=loading_lock[expert_idx],
            )
        return experts

    def prefetch_expert(self, layer_idx):
        next_layer = layer_idx + 1 if layer_idx + 1 < len(self.load_size) else 1
        self.prefetching = True
        # 取使用最多的prefetch_size个expert
        # TODO better strategy
        experts = sorted(
            enumerate(self.usage_count[next_layer]),
            key=lambda x: x[1],
            reverse=True,
        )[: self.prefetch_size]
        experts = [expert_idx for expert_idx, _ in experts]
        for idx, expert_idx in enumerate(experts):
            device = self.layer2device[next_layer]
            if not expert_idx in self.loaded_experts_idx[device][next_layer]:
                self.load_expert_weights(
                    expert_idx + (next_layer - 1) * self.expert_num_per_layer,
                    non_blocking=True,
                    loading_lock=self.prefetch_lock[idx],
                )

    def wait_prefetch(self):
        for lock in self.prefetch_lock:
            for l in lock:
                l.wait()
        self.prefetching = False

    def get_hits_and_misses(self):
        with open("usage.json", "w") as f:
            json.dump(self.usage_count, f)
        return self.hits, self.misses

    def get_expert_weights(self, expert_idx, deepcopy=False, loading_lock=None):
        layer = expert_idx // self.expert_num_per_layer
        device = self.layer2device[layer]
        if expert_idx in self.loaded_experts_idx[device][layer].keys():
            # 将已加载的专家权重移动到字典的末尾(LRU更新)
            self.loaded_experts_idx[device][layer][expert_idx] = (
                self.loaded_experts_idx[device][layer].pop(expert_idx)
            )
            memory_slot = self.loaded_experts_idx[device][layer][expert_idx]
            if deepcopy:
                loading_lock[0].wait()
                loading_lock[1].wait()
                loading_lock[2].wait()
                expert = (
                    self.gate_views[device][memory_slot].clone(),
                    self.up_views[device][memory_slot].clone(),
                    self.down_views[device][memory_slot].clone(),
                )
                # self.loading_lock[expert_idx % self.expert_num_per_layer].record()
                return expert
            else:
                expert = (
                    self.gate_views[device][memory_slot],
                    self.up_views[device][memory_slot],
                    self.down_views[device][memory_slot],
                )
                return expert
        self.load_expert_weights(
            expert_idx, non_blocking=True, loading_lock=loading_lock
        )
        return self.get_expert_weights(
            expert_idx, deepcopy=deepcopy, loading_lock=loading_lock
        )

    # CUDA_VISIBLE_DEVICES=6 nsys profile -t cuda,nvtx  python /data/yanfansun/ktrans/ktransformers/ktransformers/local_chat.py
    def load_weights_to_storage(self, gate, up, down, expert_idx, dtype):
        gate = self.weight_to_storage(gate, dtype, torch.device("cpu"))
        up = self.weight_to_storage(up, dtype, torch.device("cpu"))
        down = self.weight_to_storage(down, dtype, torch.device("cpu"))
        # 将传入的参数从 CPU 复制到已经分配好地址的 storage 中
        self.gate_storage[expert_idx].copy_(gate)
        self.up_storage[expert_idx].copy_(up)
        self.down_storage[expert_idx].copy_(down)

        # 如果有空位，先加载到显存中
        loading_lock = [torch.cuda.Event() for _ in range(3)]
        self.load_expert_weights(expert_idx, init=True, loading_lock=loading_lock)


EXPERTS_MAP = {
    "KExpertsCPU": KExpertsCPU,
    "KExpertsTorch": KExpertsTorch,
    "KExpertsMarlin": KExpertsMarlin,
}


class KTransformersExperts(BaseInjectedModule, KExpertsBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        #  device: str = "cuda",
        prefill_device: str = "cuda",
        prefill_op: str | None = "KExpertsTorch",
        generate_device: str = "cpu",
        generate_op: str | None = "KExpertsCPU",
        **kwargs,
    ):
        BaseInjectedModule.__init__(
            self, key, gguf_loader, config, orig_module, generate_device, **kwargs
        )
        KExpertsBase.__init__(
            self, key, gguf_loader, config, orig_module, generate_device, **kwargs
        )
        if generate_op is not None:
            self.generate_experts = EXPERTS_MAP[generate_op](
                key,
                gguf_loader,
                config,
                len(orig_module),
                device=generate_device,
                **kwargs,
            )
        else:
            self.generate_experts = None
        if prefill_op is not None:
            self.prefill_experts = EXPERTS_MAP[prefill_op](
                key,
                gguf_loader,
                config,
                len(orig_module),
                device=prefill_device,
                **kwargs,
            )
        else:
            self.prefill_experts = None
        self.gpu_mlp_type = prefill_op
        self.cpu_mlp_type = generate_op
        self.mode = InferenceState.UNLOAD

    def load(self, w: dict = None, mode: InferenceState = None, warmup: bool = True):
        # TODO support w as input
        if not mode:
            mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            self.prefill_experts.unload()
            self.generate_experts.load(w, warmup=warmup)
            self.device = self.generate_experts.device
            self.mode = mode
        elif mode == InferenceState.PREFILL:
            self.generate_experts.unload()
            self.prefill_experts.load(w, warmup=warmup)
            self.device = self.prefill_experts.device
            self.mode = mode
        elif mode == InferenceState.UNLOAD:
            self.unload()
            self.mode = mode
            self.device = self.generate_experts.device
        else:
            raise ValueError(
                "mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD"
            )

    def unload(self):
        if self.generate_experts is not None:
            self.generate_experts.unload()
        if self.prefill_experts is not None:
            self.prefill_experts.unload()
        self.device = self.generate_experts.device

    def forward(self, input_tensor, expert_ids, weights):
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, "generate_experts is None"
            return self.generate_experts.forward(input_tensor, expert_ids, weights)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, "prefill_experts is None"
            return self.prefill_experts.forward(input_tensor, expert_ids, weights)
        else:
            raise ValueError("load or set_inference_mode before forward")

    def set_inference_mode(self, mode: InferenceState):
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE, warmup=False)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL, warmup=False)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError(
                "mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD"
            )


from ktransformers.models.modeling_deepseek import DeepseekV2MoE
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from ktransformers.models.modeling_mixtral import MixtralSparseMoeBlock

shared_time = []


class KQwen2MoeSparseMoeBlock(BaseInjectedModule, Qwen2MoeSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        if sequence_length == 1 and hasattr(
            self.experts.generate_experts, "submit_for_one_decode"
        ):
            self.experts.generate_experts.submit_for_one_decode(
                hidden_states[0], selected_experts[0], routing_weights[0]
            )
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = (
                F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            )
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += shared_expert_output
            y.resize_(*orig_shape)
            return y, router_logits

        hidden_states_expert = (
            hidden_states.to(self.experts.device)
            if isinstance(self.experts, KExpertsBase)
            else hidden_states_expert.cpu()
        )
        selected_experts_expert = (
            selected_experts.to(self.experts.device)
            if isinstance(self.experts, KExpertsBase)
            else selected_experts_expert.cpu()
        )
        routing_weights_expert = (
            routing_weights.to(self.experts.device)
            if isinstance(self.experts, KExpertsBase)
            else routing_weights_expert.cpu()
        )
        start = time.time()
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )
        end = time.time()
        shared_time.append(end - start)
        # print(f"shared_expert forward avg time: {sum(shared_time)/len(shared_time)}")
        if isinstance(self.experts, KExpertsBase):
            y = (
                self.moe_on_cpuinfer(
                    hidden_states_expert,
                    selected_experts_expert,
                    routing_weights_expert,
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_expert.size(0) > 10:
            y = self.moe_infer(
                hidden_states_expert,
                selected_experts_expert,
                routing_weights_expert,
                orig_shape,
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_expert, selected_experts_expert, routing_weights_expert
            ).to(device=hidden_states.device)
        y += shared_expert_output
        y.resize_(*orig_shape)
        return y, router_logits

    @torch.no_grad()
    def moe_on_cpuinfer(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self,
        hidden_states_cpu: torch.Tensor,
        selected_experts_cpu: torch.Tensor,
        routing_weights_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(hidden_states_cpu[token_idx])
                    * routing_weights_cpu[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(
        self,
        hidden_states_cpu: torch.Tensor,
        selected_experts_cpu: torch.Tensor,
        routing_weights_cpu: torch.Tensor,
        orig_shape: tuple,
    ) -> torch.Tensor:

        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states_cpu.dtype,
            device=hidden_states_cpu.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts_cpu, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer.forward(current_state)
                * routing_weights_cpu[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states_cpu.dtype)
            )

        return final_hidden_states


class KDeepseekV2MoE(BaseInjectedModule, DeepseekV2MoE):
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        t1 = time.time()
        if sequence_length == 1 and hasattr(
            self.experts.generate_experts, "submit_for_one_decode"
        ):
            self.experts.generate_experts.submit_for_one_decode(
                hidden_states[0], topk_idx[0], topk_weight[0]
            )
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            t3 = time.time()
            # print(f"moe_infer time: {t3-t1}")
            return y

        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)

        if isinstance(self.experts, KExpertsBase):
            y = (
                self.moe_on_cpuinfer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_

        t2 = time.time()
        # print(f"moe_infer time: {t2-t1}")
        return y

    @torch.no_grad()
    def moe_on_cpuinfer(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class KMisrtalSparseMoEBlock(BaseInjectedModule, MixtralSparseMoeBlock):

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        if sequence_length == 1 and hasattr(
            self.experts.generate_experts, "submit_for_one_decode"
        ):
            self.experts.generate_experts.submit_for_one_decode(
                hidden_states[0], selected_experts[0], routing_weights[0]
            )
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y.resize_(*orig_shape)
            return y, router_logits

        hidden_states_expert = (
            hidden_states.to(self.experts.device)
            if isinstance(self.experts, KExpertsBase)
            else hidden_states_expert.cpu()
        )
        selected_experts_expert = (
            selected_experts.to(self.experts.device)
            if isinstance(self.experts, KExpertsBase)
            else selected_experts_expert.cpu()
        )
        routing_weights_expert = (
            routing_weights.to(self.experts.device)
            if isinstance(self.experts, KExpertsBase)
            else routing_weights_expert.cpu()
        )

        if isinstance(self.experts, KExpertsBase):
            y = (
                self.moe_on_cpuinfer(
                    hidden_states_expert,
                    selected_experts_expert,
                    routing_weights_expert,
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_expert.size(0) > 10:
            y = self.moe_infer(
                hidden_states_expert,
                selected_experts_expert,
                routing_weights_expert,
                orig_shape,
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_expert, selected_experts_expert, routing_weights_expert
            ).to(device=hidden_states.device)

        y.resize_(*orig_shape)
        return y, router_logits

    @torch.no_grad()
    def moe_on_cpuinfer(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self,
        hidden_states_cpu: torch.Tensor,
        selected_experts_cpu: torch.Tensor,
        routing_weights_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(hidden_states_cpu[token_idx])
                    * routing_weights_cpu[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(
        self,
        hidden_states_cpu: torch.Tensor,
        selected_experts_cpu: torch.Tensor,
        routing_weights_cpu: torch.Tensor,
        orig_shape: tuple,
    ) -> torch.Tensor:

        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states_cpu.dtype,
            device=hidden_states_cpu.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts_cpu, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer.forward(current_state)
                * routing_weights_cpu[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states_cpu.dtype)
            )

        return final_hidden_states
