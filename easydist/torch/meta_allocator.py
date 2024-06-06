# Copyright (c) 2023, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import ctypes
import logging

import pynvml

import torch
from torch.utils.cpp_extension import load, _join_cuda_home
import easydist

logger = logging.getLogger(__name__)

def _compile():

    print("torch load")

    device_index = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')[0]

    # (NOTE) workaround of torch.cuda.get_cuda_arch_list() which will init allocator
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
    capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{capability[0]}.{capability[1]}+PTX'

    sources_files = [
        'csrc/profiling_allocator.cpp',
        'csrc/stream_tracer.cpp',
        'csrc/cupti_callback_api.cpp',
        'csrc/python_tracer_init.cpp'
    ]

    profiling_allocator_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiler")
    _comiled_module = load(name="profiling_allocator",
                               extra_include_paths=[_join_cuda_home('extras', 'CUPTI', 'include')],
                                sources=[os.path.join(profiling_allocator_dir, f) for f in sources_files],
                                with_cuda=True)

    logger.info(f"[profiling_allocator] compiled in {_comiled_module.__file__}")

    return _comiled_module


class LazyModule():
    def __init__(self):
        self.module = None

    def _compile_if_needed(self):
        if self.module is None:
            self.module = _compile()

    def __getattr__(self, name: str):
        self._compile_if_needed()
        return self.module.__getattribute__(name)

profiling_allocator = LazyModule()


def init_meta_allocator():
    if not easydist.config.enable_memory_opt:
        return
    swap_to_profiling_allocator()


def swap_to_profiling_allocator():

    profiling_allocator._compile_if_needed()

    # swap from caching allocator to profiling allocator
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        profiling_allocator.module.__file__, 'meta_malloc', 'meta_free')
    
    profiling_allocator._save_back_allocator()
    torch.cuda.memory.change_current_allocator(new_alloc)
    
    raw_allocator = ctypes.CDLL(profiling_allocator.module.__file__)
    init_fn = ctypes.cast(getattr(raw_allocator, 'init_fn'), ctypes.c_void_p).value
    new_alloc.allocator().set_init_fn(init_fn)

    logger.info(f"[profiling_allocator] load profiling_allocator as CUDAPluggableAllocator")
