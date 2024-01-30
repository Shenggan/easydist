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

import torch


def md_bn_bwd(*args, **kwargs):
    args_list = [arg for arg in args]
    for idx, arg in enumerate(args_list):
        if isinstance(arg, torch.Tensor):
            args_list[idx] = arg.contiguous()    
    
    return torch.ops.aten.cudnn_batch_norm_backward.default(*args_list, **kwargs)


def fix_bn_bwd(fx_module: torch.fx.GraphModule):

    for node in fx_module.graph.nodes:
        if node.op == 'call_function':
            if node.target == torch.ops.aten.cudnn_batch_norm_backward.default:
                node.target = md_bn_bwd

    fx_module.recompile()

    return fx_module
