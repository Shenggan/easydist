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
import torch._custom_ops

from easydist.torch.device_mesh import get_device_mesh
from easydist.torch.passes.sharding import (all_reduce_start, all_reduce_end, 
                                            all_gather_start, all_gather_end,
                                            all_to_all_start, all_to_all_end,
                                            scatter_wrapper)


@torch._custom_ops.custom_op("easydist::tag")
def tag(input: torch.Tensor, tag: str) -> torch.Tensor:
    ...

@torch._custom_ops.impl_abstract("easydist::tag")
def tag_impl_abstract(input: torch.Tensor, tag: str) -> torch.Tensor:
    if tag.startswith("allreduce"):
        return torch.empty_like(input)
    elif tag.startswith("allgather"): # "gather[1, 2]"
        gather_dim, group_size = [int(i) for i in tag.split("[")[1].split("]")[0].split(",")]
        size = list(input.size())
        size[gather_dim] *= group_size
        return torch.empty(*size, dtype=input.dtype, device=input.device)
    elif tag.startswith("scatter"): # "gather[1, 2]"
        scatter_dim, group_size = [int(i) for i in tag.split("[")[1].split("]")[0].split(",")]
        size = list(input.size())
        size[scatter_dim] = size[scatter_dim] // group_size
        return torch.empty(*size, dtype=input.dtype, device=input.device)
    elif tag.startswith("all-to-all"):# "all-to-all[1, 2, 2]"
        scatter_dim, gather_dim, group_size = [int(i) for i in tag.split("[")[1].split("]")[0].split(",")]
        size = list(input.size())
        size[gather_dim] *= group_size
        size[scatter_dim] = size[scatter_dim] // group_size
        return torch.empty(*size, dtype=input.dtype, device=input.device) 

    raise NotImplementedError(f"Unknown tag for easydist::tag ops: {tag}")  


@torch._custom_ops.impl("easydist::tag")
def tag_impl(input: torch.Tensor, tag: str) -> torch.Tensor:
    if tag.startswith("allreduce"):
        return input
    elif tag.startswith("allgather"): # "gather[1, 2]"
        gather_dim, group_size = [int(i) for i in tag.split("[")[1].split("]")[0].split(",")]
        size = list(input.size())
        size[gather_dim] *= group_size
        return torch.empty(*size, dtype=input.dtype, device=input.device)
    elif tag.startswith("scatter"): # "gather[1, 2]"
        scatter_dim, group_size = [int(i) for i in tag.split("[")[1].split("]")[0].split(",")]
        size = list(input.size())
        size[scatter_dim] = size[scatter_dim] // group_size
        return torch.empty(*size, dtype=input.dtype, device=input.device)
    elif tag.startswith("all-to-all"):# "all-to-all[1, 2, 2]"
        scatter_dim, gather_dim, group_size = [int(i) for i in tag.split("[")[1].split("]")[0].split(",")]
        size = list(input.size())
        size[gather_dim] *= group_size
        size[scatter_dim] = size[scatter_dim] // group_size
        return torch.empty(*size, dtype=input.dtype, device=input.device) 

    raise NotImplementedError(f"Unknown tag for easydist::tag ops: {tag}")  



def process_tag(traced_graph: torch.fx.GraphModule) -> torch.fx.GraphModule:

    device_mesh = get_device_mesh()

    for node in traced_graph.graph.nodes:
        if node.target == torch.ops.easydist.tag.default:
            if node.args[1] == "allreduce[sum]":
                assert "tp" in device_mesh.mesh_dim_names
                tp_mesh = device_mesh["tp"]
                reduceOp = "sum"
                ranks = tp_mesh.mesh.flatten().tolist()
                with traced_graph.graph.inserting_before(node):
                    all_reduce_start_node = traced_graph.graph.call_function(all_reduce_start,
                                                                                args=(node.args[0],
                                                                                    reduceOp,
                                                                                    ranks))
                    all_reduce_end_node = traced_graph.graph.call_function(
                        all_reduce_end, args=(all_reduce_start_node, reduceOp, ranks))
                
                node.replace_all_uses_with(all_reduce_end_node)
            elif node.args[1].startswith("allgather"):
                assert "tp" in device_mesh.mesh_dim_names
                tp_mesh = device_mesh["tp"]
                reduceOp = "sum"
                ranks = tp_mesh.mesh.flatten().tolist()

                gather_dim, group_size = [int(i) for i in node.args[1].split("[")[1].split("]")[0].split(",")]
                assert group_size == len(ranks)
                with traced_graph.graph.inserting_before(node):
                    all_gather_start_node = traced_graph.graph.call_function(all_gather_start,
                                                                                args=(node.args[0],
                                                                                    gather_dim,
                                                                                    ranks))
                    all_gather_end_node = traced_graph.graph.call_function(
                        all_gather_end, args=(all_gather_start_node, gather_dim, ranks))

                node.replace_all_uses_with(all_gather_end_node)
            elif node.args[1].startswith("scatter"):
                assert "tp" in device_mesh.mesh_dim_names
                tp_mesh = device_mesh["tp"]
                reduceOp = "sum"
                ranks = tp_mesh.mesh.flatten().tolist()
                my_coordinate = tp_mesh.get_coordinate()
                scatter_dim, group_size = [int(i) for i in node.args[1].split("[")[1].split("]")[0].split(",")]
                assert group_size == len(ranks)
                with traced_graph.graph.inserting_before(node):
                    scatter_node = traced_graph.graph.call_function(scatter_wrapper,
                                                                                args=(node.args[0],
                                                                                    group_size,
                                                                                    scatter_dim,
                                                                                    my_coordinate[0]))

                node.replace_all_uses_with(scatter_node)

            elif node.args[1].startswith("all-to-all"):
                assert "tp" in device_mesh.mesh_dim_names
                tp_mesh = device_mesh["tp"]
                reduceOp = "sum"
                ranks = tp_mesh.mesh.flatten().tolist()
                my_coordinate = tp_mesh.get_coordinate()
                scatter_dim, gather_dim, group_size = [int(i) for i in node.args[1].split("[")[1].split("]")[0].split(",")]
                assert group_size == len(ranks)
                with traced_graph.graph.inserting_before(node):
                    all_to_all_start_node = traced_graph.graph.call_function(all_to_all_start,
                                                                                args=(node.args[0],
                                                                                    gather_dim,
                                                                                    scatter_dim,
                                                                                    group_size,
                                                                                    my_coordinate[0],
                                                                                    ranks))
                    all_to_all_end_node = traced_graph.graph.call_function(
                        all_to_all_end, args=(all_to_all_start_node, gather_dim, scatter_dim, group_size, my_coordinate[0], ranks))

                node.replace_all_uses_with(all_to_all_end_node)



    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()

    return traced_graph