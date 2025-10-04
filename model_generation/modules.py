import torch
import torch.nn as nn
from copy import deepcopy
from model_generation.operations201 import OPS as OPS_201
from model_generation.operations_energy import OPS_Energy

# -----------------------------------------------------------------------------
#   NAS-Bench‑201 Cell (unchanged, but ops configurable) -----------------------
# -----------------------------------------------------------------------------
# 在文件顶部附近加一个小工具函数（任意位置，模块内可见即可）
def _sum_tensors(tensors):
    """Accumulate without injecting Python-int 0 (safer for ONNX/TFLite)."""
    out = None
    for t in tensors:
        out = t if out is None else out + t
    return out


class InferCell201(nn.Module):
    def __init__(self, genotype, C_in, C_out, stride, ops=OPS_201):
        super().__init__()
        self.ops_map = ops
        self.layers, self.node_IN, self.node_IX = nn.ModuleList(), [], []
        self.genotype = deepcopy(genotype)

        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index, cur_innod = [], []
            for op_name, op_in in node_info:
                if op_in == 0:
                    layer = self.ops_map[op_name](C_in, C_out, stride, True, True)
                else:
                    layer = self.ops_map[op_name](C_out, C_out, 1, True, True)
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)

        self.nodes, self.in_dim, self.out_dim = len(genotype), C_in, C_out
# ------- in InferCell201.forward -------
    def forward(self, x):
        nodes = [x]
        for node_layers, node_innods in zip(self.node_IX, self.node_IN):
            parts = [ self.layers[l](nodes[n]) for l, n in zip(node_layers, node_innods) ]
            node_out = _sum_tensors(parts)           # ← 替换原来的 sum(...)
            nodes.append(node_out)
        return nodes[-1]

    def extra_repr(self):
        return f"nodes={self.nodes}, inC={self.in_dim}, outC={self.out_dim}"


# -----------------------------------------------------------------------------
#   Energy Cell with pluggable ops (default = OPS_Energy) ----------------------
# -----------------------------------------------------------------------------

class InferCellEnergy(nn.Module):
    def __init__(
        self,
        genotype,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        ops=OPS_Energy,
    ):
        super().__init__()
        self.ops_map = ops
        self.layers, self.node_IN, self.node_IX = nn.ModuleList(), [], []
        self.genotype = deepcopy(genotype)

        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index, cur_innod = [], []
            for op_name, op_in in node_info:
                if op_in == 0:
                    layer = self.ops_map[op_name](
                        C_in=C_in,
                        C_out=C_out,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                else:
                    layer = self.ops_map[op_name](
                        C_in=C_out,
                        C_out=C_out,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)

        self.nodes, self.in_dim, self.out_dim = len(genotype), C_in, C_out

# ------- in InferCellEnergy.forward -------
    def forward(self, x):
        nodes = [x]
        for node_layers, node_innods in zip(self.node_IX, self.node_IN):
            parts = [ self.layers[l](nodes[n]) for l, n in zip(node_layers, node_innods) ]
            node_out = _sum_tensors(parts)           # ← 替换原来的 sum(...)
            nodes.append(node_out)
        return nodes[-1]

    def extra_repr(self):
        return f"nodes={self.nodes}, inC={self.in_dim}, outC={self.out_dim}"