############################################################
#  Extended Model‑Generation (string‑aware)                #
#  - keeps original EnergyNet / NAS201Net logic            #
#  - adds helpers to convert Genotype  ⇄  Cell‑string      #
#  - stores `genotype_str` attr so downstream code can      #
#    directly feed into energy predictor without parsing   #
############################################################
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np, torch, torch.nn as nn

import random
from itertools import product, combinations
from model_generation.modules import InferCell201, InferCellEnergy
from model_generation.operations201 import ResNetBasicblock201
from model_generation.operations_energy import ResNetBasicblockEnergy, OPS as OPS_E, SearchSpaceNames


def generate_random_genotype(space: List[str], nodes: int = 4) -> str:
    layers = []
    for i in range(1, nodes):
        edges = []
        max_prev = min(i, 2)               # node-1 只有 1 条   node-2/3 刚好 2 条
        js = random.sample(range(i), k=max_prev)
        for j in js:
            op = random.choice(space)
            edges.append(f"{op}~{j}")
        layers.append('|' + '|'.join(edges) + '|')
    return '+'.join(layers)

def enumerate_all_genotypes(space: List[str], nodes: int = 4):
    """
    Exhaustively generate every genotype string where
      • node-1 连接 1 条前驱 (只能来自 0)
      • node-i (i≥2) 恰好 2 条不同前驱 (从 0..i-1 任选 2)
    小空间调试用；组合量随 len(space) 指数级增长。
    """
    if nodes < 2:
        return

    # build per-node edge+op choice list
    node_choices: List[List[Tuple[str, ...]]] = []
    for i in range(1, nodes):
        num_edges = 1 if i == 1 else 2
        idx_combos = combinations(range(i), num_edges)
        this_node = []
        for combo in idx_combos:
            for ops in product(space, repeat=num_edges):
                this_node.append(tuple(f"{op}~{idx}" for op, idx in zip(ops, combo)))
        node_choices.append(this_node)

    # Cartesian product to form full cell strings
    for choice in product(*node_choices):
        layers = ['|' + '|'.join(edges) + '|' for edges in choice]
        yield '+'.join(layers)

def genotype_to_str(genotype) -> str:
    """Turn nested‑list genotype into NAS‑Bench‑201 string.
    Accepts either `Structure` (has .nodes) or raw list.
    """
    nodes = getattr(genotype, "nodes", genotype)
    parts = []
    for node in nodes:
        edge_tokens = [f"{op}~{idx}" for op, idx in node]
        parts.append('|' + '|'.join(edge_tokens) + '|')
    return '+'.join(parts)

def str_to_genotype(xstr: str) -> List[Tuple[Tuple[str,int], ...]]:
    """Reverse of `genotype_to_str` – pure nested list, no Structure dep."""
    nodes = []
    for nstr in xstr.split('+'):
        edges = []
        for token in filter(None, nstr.split('|')):
            op, idx = token.split('~'); edges.append((op, int(idx)))
        nodes.append(tuple(edges))
    return nodes

def resolve_search_space(space_name: str):
    if space_name not in SearchSpaceNames:
        raise ValueError(f"Unknown search space '{space_name}'. Choices: {list(SearchSpaceNames)}")
    return SearchSpaceNames[space_name]

class NAS201Net(nn.Module):
    def __init__(self, C: int, N: int, genotype, num_classes: int, batch_size: int):
        super().__init__()
        self.genotype      = genotype  # keep reference
        self.genotype_str  = genotype_to_str(genotype)
        self._C            = C
        self._layerN       = N
        self.batch_size    = batch_size
        self.K             = np.zeros((batch_size, batch_size), np.float32)

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False), nn.BatchNorm2d(C)
        )
        C_prev, self.cells = C, nn.ModuleList()
        for i in range(3*N+2):
            reduction = (i+1)%(N+1)==0
            if reduction:
                C_out = C_prev*2; cell = ResNetBasicblock201(C_prev, C_out, 2, True)
            else:
                C_out = C_prev;  cell = InferCell201(genotype, C_prev, C_out, 1)
            self.cells.append(cell); C_prev=C_out
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._Layer = len(self.cells)

    def forward(self, x):
        for mod in (self.stem, *self.cells, self.lastact, self.global_pooling):
            x = mod(x)
        return self.classifier(x.view(x.size(0), -1))

class EnergyNet(nn.Module):
    def __init__(self, N:int, genotype, num_classes:int, batch_size:int,
                 config:Dict[str,Any], search_space:str="energy"):
        super().__init__()
        self.genotype     = genotype
        self.genotype_str = genotype_to_str(genotype)
        self._layerN      = N
        self.batch_size   = batch_size
        self.K            = np.zeros((batch_size, batch_size), np.float32)

        C_out = config.get("COUT", 16)
        self.stem = nn.Sequential(nn.Conv2d(3,C_out,3,padding=1,bias=False), nn.BatchNorm2d(C_out))
        C_prev, self.cells = C_out, nn.ModuleList()
        for i in range(3*N+2):
            reduction = (i+1)%(N+1)==0
            if reduction:
                C_next = C_prev*2; cell = ResNetBasicblockEnergy(C_prev,C_next,2)
            else:
                C_next = C_prev
                k,s = config.get("KERNEL_SIZE",3), config.get("STRIDE",1)
                p=(k-1)//2
                cell = InferCellEnergy(genotype, C_prev, C_next, k,s,p, ops=OPS_E)
            self.cells.append(cell); C_prev=C_next
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._Layer=len(self.cells)

    def forward(self,x):
        x=self.stem(x)
        for c in self.cells: x=c(x)
        x=self.lastact(x); x=self.global_pooling(x).view(x.size(0),-1)
        return self.classifier(x)


def build_energy_net_from_str(geno_str: str, N:int, num_classes:int, batch_size:int, cfg:Dict[str,int]):
    """Helper so search pipeline can keep *only* the string representation."""
    geno = str_to_genotype(geno_str)
    return EnergyNet(N, geno, num_classes, batch_size, cfg)

if __name__ == "__main__":
    from operations_energy import SearchSpaceNames
    import torch

    space = SearchSpaceNames["energy-efficient"]

    # -------------------- smoke‑test -----------------------------------
    # gstr = generate_random_genotype(space)
    # print("rand cell:", gstr)

    # g = str_to_genotype(gstr)
    # assert genotype_to_str(g) == gstr

    # cfg = {"COUT": 16, "KERNEL_SIZE": 3, "STRIDE": 1}
    # net = build_energy_net_from_str(gstr, N=5, num_classes=10, batch_size=1, cfg=cfg)
    # dummy = torch.randn(1, 3, 32, 32)
    # out = net(dummy)
    # print("forward OK, logits shape:", out.shape)

    # # -------------------- quick enumeration sample ---------------------
    # print("example enum:", next(enumerate_all_genotypes(space, nodes=4)))

    # -------------------- write genotype file --------------------------
    SAVE_PATH = "energy_genotype.txt"   # 输出文件名
    with open(SAVE_PATH, "w") as f:
        for g in enumerate_all_genotypes(space, nodes=4):
            f.write(g + "\n")

    print(f"saved genotypes → {SAVE_PATH}")


