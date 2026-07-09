from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueEncoder(nn.Module):
    """Encode one scalar expression value per gene into the latent gene field."""

    def __init__(self, dim: int, dropout: float = 0.0, max_value: float = 512.0):
        super().__init__()
        self.max_value = float(max_value)
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(0.0).clamp_max(self.max_value).unsqueeze(-1)
        return self.net(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(float(max_period))
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / max(half, 1)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


def manifold_shift_anchors(manifold_dim: int, shift_dims: int = 0, stencil: str = "axis") -> torch.Tensor:
    manifold_dim = int(manifold_dim)
    if manifold_dim <= 0:
        return torch.empty(0, 0)
    active_dims = int(shift_dims) if int(shift_dims) > 0 else manifold_dim
    active_dims = max(1, min(active_dims, manifold_dim))

    if stencil == "axis":
        dirs = []
        for dim in range(active_dims):
            for sign in (-1.0, 1.0):
                vec = torch.zeros(manifold_dim)
                vec[dim] = sign
                dirs.append(vec)
    elif stencil == "cube":
        dirs = []
        for combo in itertools.product((-1.0, 0.0, 1.0), repeat=active_dims):
            if all(v == 0.0 for v in combo):
                continue
            vec = torch.zeros(manifold_dim)
            vec[:active_dims] = torch.tensor(combo)
            dirs.append(vec)
    else:
        raise ValueError(f"Unsupported shift stencil: {stencil}")

    anchors = torch.stack(dirs, dim=0) if dirs else torch.empty(0, manifold_dim)
    return F.normalize(anchors.float(), dim=-1, eps=1e-6)


def manifold_shift_weights(
    neighbors: torch.Tensor,
    manifold_coords: torch.Tensor,
    anchors: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    if anchors.numel() == 0:
        return torch.empty(neighbors.size(0), neighbors.size(1), 0)
    coords = manifold_coords.float()
    edge_vec = coords[neighbors.long()] - coords[:, None, :]
    edge_vec = F.normalize(edge_vec, dim=-1, eps=1e-6)
    logits = torch.einsum("gkm,rm->gkr", edge_vec, anchors.float())
    return torch.softmax(float(temperature) * logits, dim=-1)


def fixed_shift_codes(anchors: torch.Tensor, dim: int) -> torch.Tensor:
    anchors = anchors.float()
    if anchors.numel() == 0:
        return torch.empty(0, int(dim))
    basis = torch.arange(1, int(dim) + 1, dtype=torch.float32, device=anchors.device)
    phase = anchors @ torch.arange(1, anchors.size(1) + 1, dtype=torch.float32, device=anchors.device)
    codes = torch.sin(phase[:, None] * basis[None] * 0.173) + torch.cos(phase[:, None] * basis[None] * 0.097)
    return F.normalize(codes, dim=-1, eps=1e-6)


def auto_grid_shape(n_items: int, dims: int = 3, side: int = 0) -> tuple[int, ...]:
    dims = max(int(dims), 1)
    if int(side) > 0:
        return tuple([int(side)] * dims)
    side_len = int(math.ceil(float(n_items) ** (1.0 / float(dims))))
    return tuple([max(side_len, 1)] * dims)


def assign_genes_to_grid(manifold_coords: torch.Tensor, grid_shape: tuple[int, ...]) -> torch.Tensor:
    n_genes = int(manifold_coords.size(0))
    grid_size = int(math.prod(grid_shape))
    if n_genes > grid_size:
        raise ValueError(f"Grid {grid_shape} has {grid_size} slots for {n_genes} genes.")
    coords = manifold_coords.float()
    if coords.numel() == 0 or coords.size(1) == 0:
        order = torch.arange(n_genes)
    else:
        score = torch.zeros(n_genes)
        for dim in range(coords.size(1)):
            score = score + coords[:, dim] * (10.0 ** -dim)
        order = torch.argsort(score)
    grid_idx = torch.empty(n_genes, dtype=torch.long)
    grid_idx[order] = torch.arange(n_genes, dtype=torch.long)
    return grid_idx


def shift_nd_nonwrap(x: torch.Tensor, offset) -> torch.Tensor:
    offset = [int(v) for v in offset]
    out = torch.zeros_like(x)
    prefix = x.dim() - len(offset)
    src_slices = [slice(None)] * x.dim()
    dst_slices = [slice(None)] * x.dim()
    for i, shift in enumerate(offset):
        dim = prefix + i
        size = x.size(dim)
        if shift > 0:
            src_slices[dim] = slice(0, size - shift)
            dst_slices[dim] = slice(shift, size)
        elif shift < 0:
            src_slices[dim] = slice(-shift, size)
            dst_slices[dim] = slice(0, size + shift)
    out[tuple(dst_slices)] = x[tuple(src_slices)]
    return out


class AdaLN(nn.Module):
    def __init__(self, dim: int, chunks: int):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(dim, chunks * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)


class VoidGeneBlock(nn.Module):
    """Attention-free gene block with signed neighbor transport and a global state."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        dropout: float = 0.0,
        residual_scale: float = 0.1,
        neighbor_gate: bool = True,
        self_weight: float = 1.0,
        neighbor_weight: float = 1.0,
        global_weight: float = 1.0,
        source_weight: float = 1.0,
        use_signed_neighbors: bool = True,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.neighbor_gate_enabled = bool(neighbor_gate)
        self.self_weight = float(self_weight)
        self.neighbor_weight = float(neighbor_weight)
        self.global_weight = float(global_weight)
        self.source_weight = float(source_weight)
        self.use_signed_neighbors = bool(use_signed_neighbors)

        self.gene_norm = nn.LayerNorm(dim)
        self.pos_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.ada = AdaLN(dim, chunks=6)

        self.self_contract = nn.Linear(dim, hidden, bias=False)
        self.pos_neighbor_contract = nn.Linear(dim, hidden, bias=False)
        self.global_contract = nn.Linear(dim, hidden, bias=False)
        self.source_contract = nn.Linear(dim, hidden, bias=False)
        self.expand = nn.Linear(hidden, dim, bias=False)

        contract_layers = [self.self_contract, self.pos_neighbor_contract,
                            self.global_contract, self.source_contract]

        if self.use_signed_neighbors:
            self.neg_norm = nn.LayerNorm(dim)
            self.neg_neighbor_contract = nn.Linear(dim, hidden, bias=False)
            contract_layers.append(self.neg_neighbor_contract)
        else:
            self.neg_norm = None
            self.neg_neighbor_contract = None

        gate_out = 2 * dim if self.use_signed_neighbors else dim
        if self.neighbor_gate_enabled:
            self.neighbor_gate = nn.Sequential(
                nn.Linear(2 * dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, gate_out),
                nn.Sigmoid(),
            )
        else:
            self.neighbor_gate = None

        self.global_update = nn.Sequential(
            nn.Linear(3 * dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

        for layer in contract_layers:
            nn.init.trunc_normal_(layer.weight, std=0.02)
        nn.init.trunc_normal_(self.expand.weight, std=0.01)

    def precompute(self, condition: torch.Tensor, source_state: torch.Tensor | None):
        ada = self.ada(condition).chunk(6, dim=-1)
        if source_state is not None and self.source_weight != 0.0:
            src_contrib = self.source_weight * self.source_contract(self.source_norm(source_state))
            source_pooled = source_state.mean(dim=1)
        else:
            src_contrib = None
            source_pooled = None
        return ada, src_contrib, source_pooled

    def forward(
        self,
        x: torch.Tensor,
        global_state: torch.Tensor,
        pos_msg: torch.Tensor,
        neg_msg: torch.Tensor | None,
        condition: torch.Tensor,
        source_state: torch.Tensor | None = None,
        cache: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache is None:
            ada, src_contrib, source_pooled = self.precompute(condition, source_state)
        else:
            ada, src_contrib, source_pooled = cache
        shift_x, scale_x, gate_x, shift_g, scale_g, gate_g = ada

        x_mod = self.gene_norm(x)
        x_mod = x_mod * (1.0 + scale_x[:, None, :]) + shift_x[:, None, :]
        g_mod = self.global_norm(global_state)
        g_mod = g_mod * (1.0 + scale_g) + shift_g

        pos_mod = self.pos_norm(pos_msg)
        if self.use_signed_neighbors:
            neg_mod = self.neg_norm(neg_msg)
        if self.neighbor_gate is not None:
            gate = self.neighbor_gate(torch.cat([g_mod, condition], dim=-1))
            if self.use_signed_neighbors:
                pos_gate, neg_gate = gate.chunk(2, dim=-1)
                neg_mod = neg_mod * neg_gate[:, None, :]
            else:
                pos_gate = gate
            pos_mod = pos_mod * pos_gate[:, None, :]

        mixed = self.self_weight * self.self_contract(x_mod)
        if self.use_signed_neighbors:
            mixed = mixed + self.neighbor_weight * (
                self.pos_neighbor_contract(pos_mod) + self.neg_neighbor_contract(neg_mod)
            )
        else:
            mixed = mixed + self.neighbor_weight * self.pos_neighbor_contract(pos_mod)
        mixed = mixed + self.global_weight * self.global_contract(g_mod[:, None, :])
        if src_contrib is not None:
            mixed = mixed + src_contrib

        dx = self.expand(self.drop(self.act(mixed)))
        x = x + self.residual_scale * torch.tanh(gate_x[:, None, :]) * dx

        pooled = x.mean(dim=1)
        if source_pooled is None:
            source_pooled = pooled
        dg = self.global_update(torch.cat([g_mod, pooled, source_pooled], dim=-1))
        global_state = global_state + self.residual_scale * torch.tanh(gate_g) * dg
        return x, global_state