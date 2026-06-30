import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueEncoder(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(-1))



class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = int(freq_dim)
        self.net = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.net(emb)




def manifold_shift_anchors(manifold_dim: int, shift_dims: int = 0, stencil: str = "axis") -> torch.Tensor:
    manifold_dim = int(manifold_dim)
    shift_dims = manifold_dim if shift_dims <= 0 else min(int(shift_dims), manifold_dim)
    if manifold_dim <= 0 or shift_dims <= 0:
        raise ValueError("--directional-shifts needs --manifold-dim >= 1")
    stencil = str(stencil).lower()
    if stencil == "axis":
        anchors = torch.zeros(2 * shift_dims, manifold_dim, dtype=torch.float32)
        for i in range(shift_dims):
            anchors[2 * i, i] = 1.0
            anchors[2 * i + 1, i] = -1.0
        return anchors
    if stencil != "cube":
        raise ValueError(f"Unsupported shift stencil: {stencil}")
    directions = []
    for linear in range(3**shift_dims):
        value = linear
        direction = []
        for _ in range(shift_dims):
            direction.append((value % 3) - 1)
            value //= 3
        if any(direction):
            directions.append(direction)
    anchors = torch.zeros(len(directions), manifold_dim, dtype=torch.float32)
    anchors[:, :shift_dims] = torch.tensor(directions, dtype=torch.float32)
    return anchors



def manifold_shift_weights(
    neighbors: torch.Tensor,
    manifold_coords: torch.Tensor,
    anchors: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    coords = manifold_coords.float()
    edge_delta = coords[neighbors.long()] - coords[:, None, :]
    edge_delta = F.normalize(edge_delta, dim=-1, eps=1e-6)
    anchors = F.normalize(anchors.float(), dim=-1, eps=1e-6)
    scores = torch.einsum("gkm,rm->gkr", edge_delta, anchors)
    return torch.softmax(scores * float(temperature), dim=-1).float()



def fixed_shift_codes(anchors: torch.Tensor, dim: int) -> torch.Tensor:
    anchors = F.normalize(anchors.float(), dim=-1, eps=1e-6)
    channels = torch.arange(int(dim), dtype=torch.long)
    patterns = []
    for axis in range(anchors.size(1)):
        period = 2 ** axis
        pattern = torch.where((channels // period) % 2 == 0, 1.0, -1.0)
        patterns.append(pattern)
    pattern_matrix = torch.stack(patterns, dim=0).float()
    directional = anchors @ pattern_matrix
    return directional / directional.abs().amax(dim=1, keepdim=True).clamp_min(1.0)



def auto_grid_shape(n_items: int, grid_dims: int, grid_side: int = 0) -> tuple[int, ...]:
    grid_dims = int(grid_dims)
    if grid_dims <= 0:
        raise ValueError("--spatial-grid-shifts needs --spatial-grid-dims >= 1")
    side = int(grid_side)
    if side <= 0:
        side = max(2, int(math.ceil(float(n_items) ** (1.0 / float(grid_dims)))))
        while side**grid_dims < n_items:
            side += 1
    if side**grid_dims < n_items:
        raise ValueError(f"Grid side {side} with {grid_dims} dims cannot hold {n_items} genes")
    return tuple([side] * grid_dims)



def assign_genes_to_grid(manifold_coords: torch.Tensor, grid_shape: tuple[int, ...]) -> torch.Tensor:
    coords = manifold_coords.detach().cpu().numpy().astype(np.float32, copy=False)
    dims = len(grid_shape)
    if coords.shape[1] < dims:
        coords = np.pad(coords, ((0, 0), (0, dims - coords.shape[1])), mode="constant")
    coords = coords[:, :dims]
    lo = coords.min(axis=0, keepdims=True)
    hi = coords.max(axis=0, keepdims=True)
    coords = (coords - lo) / np.maximum(hi - lo, 1e-6)
    coords = coords * (np.asarray(grid_shape, dtype=np.float32)[None] - 1.0)

    grid_points = np.stack(
        np.meshgrid(*[np.arange(s, dtype=np.float32) for s in grid_shape], indexing="ij"),
        axis=-1,
    ).reshape(-1, dims)

    try:
        from scipy.optimize import linear_sum_assignment

        cost = ((coords[:, None, :] - grid_points[None, :, :]) ** 2).sum(axis=-1)
        rows, cols = linear_sum_assignment(cost)
        gene_to_grid = np.empty(coords.shape[0], dtype=np.int64)
        gene_to_grid[rows] = cols
        return torch.tensor(gene_to_grid, dtype=torch.long)
    except Exception:
        used: set[int] = set()
        gene_to_grid = np.empty(coords.shape[0], dtype=np.int64)
        for gene_idx, point in enumerate(coords):
            order = np.argsort(((grid_points - point[None]) ** 2).sum(axis=1))
            for flat_idx in order:
                flat_idx = int(flat_idx)
                if flat_idx not in used:
                    used.add(flat_idx)
                    gene_to_grid[gene_idx] = flat_idx
                    break
        return torch.tensor(gene_to_grid, dtype=torch.long)



def shift_nd_nonwrap(x: torch.Tensor, offsets: torch.Tensor | tuple[int, ...]) -> torch.Tensor:
    offsets = [int(v) for v in offsets]
    if not any(offsets):
        return x
    out = torch.zeros_like(x)
    src = [slice(None), slice(None)]
    dst = [slice(None), slice(None)]
    for axis, delta in enumerate(offsets):
        size = x.shape[2 + axis]
        src_start = max(0, -delta)
        src_end = size - max(0, delta)
        dst_start = max(0, delta)
        dst_end = size - max(0, -delta)
        src.append(slice(src_start, src_end))
        dst.append(slice(dst_start, dst_end))
    out[tuple(dst)] = x[tuple(src)]
    return out



class VoidGeneBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden: int,
        dropout: float,
        residual_scale: float,
        neighbor_gate: bool = False,
        self_weight: float = 1.0,
        neighbor_weight: float = 1.0,
        global_weight: float = 1.0,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.use_neighbor_gate = bool(neighbor_gate)
        self.self_weight = float(self_weight)
        self.neighbor_weight = float(neighbor_weight)
        self.global_weight = float(global_weight)
        self.gene_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.self_contract = nn.Linear(dim, hidden, bias=False)
        self.pos_neighbor_contract = nn.Linear(dim, hidden, bias=False)
        self.neg_neighbor_contract = nn.Linear(dim, hidden, bias=False)
        self.global_contract = nn.Linear(dim, hidden, bias=False)
        if self.use_neighbor_gate:
            self.neighbor_gate = nn.Sequential(
                nn.Linear(2 * dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2 * hidden),
                nn.Sigmoid(),
            )
        else:
            self.neighbor_gate = None
        self.expand = nn.Linear(hidden, dim, bias=False)
        self.global_update = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, global_state, pos_neighbor_mean, neg_neighbor_mean, condition):
        sx, ax, gx, sg, ag, gg = self.ada(condition).chunk(6, dim=-1)
        x_mod = self.gene_norm(x) * (1.0 + ax[:, None, :]) + sx[:, None, :]
        g_mod = self.global_norm(global_state) * (1.0 + ag) + sg
        mixed = self.self_weight * self.self_contract(x_mod)
        pos_msg = self.pos_neighbor_contract(pos_neighbor_mean)
        neg_msg = self.neg_neighbor_contract(neg_neighbor_mean)
        if self.neighbor_gate is not None:
            pos_gate, neg_gate = self.neighbor_gate(torch.cat([g_mod, condition], dim=-1)).chunk(2, dim=-1)
            pos_msg = pos_gate[:, None, :] * pos_msg
            neg_msg = neg_gate[:, None, :] * neg_msg
        mixed = mixed + self.neighbor_weight * pos_msg
        mixed = mixed + self.neighbor_weight * neg_msg
        mixed = mixed + self.global_weight * self.global_contract(g_mod[:, None, :])
        dx = self.expand(self.drop(F.gelu(mixed)))
        x = x + self.residual_scale * torch.tanh(gx[:, None, :]) * dx
        pooled = x.mean(dim=1)
        dg = self.global_update(torch.cat([g_mod, pooled], dim=-1))
        global_state = global_state + self.residual_scale * torch.tanh(gg) * dg
        return x, global_state


