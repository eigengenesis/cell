from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from pathlib import Path

from blocks import ValueEncoder, TimestepEmbedder, VoidGeneBlock, manifold_shift_anchors, manifold_shift_weights, fixed_shift_codes, auto_grid_shape, assign_genes_to_grid, shift_nd_nonwrap

def spectral_gene_coordinates(neighbors: np.ndarray, weights: np.ndarray, manifold_dim: int) -> np.ndarray:
    n_genes = int(neighbors.shape[0])
    manifold_dim = int(manifold_dim)
    if manifold_dim <= 0:
        return np.zeros((n_genes, 0), dtype=np.float32)

    try:
        from scipy import sparse
        from scipy.sparse import linalg as sparse_linalg

        rows = np.repeat(np.arange(n_genes), neighbors.shape[1])
        cols = neighbors.reshape(-1)
        vals = np.abs(weights.reshape(-1)).astype(np.float32)
        graph = sparse.coo_matrix((vals, (rows, cols)), shape=(n_genes, n_genes)).tocsr()
        graph = graph.maximum(graph.T)
        degree = np.asarray(graph.sum(axis=1)).reshape(-1)
        degree[degree <= 1e-8] = 1.0
        inv_sqrt = sparse.diags(1.0 / np.sqrt(degree))
        laplacian = sparse.eye(n_genes, format="csr") - inv_sqrt @ graph @ inv_sqrt
        k = min(manifold_dim + 1, n_genes - 1)
        eigvals, eigvecs = sparse_linalg.eigsh(laplacian, k=k, which="SM", tol=1e-3)
        order = np.argsort(eigvals)
        coords = eigvecs[:, order[1 : manifold_dim + 1]]
    except Exception:
        signed_graph = np.zeros((n_genes, n_genes), dtype=np.float32)
        rows = np.repeat(np.arange(n_genes), neighbors.shape[1])
        signed_graph[rows, neighbors.reshape(-1)] = weights.reshape(-1).astype(np.float32)
        signed_graph = 0.5 * (signed_graph + signed_graph.T)
        try:
            u, s, _ = np.linalg.svd(signed_graph, full_matrices=False)
            coords = u[:, :manifold_dim] * np.sqrt(s[:manifold_dim])[None]
        except Exception:
            coords = np.random.default_rng(0).normal(size=(n_genes, manifold_dim))

    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[1] < manifold_dim:
        pad = np.zeros((n_genes, manifold_dim - coords.shape[1]), dtype=np.float32)
        coords = np.concatenate([coords, pad], axis=1)
    coords = coords[:, :manifold_dim]
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = coords.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (coords / scale).astype(np.float32)



def build_coexpression_geometry(
    x_all: np.ndarray,
    k: int,
    manifold_dim: int,
    cache_path: Path | None = None,
    row_idx: np.ndarray | None = None,
    max_cells: int = 8192,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if isinstance(cached, dict) and {"neighbors", "weights", "coords"}.issubset(cached):
            return cached
    if row_idx is None:
        row_idx = np.arange(x_all.shape[0])
    if max_cells > 0 and len(row_idx) > max_cells:
        rng = np.random.default_rng(seed)
        row_idx = rng.choice(row_idx, size=max_cells, replace=False)
    x = x_all[row_idx].astype(np.float32, copy=False)
    x = np.nan_to_num(x)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x = x / std
    corr = (x.T @ x) / max(x.shape[0] - 1, 1)
    np.fill_diagonal(corr, 0.0)
    kth = min(k, corr.shape[1] - 2)
    neighbors = np.argpartition(-np.abs(corr), kth=kth, axis=1)[:, :k]
    edge_weights = corr[np.arange(corr.shape[0])[:, None], neighbors]
    order = np.argsort(-np.abs(edge_weights), axis=1)
    neighbors = np.take_along_axis(neighbors, order, axis=1)
    edge_weights = np.take_along_axis(edge_weights, order, axis=1).astype(np.float32)
    coords = spectral_gene_coordinates(neighbors, edge_weights, manifold_dim)
    graph = {
        "neighbors": torch.tensor(neighbors.astype(np.int64), dtype=torch.long),
        "weights": torch.tensor(edge_weights, dtype=torch.float32),
        "coords": torch.tensor(coords, dtype=torch.float32),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, cache_path)
    return graph



class VoidCellModel(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_perturbations: int,
        neighbors: torch.Tensor,
        edge_weights: torch.Tensor,
        manifold_coords: torch.Tensor,
        dim: int = 192,
        hidden: int = 512,
        encode_blocks: int = 4,
        think_steps: int = 8,
        dropout: float = 0.05,
        residual_scale: float = 0.1,
        neighbor_chunk: int = 256,
        neighbor_gate: bool = False,
        directional_shifts: bool = False,
        directional_residual_gate: bool = False,
        directional_gate_init: float = -4.0,
        shift_dims: int = 0,
        shift_stencil: str = "axis",
        shift_temperature: float = 4.0,
        shift_code_strength: float = 1.0,
        spatial_grid_shifts: bool = False,
        spatial_grid_dims: int = 3,
        spatial_grid_side: int = 0,
        spatial_shift_stencil: str = "cube",
        spatial_shift_code_strength: float = 1.0,
        graph_message_weight: float = 1.0,
        spatial_message_weight: float = 1.0,
        self_weight: float = 1.0,
        neighbor_weight: float = 1.0,
        global_weight: float = 1.0,
        source_memory: str = "static",
        source_weight: float = 1.0,
        action_field: str = "target_gene",
        action_strength: float = 1.0,
        action_init_std: float = 1e-4,
        share_pert_gene_embedding: bool = False,
        dynamic_edge_gate: bool = False,
        edge_gate_init: float = 2.0,
        checkpoint_blocks: bool = False,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.dim = int(dim)
        self.think_steps = int(think_steps)
        self.neighbor_chunk = int(neighbor_chunk)
        self.directional_shifts = bool(directional_shifts)
        self.directional_residual_gate = bool(directional_residual_gate)
        self.shift_code_strength = float(shift_code_strength)
        self.spatial_grid_shifts = bool(spatial_grid_shifts)
        self.spatial_shift_code_strength = float(spatial_shift_code_strength)
        self.graph_message_weight = float(graph_message_weight)
        self.spatial_message_weight = float(spatial_message_weight)
        if source_memory not in {"none", "static"}:
            raise ValueError(f"Unsupported source memory: {source_memory}")
        if action_field not in {"none", "target_gene", "drug_manifold"}:
            raise ValueError(f"Unsupported action field: {action_field}")
        self.source_memory = source_memory
        self.source_weight = float(source_weight)
        self.action_field = action_field
        self.action_strength = float(action_strength)
        self.action_init_std = float(action_init_std)
        self.share_pert_gene_embedding = bool(share_pert_gene_embedding)
        self.dynamic_edge_gate = bool(dynamic_edge_gate)
        self.edge_gate_init = float(edge_gate_init)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.register_buffer("neighbors", neighbors.long(), persistent=True)
        self.register_buffer("edge_weights", edge_weights.float(), persistent=True)
        self.register_buffer("manifold_coords", manifold_coords.float(), persistent=True)
        self.manifold_dim = int(manifold_coords.shape[1])
        if self.action_field == "drug_manifold" and self.manifold_dim <= 0:
            raise ValueError("--action-field drug_manifold needs --manifold-dim >= 1")
        if self.directional_shifts:
            anchors = manifold_shift_anchors(manifold_coords.shape[1], shift_dims, shift_stencil)
            shift_weights = manifold_shift_weights(neighbors, manifold_coords, anchors, shift_temperature)
            shift_codes = fixed_shift_codes(anchors, dim)
            self.register_buffer("shift_anchors", anchors, persistent=True)
            self.register_buffer("shift_weights", shift_weights, persistent=True)
            self.register_buffer("shift_codes", shift_codes, persistent=True)
            self.n_shift_dirs = int(anchors.size(0))
        else:
            self.register_buffer("shift_anchors", torch.empty(0, manifold_coords.shape[1]), persistent=True)
            self.register_buffer("shift_weights", torch.empty(0), persistent=True)
            self.register_buffer("shift_codes", torch.empty(0, dim), persistent=True)
            self.n_shift_dirs = 0
        if self.directional_residual_gate:
            self.directional_gate_logit = nn.Parameter(torch.tensor(float(directional_gate_init)))
        else:
            self.register_buffer("directional_gate_logit", torch.tensor(float(directional_gate_init)), persistent=True)
        if self.spatial_grid_shifts:
            self.grid_shape = auto_grid_shape(n_genes, spatial_grid_dims, spatial_grid_side)
            grid_gene_index = assign_genes_to_grid(manifold_coords, self.grid_shape)
            spatial_offsets = manifold_shift_anchors(len(self.grid_shape), len(self.grid_shape), spatial_shift_stencil).long()
            spatial_codes = fixed_shift_codes(spatial_offsets.float(), dim)
            self.register_buffer("grid_gene_index", grid_gene_index, persistent=True)
            self.register_buffer("spatial_offsets", spatial_offsets, persistent=True)
            self.register_buffer("spatial_codes", spatial_codes, persistent=True)
            self.grid_size = int(np.prod(self.grid_shape))
            self.n_spatial_dirs = int(spatial_offsets.size(0))
        else:
            self.grid_shape = ()
            self.grid_size = 0
            self.n_spatial_dirs = 0
            self.register_buffer("grid_gene_index", torch.empty(0, dtype=torch.long), persistent=True)
            self.register_buffer("spatial_offsets", torch.empty(0, 0, dtype=torch.long), persistent=True)
            self.register_buffer("spatial_codes", torch.empty(0, dim), persistent=True)
        self.gene_embedding = nn.Embedding(n_genes, dim)
        self.coord_embedding = nn.Sequential(
            nn.Linear(max(1, manifold_coords.shape[1]), dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.perturbation_embedding = nn.Embedding(n_perturbations, dim)
        self.pert_gene_embedding = nn.Embedding(n_genes + 1, dim)
        self.value_current = ValueEncoder(dim, dropout)
        self.value_control = ValueEncoder(dim, dropout)
        self.time_embedding = TimestepEmbedder(dim)
        self.input_fusion = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.source_fusion = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.condition_fusion = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_vector = nn.Linear(dim, dim)
        self.action_projection = nn.Linear(dim, dim, bias=False)
        nn.init.trunc_normal_(self.action_projection.weight, std=self.action_init_std)
        if self.manifold_dim > 0:
            self.drug_to_manifold = nn.Linear(dim, self.manifold_dim)
        else:
            self.drug_to_manifold = None
        self.action_log_temperature = nn.Parameter(torch.tensor(math.log(1.0)))
        self.action_log_radius = nn.Parameter(torch.tensor(math.log(1.0)))
        if self.dynamic_edge_gate:
            self.edge_gate = nn.Sequential(
                nn.Linear(4 * dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2),
            )
            nn.init.zeros_(self.edge_gate[-1].weight)
            nn.init.constant_(self.edge_gate[-1].bias, self.edge_gate_init)
        else:
            self.edge_gate = None
        self.encode = nn.ModuleList(
            [
                VoidGeneBlock(
                    dim,
                    hidden,
                    dropout,
                    residual_scale,
                    neighbor_gate=neighbor_gate,
                    self_weight=self_weight,
                    neighbor_weight=neighbor_weight,
                    global_weight=global_weight,
                    source_weight=source_weight,
                )
                for _ in range(encode_blocks)
            ]
        )
        self.ghost = VoidGeneBlock(
            dim,
            hidden,
            dropout,
            residual_scale,
            neighbor_gate=neighbor_gate,
            self_weight=self_weight,
            neighbor_weight=neighbor_weight,
            global_weight=global_weight,
            source_weight=source_weight,
        )
        self.ghost_gate_logit = nn.Parameter(torch.tensor(-2.1972))
        self.out_norm = nn.LayerNorm(dim)
        self.velocity = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def neighbor_messages(self, x):
        b, g, d = x.shape
        k = self.neighbors.size(1)
        chunk = self.neighbor_chunk if self.neighbor_chunk > 0 else g
        eps = 1e-6
        if chunk >= g:
            gathered = x.index_select(1, self.neighbors.reshape(-1)).reshape(b, g, k, d)
            weights = self.edge_weights.to(device=x.device, dtype=x.dtype)
            pos = weights.clamp_min(0.0)
            neg = (-weights).clamp_min(0.0)
            pos_msg = (gathered * pos[None, :, :, None]).sum(dim=2) / pos.sum(dim=1).clamp_min(eps)[None, :, None]
            neg_msg = (gathered * neg[None, :, :, None]).sum(dim=2) / neg.sum(dim=1).clamp_min(eps)[None, :, None]
            return pos_msg, neg_msg

        pos_out = torch.empty_like(x)
        neg_out = torch.empty_like(x)
        weights = self.edge_weights.to(device=x.device, dtype=x.dtype)
        for start in range(0, g, chunk):
            end = min(start + chunk, g)
            idx = self.neighbors[start:end].reshape(-1)
            gathered = x.index_select(1, idx).reshape(b, end - start, k, d)
            w = weights[start:end]
            pos = w.clamp_min(0.0)
            neg = (-w).clamp_min(0.0)
            pos_out[:, start:end, :] = (gathered * pos[None, :, :, None]).sum(dim=2) / pos.sum(dim=1).clamp_min(eps)[None, :, None]
            neg_out[:, start:end, :] = (gathered * neg[None, :, :, None]).sum(dim=2) / neg.sum(dim=1).clamp_min(eps)[None, :, None]
        return pos_out, neg_out

    def directional_neighbor_messages(self, x):
        b, g, d = x.shape
        k = self.neighbors.size(1)
        chunk = self.neighbor_chunk if self.neighbor_chunk > 0 else g
        eps = 1e-6
        pos_out = torch.zeros_like(x)
        neg_out = torch.zeros_like(pos_out)
        weights = self.edge_weights.to(device=x.device, dtype=x.dtype)
        shift_weights = self.shift_weights.to(device=x.device, dtype=x.dtype)
        shift_codes = self.shift_codes.to(device=x.device, dtype=x.dtype)
        for start in range(0, g, chunk):
            end = min(start + chunk, g)
            idx = self.neighbors[start:end].reshape(-1)
            gathered = x.index_select(1, idx).reshape(b, end - start, k, d)
            edge = weights[start:end]
            pos_base = edge.clamp_min(0.0)
            neg_base = (-edge).clamp_min(0.0)
            pos_den = pos_base.sum(dim=1).clamp_min(eps)[None, :, None]
            neg_den = neg_base.sum(dim=1).clamp_min(eps)[None, :, None]
            base_pos = (gathered * pos_base[None, :, :, None]).sum(dim=2) / pos_den
            base_neg = (gathered * neg_base[None, :, :, None]).sum(dim=2) / neg_den
            edge_codes = torch.einsum("gkr,rd->gkd", shift_weights[start:end], shift_codes)
            directional = gathered * edge_codes[None]
            pos_delta = (directional * pos_base[None, :, :, None]).sum(dim=2) / pos_den
            neg_delta = (directional * neg_base[None, :, :, None]).sum(dim=2) / neg_den
            pos_out[:, start:end, :] = base_pos + self.shift_code_strength * pos_delta
            neg_out[:, start:end, :] = base_neg + self.shift_code_strength * neg_delta
        return pos_out, neg_out

    def directional_residual_neighbor_messages(self, x):
        b, g, d = x.shape
        k = self.neighbors.size(1)
        chunk = self.neighbor_chunk if self.neighbor_chunk > 0 else g
        eps = 1e-6
        pos_out = torch.empty_like(x)
        neg_out = torch.empty_like(x)
        weights = self.edge_weights.to(device=x.device, dtype=x.dtype)
        shift_weights = self.shift_weights.to(device=x.device, dtype=x.dtype)
        shift_codes = self.shift_codes.to(device=x.device, dtype=x.dtype)
        gate = torch.sigmoid(self.directional_gate_logit).to(device=x.device, dtype=x.dtype)
        for start in range(0, g, chunk):
            end = min(start + chunk, g)
            idx = self.neighbors[start:end].reshape(-1)
            gathered = x.index_select(1, idx).reshape(b, end - start, k, d)
            edge = weights[start:end]
            pos = edge.clamp_min(0.0)
            neg = (-edge).clamp_min(0.0)
            pos_den = pos.sum(dim=1).clamp_min(eps)[None, :, None]
            neg_den = neg.sum(dim=1).clamp_min(eps)[None, :, None]

            base_pos = (gathered * pos[None, :, :, None]).sum(dim=2) / pos_den
            base_neg = (gathered * neg[None, :, :, None]).sum(dim=2) / neg_den

            edge_codes = torch.einsum("gkr,rd->gkd", shift_weights[start:end], shift_codes)
            directional = gathered * edge_codes[None]
            pos_delta = (directional * pos[None, :, :, None]).sum(dim=2) / pos_den
            neg_delta = (directional * neg[None, :, :, None]).sum(dim=2) / neg_den

            scaled_gate = gate * self.shift_code_strength
            pos_out[:, start:end, :] = base_pos + scaled_gate * pos_delta
            neg_out[:, start:end, :] = base_neg + scaled_gate * neg_delta
        return pos_out, neg_out

    def spatial_grid_message(self, x):
        b, g, d = x.shape
        grid_idx = self.grid_gene_index.to(device=x.device)
        flat = x.new_zeros((b, d, self.grid_size))
        flat.index_copy_(2, grid_idx, x.transpose(1, 2))
        grid = flat.reshape(b, d, *self.grid_shape)
        acc = grid
        offsets = self.spatial_offsets.to(device=x.device)
        codes = self.spatial_codes.to(device=x.device, dtype=x.dtype)
        view_shape = (1, d) + (1,) * len(self.grid_shape)
        for offset, code in zip(offsets, codes):
            shifted = shift_nd_nonwrap(grid, offset)
            acc = acc + shifted + self.spatial_shift_code_strength * shifted * code.reshape(view_shape)
        acc = acc / float(self.n_spatial_dirs + 1)
        return acc.reshape(b, d, self.grid_size).index_select(2, grid_idx).transpose(1, 2)

    def shared_gene_perturbation_embedding(self, perturbation_id, perturbation_gene_id):
        fallback = self.perturbation_embedding(perturbation_id.clamp_min(0))
        gene_ids = perturbation_gene_id.long()
        valid = (gene_ids >= 0) & (gene_ids < self.n_genes)
        safe_gene_ids = gene_ids.clamp(0, max(self.n_genes - 1, 0))
        shared = self.gene_embedding(safe_gene_ids)
        tokens = torch.where(valid[..., None], shared, fallback)
        return tokens.mean(dim=1)

    def condition(self, t, perturbation_id, perturbation_gene_id):
        time = self.time_embedding(t)
        if self.share_pert_gene_embedding and self.action_field == "target_gene":
            pert = self.shared_gene_perturbation_embedding(perturbation_id, perturbation_gene_id)
            pert_gene = pert
        else:
            pert = self.perturbation_embedding(perturbation_id.clamp_min(0)).mean(dim=1)
            gene_ids = perturbation_gene_id.clamp_min(-1) + 1
            pert_gene = self.pert_gene_embedding(gene_ids).mean(dim=1)
        return self.condition_fusion(torch.cat([time, pert, pert_gene], dim=-1)), pert

    def build_source_state(self, gene, coord, control):
        if self.source_memory == "none" or self.source_weight == 0.0:
            return None
        return self.source_fusion(torch.cat([gene, coord, self.value_control(control)], dim=-1))

    def target_gene_action_gate(self, perturbation_gene_id, b: int, g: int, device, dtype):
        gate = torch.zeros((b, g), device=device, dtype=dtype)
        gene_ids = perturbation_gene_id.long()
        valid = (gene_ids >= 0) & (gene_ids < g)
        if valid.any():
            scatter_ids = gene_ids.clamp(0, max(g - 1, 0))
            gate.scatter_add_(1, scatter_ids, valid.to(dtype))
            gate.clamp_(0.0, 1.0)
        return gate

    def drug_manifold_action_gate(self, pert, device, dtype):
        if self.drug_to_manifold is None or self.manifold_dim <= 0:
            raise ValueError("--action-field drug_manifold needs --manifold-dim >= 1")
        coords = self.manifold_coords.to(device=device, dtype=dtype)
        drug_coord = self.drug_to_manifold(pert).to(dtype=dtype)
        dist_sq = (drug_coord[:, None, :] - coords[None, :, :]).square().sum(dim=-1)
        temperature = F.softplus(self.action_log_temperature).to(device=device, dtype=dtype).clamp_min(1e-4)
        radius = F.softplus(self.action_log_radius).to(device=device, dtype=dtype).clamp_min(1e-4)
        return torch.sigmoid(temperature * (radius - dist_sq))

    def build_action_field(self, pert, perturbation_gene_id, b: int, g: int, device, dtype):
        if self.action_field == "none" or self.action_strength == 0.0:
            empty = torch.zeros((b, g, self.dim), device=device, dtype=dtype)
            return empty, empty.mean(dim=1), None

        action_vec = self.action_vector(pert).to(dtype=dtype)
        if self.action_field == "target_gene":
            gate = self.target_gene_action_gate(perturbation_gene_id, b, g, device, dtype)
        elif self.action_field == "drug_manifold":
            gate = self.drug_manifold_action_gate(pert, device, dtype)
        else:
            raise ValueError(f"Unsupported action field: {self.action_field}")

        action = gate[:, :, None] * action_vec[:, None, :]
        return action, action.mean(dim=1), gate

    def action_auxiliary_loss(self, perturbation_id, perturbation_gene_id):
        if self.action_field != "drug_manifold":
            param = self.action_projection.weight
            return param.new_tensor(0.0)
        pert = self.perturbation_embedding(perturbation_id.clamp_min(0)).mean(dim=1)
        gate = self.drug_manifold_action_gate(pert, pert.device, pert.dtype)
        return gate.square().mean()

    def apply_dynamic_edge_gate(self, x, global_state, cond, pos_msg, neg_msg, action_summary):
        if self.edge_gate is None:
            return pos_msg, neg_msg
        b, g, _ = x.shape
        if action_summary is None:
            action_summary = torch.zeros_like(global_state)
        global_field = global_state[:, None, :].expand(b, g, -1)
        cond_field = cond[:, None, :].expand(b, g, -1)
        action_field = action_summary[:, None, :].expand(b, g, -1)
        gate_logits = self.edge_gate(torch.cat([x, global_field, cond_field, action_field], dim=-1))
        baseline = torch.sigmoid(
            torch.tensor(self.edge_gate_init, device=x.device, dtype=x.dtype)
        ).clamp_min(1e-4)
        gates = torch.sigmoid(gate_logits).to(dtype=x.dtype) / baseline
        return pos_msg * gates[..., 0:1], neg_msg * gates[..., 1:2]

    def run_block(self, block, x, global_state, cond, source_state=None, action_summary=None):
        if self.spatial_grid_shifts:
            spatial_msg = self.spatial_grid_message(x)
            if self.graph_message_weight == 0.0:
                pos_msg = self.spatial_message_weight * spatial_msg
                neg_msg = torch.zeros_like(spatial_msg)
            else:
                if self.directional_shifts and self.directional_residual_gate:
                    pos_msg, neg_msg = self.directional_residual_neighbor_messages(x)
                elif self.directional_shifts:
                    pos_msg, neg_msg = self.directional_neighbor_messages(x)
                else:
                    pos_msg, neg_msg = self.neighbor_messages(x)
                pos_msg = self.graph_message_weight * pos_msg + self.spatial_message_weight * spatial_msg
                neg_msg = self.graph_message_weight * neg_msg
            pos_msg, neg_msg = self.apply_dynamic_edge_gate(
                x, global_state, cond, pos_msg, neg_msg, action_summary
            )
            return block(x, global_state, pos_msg, neg_msg, cond, source_state)
        if self.directional_shifts and self.directional_residual_gate:
            pos_msg, neg_msg = self.directional_residual_neighbor_messages(x)
        elif self.directional_shifts:
            pos_msg, neg_msg = self.directional_neighbor_messages(x)
        else:
            pos_msg, neg_msg = self.neighbor_messages(x)
        pos_msg, neg_msg = self.apply_dynamic_edge_gate(
            x, global_state, cond, pos_msg, neg_msg, action_summary
        )
        return block(x, global_state, pos_msg, neg_msg, cond, source_state)

    def maybe_checkpoint_block(self, block, x, global_state, cond, source_state=None, action_summary=None):
        if self.training and self.checkpoint_blocks:
            return checkpoint(
                lambda a, b, c, d, e: self.run_block(block, a, b, c, d, e),
                x,
                global_state,
                cond,
                source_state,
                action_summary,
                use_reentrant=False,
            )
        return self.run_block(block, x, global_state, cond, source_state, action_summary)

    def forward(self, x_t, control, t, perturbation_id, perturbation_gene_id):
        b, g = x_t.shape
        gene_ids = torch.arange(g, device=x_t.device)
        gene = self.gene_embedding(gene_ids)[None].expand(b, -1, -1)
        coords = self.manifold_coords.to(device=x_t.device, dtype=x_t.dtype)
        if coords.size(1) == 0:
            coords = torch.zeros(g, 1, device=x_t.device, dtype=x_t.dtype)
        coord = self.coord_embedding(coords)[None].expand(b, -1, -1)
        cond, pert = self.condition(t, perturbation_id, perturbation_gene_id)
        source_state = self.build_source_state(gene, coord, control)
        x = self.input_fusion(torch.cat([gene, coord, self.value_current(x_t), self.value_control(control)], dim=-1))
        action, action_summary, _ = self.build_action_field(
            pert,
            perturbation_gene_id,
            b,
            g,
            x_t.device,
            x.dtype,
        )
        if self.action_field != "none" and self.action_strength != 0.0:
            x = x + self.action_strength * self.action_projection(action)
        global_state = x.mean(dim=1)

        for block in self.encode:
            x, global_state = self.maybe_checkpoint_block(
                block, x, global_state, cond, source_state, action_summary
            )

        x_init = x
        global_init = global_state
        gate = torch.sigmoid(self.ghost_gate_logit)
        for _ in range(self.think_steps):
            ghost_x = x + x_init
            cand_x, cand_global = self.maybe_checkpoint_block(
                self.ghost,
                ghost_x,
                global_state + global_init,
                cond,
                source_state,
                action_summary,
            )
            x = x + gate * (cand_x - x)
            global_state = global_state + gate * (cand_global - global_state)

        x = self.out_norm(x)
        pert_field = pert[:, None, :].expand(-1, g, -1)
        return self.velocity(torch.cat([x, pert_field], dim=-1)).squeeze(-1)
