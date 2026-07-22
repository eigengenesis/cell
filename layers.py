import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict


class BatchLabelEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.enc_norm(self.embedding(x))


class ExprDecoder(nn.Module):
    def __init__(self, d_model: int, explicit_zero_prob: bool = False, use_batch_labels: bool = False):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model), nn.LeakyReLU(),
            nn.Linear(d_model, d_model), nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model), nn.LeakyReLU(),
                nn.Linear(d_model, d_model), nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        pred_value = self.fc(x).squeeze(-1)
        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_probs = torch.sigmoid(self.zero_logit(x).squeeze(-1))
        return dict(pred=pred_value, zero_probs=zero_probs)


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        use_perturbation_interaction: bool = False,
        mask_path: str = None,
        use_wire: bool = True,
        wire_path: str = None,
        grn_mask_path: str = None,
        neighbor_cap: int = 128,
        corr_path: str = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)
        self.use_perturbation_interaction = use_perturbation_interaction
        self.use_wire = use_wire and wire_path is not None
        self.neighbor_cap = neighbor_cap
        self.manifold_dim = 0

        if use_perturbation_interaction:
            adj = ~torch.load(mask_path).bool()
            adj.fill_diagonal_(False)
            neighbors = self._adjacency_to_neighbor_table(adj, neighbor_cap)

            if grn_mask_path is not None and grn_mask_path != '':
                grn_raw = torch.load(grn_mask_path)
                if grn_raw.dtype == torch.bool and grn_raw.dim() == 2 and grn_raw.shape[0] == grn_raw.shape[1]:
                    g = grn_raw.clone()
                    g.fill_diagonal_(False)
                    grn_neighbors = self._adjacency_to_neighbor_table(g, neighbor_cap)
                else:
                    grn_neighbors = grn_raw.long()
                neighbors = self._merge_neighbor_tables(neighbors, grn_neighbors, neighbor_cap)

            true_degree = int((neighbors >= 0).sum(dim=1).max().item())
            neighbors = neighbors[:, :max(true_degree, 1)]
            self.neighbor_cap = neighbors.shape[1]

            valid = neighbors >= 0
            if corr_path is not None and corr_path != '':
                corr = torch.load(corr_path).float()
                edge_corr = corr.gather(1, neighbors.clamp_min(0)) * valid.float()
                edge_weight_pos = edge_corr.clamp_min(0.0)
                edge_weight_neg = (-edge_corr).clamp_min(0.0)
            else:
                edge_weight_pos = valid.float()
                edge_weight_neg = torch.zeros_like(edge_weight_pos)

            self.register_buffer('neighbors', neighbors, persistent=True)
            self.register_buffer('edge_weights_pos', edge_weight_pos, persistent=True)
            self.register_buffer('edge_weights_neg', edge_weight_neg, persistent=True)

        if self.use_wire:
            eigvecs = torch.load(wire_path, weights_only=True).float()
            self.manifold_dim = eigvecs.shape[1]
            self.register_buffer('manifold_coords', eigvecs, persistent=True)
            self.coord_embedding = nn.Sequential(
                nn.Linear(self.manifold_dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
            )

    @staticmethod
    def _adjacency_to_neighbor_table(adj, k_max):
        adj = adj.bool()
        V = adj.shape[0]
        table = torch.full((V, k_max), -1, dtype=torch.long)
        for i in range(V):
            idx = torch.nonzero(adj[i], as_tuple=False).reshape(-1)
            if idx.numel() > k_max:
                idx = idx[:k_max]
            if idx.numel() > 0:
                table[i, :idx.numel()] = idx
        return table

    @staticmethod
    def _merge_neighbor_tables(a, b, k_max):
        V = a.shape[0]
        merged = torch.full((V, k_max), -1, dtype=torch.long)
        for i in range(V):
            ids = torch.cat([a[i][a[i] >= 0], b[i][b[i] >= 0]])
            ids = torch.unique(ids)[:k_max]
            if ids.numel() > 0:
                merged[i, :ids.numel()] = ids
        return merged

    def local_graph(self, gene_id_row: Tensor):
        device = gene_id_row.device
        g = gene_id_row.numel()
        pos_table = torch.full((self.embedding.num_embeddings,), -1, dtype=torch.long, device=device)
        pos_table[gene_id_row] = torch.arange(g, device=device)

        abs_nbr = self.neighbors[gene_id_row]
        valid = abs_nbr >= 0
        local_nbr = pos_table[abs_nbr.clamp_min(0)]
        valid = valid & (local_nbr >= 0)
        local_nbr = local_nbr.clamp_min(0)
        weight_pos = self.edge_weights_pos[gene_id_row] * valid.to(self.edge_weights_pos.dtype)
        weight_neg = self.edge_weights_neg[gene_id_row] * valid.to(self.edge_weights_neg.dtype)
        return local_nbr, weight_pos, weight_neg

    def forward(self, x: Tensor) -> Tensor:
        gene_emb = self.enc_norm(self.embedding(x))
        if self.use_wire:
            coords = self.manifold_coords[x]
            gene_emb = gene_emb + self.coord_embedding(coords)
        return gene_emb