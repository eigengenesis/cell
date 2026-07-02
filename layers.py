import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict
import math
import torch.nn.functional as F
from .blocks import WireRotaryEncoding, apply_rotary


class GeneNeighborAttention(nn.Module):
    """
    Each query gene attends only to its k graph neighbors. Keys and values are
    computed once per forward as (G_q, cap, H, Dh) (batch-independent: they come
    from the embedding table and the fixed neighbor table) and broadcast over the
    batch in the final einsum. WIRE rotary is applied batch-free on the neighbor
    coordinates. Genes with no valid neighbor produce a zero update via nan_to_num.
    """
    def __init__(self, d_model, nhead, mlp_ratio=4, dropout=0.1, eigvec_dim=None, use_wire=True):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scaling = self.head_dim ** -0.5
        self.use_wire = use_wire and eigvec_dim is not None

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        ffn = int(mlp_ratio * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d_model)
        )
        if self.use_wire:
            self.wire = WireRotaryEncoding(eigvec_dim, self.head_dim, nhead)

    def forward(self, x, nbr_emb, valid, coords_q=None, coords_nbr=None):
        # x: (B, G, d); nbr_emb: (G, cap, d); valid: (G, cap) bool
        # coords_q: (G, e); coords_nbr: (G, cap, e)
        B, G, d = x.shape
        cap = nbr_emb.shape[1]
        H, Dh = self.nhead, self.head_dim

        q = self.q_proj(self.norm1(x)).view(B, G, H, Dh)
        k = self.k_proj(nbr_emb).view(G, cap, H, Dh)
        v = self.v_proj(nbr_emb).view(G, cap, H, Dh)

        if self.use_wire and coords_q is not None:
            aq = torch.einsum('ge,hfe->ghf', coords_q.float(), self.wire.omega.float())
            q = apply_rotary(q.float(), aq).type_as(q)
            ak = torch.einsum('gke,hfe->gkhf', coords_nbr.float(), self.wire.omega.float())
            k = apply_rotary(k.float(), ak).type_as(k)

        attn = torch.einsum('bghd,gkhd->bghk', q * self.scaling, k)            # (B, G, H, cap)
        attn = attn.masked_fill(~valid.view(1, G, 1, cap), float('-inf'))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(attn)
        attn = torch.nan_to_num(attn, nan=0.0)                                 # empty rows -> zero update
        attn = self.drop(attn)

        out = torch.einsum('bghk,gkhd->bghd', attn, v).reshape(B, G, d)
        out = self.out_proj(out)

        x = x + self.drop(out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class GeneadaLN(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, gene_emb: Tensor, value_emb: Tensor) -> Tensor:
        shift, gate, scale = self.adaLN_modulation(gene_emb).chunk(3, dim=-1)
        return value_emb + gate * (self.norm(value_emb) * scale + shift)


class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        nhead: int = 8,
        use_perturbation_interaction: bool = False,
        dropout: float = 0.1,
        mask_path: str = None,
        use_wire: bool = True,
        wire_path: str = None,
        grn_mask_path: str = None,
        neighbor_cap: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)
        self.use_perturbation_interaction = use_perturbation_interaction
        self.use_wire = use_wire
        self.use_grn = grn_mask_path is not None and grn_mask_path != ''
        self.neighbor_cap = neighbor_cap

        if use_perturbation_interaction:
            # Stored adjacency has True = NON-neighbor (~99% dense); real KNN are the
            # False entries. Invert, drop self-loops, gather into a (vocab, cap) id table.
            adj = ~torch.load(mask_path).bool()
            adj.fill_diagonal_(False)
            self.register_buffer('coexp_neighbors',
                                 self._adjacency_to_neighbor_table(adj, neighbor_cap))

            eigvec_dim = None
            if use_wire and wire_path is not None:
                eigvecs = torch.load(wire_path, weights_only=True).float()
                self.register_buffer('spectral_coords', eigvecs)
                eigvec_dim = eigvecs.shape[1]

            self.coexp_attn = GeneNeighborAttention(
                embedding_dim, nhead, mlp_ratio=4, dropout=dropout,
                eigvec_dim=eigvec_dim, use_wire=(use_wire and eigvec_dim is not None),
            )

            if self.use_grn:
                grn_raw = torch.load(grn_mask_path)
                if grn_raw.dtype == torch.bool and grn_raw.dim() == 2 and grn_raw.shape[0] == grn_raw.shape[1]:
                    g = grn_raw.clone()
                    g.fill_diagonal_(False)
                    grn_table = self._adjacency_to_neighbor_table(g, neighbor_cap)
                else:
                    grn_table = grn_raw.long()
                self.register_buffer('grn_neighbors', grn_table)
                self.grn_attn = GeneNeighborAttention(
                    embedding_dim, nhead, mlp_ratio=4, dropout=dropout,
                    eigvec_dim=None, use_wire=False,
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

    def _gather_neighbors(self, gene_ids_row, table, with_coords):
        nbr = table[gene_ids_row]                                    # (G, cap)
        valid = nbr >= 0
        nbr_c = nbr.clamp_min(0)
        nbr_emb = self.enc_norm(self.embedding(nbr_c))               # (G, cap, d)
        coords = None
        if with_coords and hasattr(self, 'spectral_coords'):
            coords = self.spectral_coords[nbr_c] * valid.unsqueeze(-1).to(self.spectral_coords.dtype)
        return nbr_emb, valid, coords

    def forward(self, x: Tensor) -> Tensor:
        gene_ids = x
        x = self.enc_norm(self.embedding(x))
        if not self.use_perturbation_interaction:
            return x

        with_coords = self.use_wire and hasattr(self, 'spectral_coords')
        coords_q = self.spectral_coords[gene_ids[0]] if with_coords else None

        nbr_emb, valid, coords_nbr = self._gather_neighbors(
            gene_ids[0], self.coexp_neighbors, with_coords=with_coords)
        x = self.coexp_attn(x, nbr_emb, valid, coords_q, coords_nbr)

        if self.use_grn:
            grn_emb, grn_valid, _ = self._gather_neighbors(
                gene_ids[0], self.grn_neighbors, with_coords=False)
            x = self.grn_attn(x, grn_emb, grn_valid)
        return x


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


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))