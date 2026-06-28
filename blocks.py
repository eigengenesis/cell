import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneREPO(nn.Module):
    """SwiGLU-gated scalar position per gene per head. Used only by differential fusion paths."""
    def __init__(self, embed_dim: int, num_heads: int, d_pos: int = None):
        super().__init__()
        if d_pos is None:
            d_pos = max(embed_dim // 8, 16)
        self.W_g = nn.Linear(embed_dim, d_pos, bias=False)
        self.W_c = nn.Linear(embed_dim, d_pos, bias=False)
        self.W_z = nn.Linear(d_pos, num_heads, bias=False)
        nn.init.normal_(self.W_z.weight, std=0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        r = F.silu(self.W_g(h)) * self.W_c(h)
        return self.W_z(r)


class WireRotaryEncoding(nn.Module):
    """Learnable WIRE frequency projection. Used only by differential fusion paths."""
    def __init__(self, eigvec_dim: int, head_dim: int, nhead: int):
        super().__init__()
        self.half = head_dim // 2
        omega = torch.zeros(nhead, self.half, eigvec_dim)
        for f in range(self.half):
            omega[:, f, f % eigvec_dim] = 1.0
        self.omega = nn.Parameter(omega)


def apply_rotary(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    half = angles.shape[-1]
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half:2 * half]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if x.shape[-1] > 2 * half:
        out = torch.cat([out, x[..., 2 * half:]], dim=-1)
    return out


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            out = out * self.weight
        return out


class MultiheadDiffAttn(nn.Module):
    """Differential attention for the differential_transformer / differential_perceiver paths."""
    def __init__(self, embed_dim, num_heads, depth, cross=False,
                 use_repo=True, use_wire=False, eigvec_dim=None):
        super().__init__()
        self.cross = cross
        self.use_repo = use_repo
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.half = self.head_dim // 2

        self.q_proj_1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj_1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj_2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj_2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))

        self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

        if use_repo:
            self.repo_query   = GeneREPO(embed_dim, num_heads)
            self.repo_context = GeneREPO(embed_dim, num_heads)
        self.wire = WireRotaryEncoding(eigvec_dim, self.head_dim, num_heads) \
            if (use_wire and eigvec_dim is not None) else None

        self.register_buffer(
            'freqs',
            1.0 / (10000.0 ** (torch.arange(0, self.half).float() / self.half)),
            persistent=False,
        )

    def _angles(self, z, coords):
        ang = None
        if self.use_repo and z is not None:
            ang = z.permute(0, 2, 1).unsqueeze(-1).float() * self.freqs
        if self.wire is not None and coords is not None:
            w = torch.einsum('ble,hfe->bhlf', coords.float(), self.wire.omega.float())
            ang = w if ang is None else ang + w
        return ang

    def forward(self, noisy_y, x, spectral_coords=None):
        B, T, _ = noisy_y.shape
        H, Dh = self.num_heads, self.head_dim

        z_n = self.repo_query(noisy_y) if self.use_repo else None
        z_x = self.repo_context(x) if self.use_repo else None

        if self.cross:
            q1, k1 = self.q_proj_1(noisy_y), self.k_proj_1(x)
            q2, k2 = self.q_proj_2(noisy_y), self.k_proj_2(x)
            zq1, zk1, zq2, zk2 = z_n, z_x, z_n, z_x
        else:
            q1, k1 = self.q_proj_1(noisy_y), self.k_proj_1(noisy_y)
            q2, k2 = self.q_proj_2(x), self.k_proj_2(x)
            zq1, zk1, zq2, zk2 = z_n, z_n, z_n, z_x
        v = self.v_proj(noisy_y)

        def shp(t):
            return t.view(B, t.shape[1], H, Dh).transpose(1, 2)
        q1, k1, q2, k2, v = shp(q1), shp(k1), shp(q2), shp(k2), shp(v)

        def rot(tensor, z):
            if spectral_coords is None and not self.use_repo:
                return tensor
            L = tensor.shape[2]
            coords = spectral_coords[:, :L] if spectral_coords is not None else None
            ang = self._angles(z, coords)
            if ang is None:
                return tensor
            return apply_rotary(tensor.float(), ang).type_as(tensor)
        q1, k1, q2, k2 = rot(q1, zq1), rot(k1, zk1), rot(q2, zq2), rot(k2, zk2)

        lambda_1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum().float()).type_as(q1)
        lambda_2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum().float()).type_as(q1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        o1 = F.scaled_dot_product_attention(q1, k1, v)
        o2 = F.scaled_dot_product_attention(q2, k2, v)
        attn = o1 - lambda_full * o2

        attn = self.subln(attn) * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(B, T, H * Dh)
        return self.out_proj(attn)


# ---------------------------------------------------------------------------
# VOID fusion components: signed-mean graph message passing + global token.
# No attention, no REPO, no WIRE rotary.
# ---------------------------------------------------------------------------


class TimestepEmbedderLocal(nn.Module):
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


class SignedNeighborMessage(nn.Module):
    """
    Two weighted means over a fixed panel-local KNN graph, split by edge sign.
    Gather + weighted sum only (no per-neighbor projection), chunked over genes.
    """
    def __init__(self, neighbor_chunk: int = 256):
        super().__init__()
        self.neighbor_chunk = int(neighbor_chunk)

    def forward(self, x, neighbors, weights):
        B, G, d = x.shape
        k = neighbors.shape[1]
        eps = 1e-6
        chunk = self.neighbor_chunk if self.neighbor_chunk > 0 else G
        pos_w = weights.clamp_min(0.0)
        neg_w = (-weights).clamp_min(0.0)
        if chunk >= G:
            gathered = x.index_select(1, neighbors.reshape(-1)).reshape(B, G, k, d)
            pos = (gathered * pos_w[None, :, :, None]).sum(2) / pos_w.sum(1).clamp_min(eps)[None, :, None]
            neg = (gathered * neg_w[None, :, :, None]).sum(2) / neg_w.sum(1).clamp_min(eps)[None, :, None]
            return pos, neg
        pos_out = torch.empty_like(x)
        neg_out = torch.empty_like(x)
        for s in range(0, G, chunk):
            e = min(s + chunk, G)
            idx = neighbors[s:e].reshape(-1)
            gathered = x.index_select(1, idx).reshape(B, e - s, k, d)
            pw = pos_w[s:e]
            nw = neg_w[s:e]
            pos_out[:, s:e] = (gathered * pw[None, :, :, None]).sum(2) / pw.sum(1).clamp_min(eps)[None, :, None]
            neg_out[:, s:e] = (gathered * nw[None, :, :, None]).sum(2) / nw.sum(1).clamp_min(eps)[None, :, None]
        return pos_out, neg_out


class VoidGeneBlock(nn.Module):
    """
    Signed graph message passing with a global token, AdaLN-conditioned, gated residual.
    Self / positive-neighbor / negative-neighbor streams share one fused contraction.
    A single instance is reused for the think loop.
    """
    def __init__(self, dim, hidden, dropout, residual_scale, neighbor_chunk=256):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.gene_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.triple_contract = nn.Linear(3 * dim, hidden, bias=False)
        self.global_contract = nn.Linear(dim, hidden, bias=False)
        self.signed = SignedNeighborMessage(neighbor_chunk)
        self.expand = nn.Linear(hidden, dim, bias=False)
        self.global_update = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, global_state, condition, neighbors, weights):
        sx, ax, gx, sg, ag, gg = self.ada(condition).chunk(6, dim=-1)
        x_mod = self.gene_norm(x) * (1.0 + ax[:, None, :]) + sx[:, None, :]
        g_mod = self.global_norm(global_state) * (1.0 + ag) + sg

        pos_mean, neg_mean = self.signed(x, neighbors, weights)
        mixed = self.triple_contract(torch.cat([x_mod, pos_mean, neg_mean], dim=-1))
        mixed = mixed + self.global_contract(g_mod)[:, None, :]
        dx = self.expand(self.drop(F.gelu(mixed)))
        x = x + self.residual_scale * torch.tanh(gx[:, None, :]) * dx

        pooled = x.mean(dim=1)
        dg = self.global_update(torch.cat([g_mod, pooled], dim=-1))
        global_state = global_state + self.residual_scale * torch.tanh(gg) * dg
        return x, global_state

class ConditionEncoder(nn.Module):
    """Fuse timestep, mean perturbation-token embedding, and mean perturbed-gene embedding."""
    def __init__(self, dim, n_perturbations, n_genes):
        super().__init__()
        self.t_embedder = TimestepEmbedderLocal(dim)
        self.perturbation_embedding = nn.Embedding(n_perturbations, dim)
        self.pert_gene_embedding = nn.Embedding(n_genes + 1, dim)
        self.fusion = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t, perturbation_id, perturbation_gene_id):
        time = self.t_embedder(t)
        pert = self.perturbation_embedding(perturbation_id.clamp_min(0)).mean(dim=1)
        if perturbation_gene_id is None:
            pert_gene = torch.zeros_like(pert)
        else:
            gene_ids = perturbation_gene_id.clamp_min(-1) + 1
            pert_gene = self.pert_gene_embedding(gene_ids).mean(dim=1)
        c = self.fusion(torch.cat([time, pert, pert_gene], dim=-1))
        return c, pert