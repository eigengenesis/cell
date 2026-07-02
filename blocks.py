import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneREPO(nn.Module):
    """
    Scalar position per gene per head from the gene hidden state via SwiGLU gating.
    r_i = SiLU(h W_g) * (h W_c);  z_i = r_i W_z, W_z near zero so training starts
    with no positional bias.
    """
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
        return self.W_z(r)                       # (B, G, num_heads)


class WireRotaryEncoding(nn.Module):
    """Holds the learnable WIRE frequency projection omega: (nhead, half, eigvec_dim)."""
    def __init__(self, eigvec_dim: int, head_dim: int, nhead: int):
        super().__init__()
        self.half = head_dim // 2
        omega = torch.zeros(nhead, self.half, eigvec_dim)
        for f in range(self.half):
            omega[:, f, f % eigvec_dim] = 1.0
        self.omega = nn.Parameter(omega)


def apply_rotary(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """x: (..., head_dim); angles: (..., half) broadcastable to x's leading dims."""
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
    """
    Differential attention via the identity
        (softmax(A1) - lambda * softmax(A2)) V = softmax(A1) V - lambda * softmax(A2) V,
    so two scaled_dot_product_attention calls replace the explicit (B,H,G,G) maps.
    REPO and WIRE are applied as additive rotary angles on q/k before attention.
    """
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

        self.lambda_proj = nn.Linear(embed_dim, num_heads, bias=False)
        nn.init.zeros_(self.lambda_proj.weight)

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
            ang = z.permute(0, 2, 1).unsqueeze(-1).float() * self.freqs   # (B, H, L, half)
        if self.wire is not None and coords is not None:
            w = torch.einsum('ble,hfe->bhlf', coords.float(), self.wire.omega.float())
            ang = w if ang is None else ang + w
        return ang

    def forward(self, noisy_y, x, spectral_coords=None, perturbation_emb=None):
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
        lambda_base = lambda_1 - lambda_2 + self.lambda_init

        if perturbation_emb is not None:
            lambda_delta = torch.sigmoid(self.lambda_proj(perturbation_emb))   # (B, H)
            lambda_full = lambda_base + lambda_delta.unsqueeze(2).unsqueeze(3) # (B, H, 1, 1)
        else:
            lambda_full = lambda_base

        o1 = F.scaled_dot_product_attention(q1, k1, v)
        o2 = F.scaled_dot_product_attention(q2, k2, v)
        attn = o1 - lambda_full * o2

        attn = self.subln(attn) * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(B, T, H * Dh)
        return self.out_proj(attn)