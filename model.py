import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict
import math
from timm.models.vision_transformer import Mlp

from .layers import (GeneadaLN, ContinuousValueEncoder, GeneEncoder,
                     BatchLabelEncoder, TimestepEmbedder, ExprDecoder)
from .blocks import MultiheadDiffAttn, modulate, GeneREPO


class DifferentialTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, depth, mlp_ratio=4.0, cross=False,
                 use_repo=True, use_wire=False, eigvec_dim=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadDiffAttn(
            hidden_size, num_heads, depth,
            cross=cross, use_repo=use_repo, use_wire=use_wire, eigvec_dim=eigvec_dim,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, y, x, c, spectral_coords=None, perturbation_emb=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        normed_y = modulate(self.norm1(y), shift_msa, scale_msa)
        y = y + gate_msa.unsqueeze(1) * self.attn(
            normed_y, x, spectral_coords=spectral_coords, perturbation_emb=perturbation_emb
        )
        y = y + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(y), shift_mlp, scale_mlp))
        return y


class PerceiverBlock(nn.Module):
    def __init__(self, d_in, d_latent, heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln_z1 = nn.LayerNorm(d_latent)
        self.q  = nn.Linear(d_latent, d_latent)
        self.k  = nn.Linear(d_in, d_latent)
        self.v  = nn.Linear(d_in, d_latent)
        self.q2 = nn.Linear(d_latent, d_latent)
        self.k2 = nn.Linear(d_latent, d_latent)
        self.v2 = nn.Linear(d_latent, d_latent)
        self.cross     = nn.MultiheadAttention(d_latent, heads, dropout=dropout, batch_first=True)
        self.ln_z2     = nn.LayerNorm(d_latent)
        self.self_attn = nn.MultiheadAttention(d_latent, heads, dropout=dropout, batch_first=True)
        self.ln_z3     = nn.LayerNorm(d_latent)
        self.mlp = nn.Sequential(
            nn.Linear(d_latent, int(mlp_ratio * d_latent)), nn.GELU(),
            nn.Linear(int(mlp_ratio * d_latent), d_latent)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_latent, 6 * d_latent, bias=True)
        )

    def forward(self, z, x, t):
        shift_self, scale_self, gate_self, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t).chunk(6, dim=1)
        z = z + self.cross(self.q(self.ln_z1(z)), self.k(x), self.v(x))[0]
        z_normed = modulate(self.ln_z2(z), shift_self, scale_self)
        z = z + gate_self.unsqueeze(1) * self.self_attn(self.q2(z_normed), self.k2(z_normed), self.v2(z_normed))[0]
        z = z + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.ln_z3(z), shift_mlp, scale_mlp))
        return z


class DiffPerceiverBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, depth, mlp_ratio=4.0,
                 use_repo=True, use_wire=False, eigvec_dim=None):
        super().__init__()
        self.diff_self_attn = DifferentialTransformerBlock(
            hidden_size, num_heads, depth, mlp_ratio=mlp_ratio,
            cross=True, use_repo=use_repo, use_wire=use_wire, eigvec_dim=eigvec_dim,
        )
        self.diff_cross_attn = DifferentialTransformerBlock(
            hidden_size, num_heads, depth, mlp_ratio=mlp_ratio,
            cross=False, use_repo=use_repo, use_wire=use_wire, eigvec_dim=eigvec_dim,
        )
    
    # Self pass: WIRE off (graph coords encode inter-gene structure, not intra-sequence
    # position); REPO still assigns content-based relative positions. Cross pass uses WIRE.

    def forward(self, y, x, c, spectral_coords=None, perturbation_emb=None):
        y = self.diff_self_attn(y, y, c, spectral_coords=None, perturbation_emb=perturbation_emb)
        y = self.diff_cross_attn(y, x, c, spectral_coords=spectral_coords, perturbation_emb=perturbation_emb)
        return y


class model(nn.Module):
    def __init__(self,
                 ntoken: int = 6000,
                 d_model: int = 512,
                 nhead: int = 8,
                 d_hid: int = 2048,
                 nlayers: int = 8,
                 dropout: float = 0.1,
                 fusion_method: str = 'cross',
                 perturbation_function: str = 'crisper',
                 use_perturbation_interaction: bool = True,
                 mask_path: str = None,
                 wire_path: str = None,
                 use_wire: bool = True,
                 use_repo: bool = True,
                 grn_mask_path: str = None,
                 ):
        super().__init__()
        self.t_embedder = TimestepEmbedder(d_model)
        self.perturbation_embedder = BatchLabelEncoder(ntoken, d_model)
        self.fusion_method = fusion_method
        self.perturbation_function = perturbation_function
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )

        self.value_encoder_1 = ContinuousValueEncoder(d_model, dropout)
        self.value_encoder_2 = ContinuousValueEncoder(d_model, dropout)
        self.encoder = GeneEncoder(
            ntoken, d_model,
            use_perturbation_interaction=use_perturbation_interaction,
            mask_path=mask_path, use_wire=use_wire, wire_path=wire_path,
            nhead=nhead, dropout=dropout, grn_mask_path=grn_mask_path,
        )
        self.use_perturbation_interaction = use_perturbation_interaction

        wire_in_blocks = use_wire and use_perturbation_interaction and hasattr(self.encoder, 'spectral_coords')
        self.use_wire = wire_in_blocks
        eigvec_dim = self.encoder.spectral_coords.shape[1] if wire_in_blocks else None

        if fusion_method == 'differential_transformer':
            self.blocks = nn.ModuleList([
                DifferentialTransformerBlock(
                    d_model, nhead, i, mlp_ratio=4.0, cross=False,
                    use_repo=use_repo, use_wire=wire_in_blocks, eigvec_dim=eigvec_dim,
                ) for i in range(nlayers)
            ])
        elif fusion_method == 'differential_perceiver':
            self.blocks = nn.ModuleList([
                DiffPerceiverBlock(
                    d_model, nhead, i, mlp_ratio=4.0,
                    use_repo=use_repo, use_wire=wire_in_blocks, eigvec_dim=eigvec_dim,
                ) for i in range(nlayers)
            ])
        elif fusion_method == 'perceiver':
            self.blocks = nn.ModuleList([
                PerceiverBlock(d_model, d_model, heads=nhead, mlp_ratio=4.0, dropout=0.1)
                for _ in range(nlayers)
            ])
        else:
            raise ValueError(f"Invalid fusion method: {fusion_method}")

        self.gene_adaLN = nn.ModuleList([GeneadaLN(d_model, dropout) for _ in range(nlayers)])
        self.adapter_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.LeakyReLU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model), nn.LeakyReLU(),
            ) for _ in range(nlayers)
        ])

        self.p_mask_embed = nn.Parameter(torch.randn(d_model))
        self.p_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.final_layer = ExprDecoder(d_model, explicit_zero_prob=False, use_batch_labels=True)
        self.initialize_weights()

        self.velocity_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 1, bias=True),
        )
        nn.init.constant_(self.velocity_gate[-1].bias, -2.0)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        for module in self.modules():
            if isinstance(module, GeneREPO):
                nn.init.normal_(module.W_z.weight, std=0.01)

    def get_perturbation_emb(self, perturbation_id=None, perturbation_emb=None,
                             cell_1=None, use_mask: bool = False):
        if use_mask:
            B = cell_1.size(0)
            return self.p_mask_embed[None, :].expand(B, -1).to(cell_1.device, dtype=cell_1.dtype)

        assert perturbation_emb is None or perturbation_id is None
        if perturbation_id is not None:
            if self.perturbation_function == 'crisper':
                perturbation_emb = self.encoder(perturbation_id)
            else:
                perturbation_emb = self.perturbation_embedder(perturbation_id)
            perturbation_emb = perturbation_emb.mean(1)
        elif perturbation_emb is not None:
            perturbation_emb = perturbation_emb.to(cell_1.device, dtype=cell_1.dtype)
            if perturbation_emb.dim() == 1:
                perturbation_emb = perturbation_emb.unsqueeze(0)
            if perturbation_emb.size(0) == 1:
                perturbation_emb = perturbation_emb.expand(cell_1.shape[0], -1).contiguous()
            perturbation_emb = self.perturbation_embedder.enc_norm(perturbation_emb)
        return perturbation_emb

    def forward(self, gene_id, cell_1, t, cell_2, perturbation_id=None, gene_id_all=None,
                perturbation_emb=None, mode="predict_y"):
        if t.dim() == 0:
            t = t.repeat(cell_1.size(0))

        gene_emb = self.encoder(gene_id)
        spectral_coords = self.encoder.spectral_coords[gene_id] if self.use_wire else None

        value_emb_1 = self.value_encoder_1(cell_1) + gene_emb
        value_emb_2 = self.value_encoder_2(cell_2) + gene_emb

        value_emb = self.fusion_layer(torch.cat([value_emb_1, value_emb_2], dim=-1))
        x = value_emb
        t_emb = self.t_embedder(t)
        perturbation_emb = self.get_perturbation_emb(perturbation_id, perturbation_emb, cell_1)

        for i, block in enumerate(self.blocks):
            x = self.gene_adaLN[i](gene_emb, x)
            perturbation_exp = perturbation_emb[:, None, :].expand(-1, x.size(1), -1)
            x = self.adapter_layer[i](torch.cat([x, perturbation_exp], dim=-1))
            if isinstance(block, (DifferentialTransformerBlock, DiffPerceiverBlock)):
                x = block(x, value_emb_2, t_emb, spectral_coords=spectral_coords,
                        perturbation_emb=perturbation_emb)
            else:
                x = block(x, value_emb_2, t_emb)

        if mode == "predict_p":
            return self.p_head(x.mean(dim=1))

        x_cat = torch.cat([x, perturbation_emb[:, None, :].expand(-1, x.size(1), -1)], dim=-1)
        gate = torch.sigmoid(self.velocity_gate(x_cat))          # (B, G, 1)
        return self.final_layer(x_cat)['pred'] * gate.squeeze(-1)