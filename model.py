import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict
import math
from timm.models.vision_transformer import Mlp
from torch.utils.checkpoint import checkpoint

from .layers import (GeneadaLN, ContinuousValueEncoder, GeneEncoder,
                     BatchLabelEncoder, TimestepEmbedder, ExprDecoder)
from .blocks import (MultiheadDiffAttn, modulate, GeneREPO,
                     VoidGeneBlock, ConditionEncoder)


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

    def forward(self, y, x, c, spectral_coords=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        normed_y = modulate(self.norm1(y), shift_msa, scale_msa)
        y = y + gate_msa.unsqueeze(1) * self.attn(normed_y, x, spectral_coords=spectral_coords)
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

    def forward(self, y, x, c, spectral_coords=None):
        y = self.diff_self_attn(y, y, c, spectral_coords=None)
        y = self.diff_cross_attn(y, x, c, spectral_coords=spectral_coords)
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
                 void_encode_blocks: int = 4,
                 void_think_steps: int = 8,
                 void_hidden: int = 512,
                 void_manifold_dim: int = 8,
                 void_neighbor_chunk: int = 256,
                 void_checkpoint: bool = False,
                 ):
        super().__init__()
        self.fusion_method = fusion_method
        self.perturbation_function = perturbation_function
        self.t_embedder = TimestepEmbedder(d_model)
        self.perturbation_embedder = BatchLabelEncoder(ntoken, d_model)
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.value_encoder_1 = ContinuousValueEncoder(d_model, dropout)
        self.value_encoder_2 = ContinuousValueEncoder(d_model, dropout)

        self.p_mask_embed = nn.Parameter(torch.randn(d_model))
        self.p_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.final_layer = ExprDecoder(d_model, explicit_zero_prob=False, use_batch_labels=True)

        if fusion_method == 'void':
            # Wire-free, mask-free plain gene embedding (no co-expression graph attention).
            self.encoder = GeneEncoder(
                ntoken, d_model, use_perturbation_interaction=False, use_wire=False,
                mask_path=mask_path, wire_path=None, grn_mask_path=None,
                nhead=nhead, dropout=dropout,
            )
            self.use_wire = False
            self.use_perturbation_interaction = False
            self.void_think_steps = int(void_think_steps)
            self.void_manifold_dim = int(void_manifold_dim)
            self.void_checkpoint = bool(void_checkpoint)
            self.cond_encoder = ConditionEncoder(d_model, ntoken, ntoken)
            self.coord_embedding = nn.Sequential(
                nn.Linear(max(1, void_manifold_dim), d_model), nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            self.input_fusion = nn.Sequential(
                nn.Linear(4 * d_model, d_model), nn.GELU(),
                nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            )
            self.void_encode = nn.ModuleList([
                VoidGeneBlock(d_model, void_hidden, dropout, residual_scale=0.1,
                              neighbor_chunk=void_neighbor_chunk)
                for _ in range(int(void_encode_blocks))
            ])
            self.ghost = VoidGeneBlock(d_model, void_hidden, dropout, residual_scale=0.1,
                                       neighbor_chunk=void_neighbor_chunk)
            self.ghost_gate_logit = nn.Parameter(torch.tensor(-2.1972))
            self.out_norm = nn.LayerNorm(d_model)
            self.blocks = nn.ModuleList([])
        else:
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

        self.initialize_weights()

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

    def set_geometry(self, neighbors, weights, coords):
        self.register_buffer('void_neighbors', neighbors.long(), persistent=False)
        self.register_buffer('void_weights', weights.float(), persistent=False)
        self.register_buffer('void_coords', coords.float(), persistent=False)

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

    def forward_void(self, gene_id, cell_1, t, cell_2, perturbation_id,
                     perturbation_gene_id, mode="predict_y"):
        if t.dim() == 0:
            t = t.repeat(cell_1.size(0))
        B, G = gene_id.shape
        gene_emb = self.encoder(gene_id[:1]).expand(B, -1, -1)             # static across cells

        coords = self.void_coords.to(dtype=cell_1.dtype)
        if coords.size(1) == 0:
            coords = torch.zeros(G, 1, device=cell_1.device, dtype=cell_1.dtype)
        coord_emb = self.coord_embedding(coords)[None].expand(B, -1, -1)

        x = self.input_fusion(torch.cat(
            [gene_emb, coord_emb, self.value_encoder_1(cell_1), self.value_encoder_2(cell_2)], dim=-1))
        global_state = x.mean(dim=1)
        c, pert = self.cond_encoder(t, perturbation_id, perturbation_gene_id)

        nbr, w = self.void_neighbors, self.void_weights
        use_ckpt = self.void_checkpoint and self.training

        def run(block, x_in, g_in):
            if use_ckpt:
                return checkpoint(block, x_in, g_in, c, nbr, w, use_reentrant=False)
            return block(x_in, g_in, c, nbr, w)

        for block in self.void_encode:
            x, global_state = run(block, x, global_state)

        x_init, g_init = x, global_state
        gate = torch.sigmoid(self.ghost_gate_logit)
        for _ in range(self.void_think_steps):
            cand_x, cand_g = run(self.ghost, x + x_init, global_state + g_init)
            x = x + gate * (cand_x - x)
            global_state = global_state + gate * (cand_g - global_state)

        x = self.out_norm(x)
        if mode == "predict_p":
            return self.p_head(x.mean(dim=1))
        x = torch.cat([x, pert[:, None, :].expand(-1, G, -1)], dim=-1)
        return self.final_layer(x)['pred']

    def forward(self, gene_id, cell_1, t, cell_2, perturbation_id=None, gene_id_all=None,
                perturbation_emb=None, perturbation_gene_id=None, mode="predict_y"):
        if t.dim() == 0:
            t = t.repeat(cell_1.size(0))

        if self.fusion_method == 'void':
            return self.forward_void(gene_id, cell_1, t, cell_2, perturbation_id,
                                     perturbation_gene_id, mode=mode)

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
                x = block(x, value_emb_2, t_emb, spectral_coords=spectral_coords)
            else:
                x = block(x, value_emb_2, t_emb)

        if mode == "predict_p":
            return self.p_head(x.mean(dim=1))

        x = torch.cat([x, perturbation_emb[:, None, :].expand(-1, x.size(1), -1)], dim=-1)
        return self.final_layer(x)['pred']