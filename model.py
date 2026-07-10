import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

try:
    from .layers import GeneEncoder, BatchLabelEncoder, ExprDecoder
    from .blocks import ValueEncoder, TimestepEmbedder, VoidGeneBlock
except ImportError:
    from layers import GeneEncoder, BatchLabelEncoder, ExprDecoder
    from blocks import ValueEncoder, TimestepEmbedder, VoidGeneBlock


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
                 corr_path: str = None,
                 think_steps: int = 8,
                 residual_scale: float = 0.1,
                 neighbor_cap: int = 128,
                 neighbor_gate: bool = True,
                 control_token_id: int | None = None,
                 **kwargs):
        super().__init__()
        self.perturbation_function = perturbation_function
        self.think_steps = int(think_steps)
        self.control_token_id = control_token_id

        self.encoder = GeneEncoder(
            ntoken, d_model,
            use_perturbation_interaction=use_perturbation_interaction,
            mask_path=mask_path, use_wire=use_wire, wire_path=wire_path,
            grn_mask_path=grn_mask_path, corr_path=corr_path, neighbor_cap=neighbor_cap,
        )
        self.use_perturbation_interaction = use_perturbation_interaction
        self.perturbation_embedder = BatchLabelEncoder(ntoken, d_model)

        self.value_current = ValueEncoder(d_model, dropout)
        self.value_control = ValueEncoder(d_model, dropout)
        self.t_embedder = TimestepEmbedder(d_model)

        self.condition_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.action_phi = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.action_rho = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.input_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )

        self.action_vector = nn.Linear(d_model, d_model)
        self.action_projection = nn.Linear(d_model, d_model, bias=False)
        nn.init.trunc_normal_(self.action_projection.weight, std=1e-4)

        self.encode = nn.ModuleList([
            VoidGeneBlock(d_model, d_hid, dropout, residual_scale,
                          neighbor_gate=neighbor_gate, use_signed_neighbors=True)
            for _ in range(nlayers)
        ])
        self.ghost = VoidGeneBlock(d_model, d_hid, dropout, residual_scale,
                                   neighbor_gate=neighbor_gate, use_signed_neighbors=True)
        self.ghost_gate_logit = nn.Parameter(torch.tensor(-2.1972))

        self.out_norm = nn.LayerNorm(d_model)
        self.p_mask_embed = nn.Parameter(torch.randn(d_model))
        self.p_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.final_layer = ExprDecoder(d_model, explicit_zero_prob=False, use_batch_labels=True)

        self.velocity_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 1, bias=True),
        )
        nn.init.constant_(self.velocity_gate[-1].bias, -2.0)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        for block in [*self.encode, self.ghost]:
            block.reset_void_parameters()
        nn.init.trunc_normal_(self.action_projection.weight, std=1e-2)
        nn.init.constant_(self.velocity_gate[-1].bias, -2.0)

    def compose_perturbation(self, perturbation_id=None, perturbation_emb=None, cell_1=None):
        if perturbation_id is None:
            global_action = self.get_perturbation_emb(
                perturbation_id=None, perturbation_emb=perturbation_emb, cell_1=cell_1
            )
            return global_action, None, None

        if self.perturbation_function != 'crisper':
            components = self.perturbation_embedder(perturbation_id)
            valid = torch.ones(components.shape[:2], device=components.device, dtype=torch.bool)
        else:
            components = self.encoder(perturbation_id)
            valid = torch.ones(components.shape[:2], device=components.device, dtype=torch.bool)
            if self.control_token_id is not None:
                valid = perturbation_id.ne(int(self.control_token_id))

        encoded = self.action_phi(components) * valid.unsqueeze(-1).to(components.dtype)
        global_action = self.action_rho(encoded.sum(dim=1))
        global_action = global_action * valid.any(dim=1, keepdim=True).to(global_action.dtype)
        return global_action, components, valid

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

    def target_gene_gate(self, perturbation_id, gene_id_row, b: int, g: int, device, dtype):
        """Hard gate over genes, 1.0 at the literal perturbed gene node(s). Only
        meaningful in 'crisper' mode where perturbation_id already holds gene
        vocabulary ids; otherwise returns None and the caller falls back to a
        uniform broadcast, matching the original per-layer adapter behavior."""
        if self.perturbation_function != 'crisper' or perturbation_id is None:
            return None
        vocab_size = self.encoder.embedding.num_embeddings
        pos_table = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        pos_table[gene_id_row] = torch.arange(g, device=device)

        pid = perturbation_id.long()
        in_range = (pid >= 0) & (pid < vocab_size)
        local = pos_table[pid.clamp(0, vocab_size - 1)]
        valid = in_range & (local >= 0)

        gate = torch.zeros((b, g), device=device, dtype=dtype)
        if valid.any():
            gate.scatter_add_(1, local.clamp_min(0), valid.to(dtype))
            gate.clamp_(0.0, 1.0)
        return gate

    def local_action_field(self, perturbation_id, components, valid, gene_id_row, b, g, dtype):
        if self.perturbation_function != 'crisper' or perturbation_id is None or components is None:
            return None
        vocab_size = self.encoder.embedding.num_embeddings
        pos_table = torch.full((vocab_size,), -1, dtype=torch.long, device=gene_id_row.device)
        pos_table[gene_id_row] = torch.arange(g, device=gene_id_row.device)
        local = pos_table[perturbation_id.clamp(0, vocab_size - 1)]
        active = valid & (local >= 0)
        action_components = self.action_vector(components).to(dtype=dtype)
        field = torch.zeros((b, g, action_components.size(-1)), device=gene_id_row.device, dtype=dtype)
        if active.any():
            scatter_idx = local.clamp_min(0).unsqueeze(-1).expand_as(action_components)
            field.scatter_add_(1, scatter_idx, action_components * active.unsqueeze(-1).to(dtype))
        return field

    @staticmethod
    def _build_adjacency(local_nbr: Tensor, edge_weight_pos: Tensor, edge_weight_neg: Tensor,
                         eps: float = 1e-6):
        G, k = local_nbr.shape
        device = local_nbr.device
        rows = torch.arange(G, device=device).unsqueeze(1).expand(G, k).reshape(-1)
        cols = local_nbr.reshape(-1)

        adj_pos = torch.zeros((G, G), device=device, dtype=edge_weight_pos.dtype)
        adj_pos.index_put_((rows, cols), edge_weight_pos.reshape(-1), accumulate=True)
        adj_pos = adj_pos / adj_pos.sum(dim=1, keepdim=True).clamp_min(eps)

        adj_neg = torch.zeros((G, G), device=device, dtype=edge_weight_neg.dtype)
        adj_neg.index_put_((rows, cols), edge_weight_neg.reshape(-1), accumulate=True)
        adj_neg = adj_neg / adj_neg.sum(dim=1, keepdim=True).clamp_min(eps)
        return adj_pos, adj_neg

    @staticmethod
    def _neighbor_messages(x: Tensor, adj_pos: Tensor, adj_neg: Tensor):
        B, G, d = x.shape
        x_flat = x.permute(1, 0, 2).reshape(G, B * d)
        pos_msg = torch.matmul(adj_pos, x_flat).reshape(G, B, d).permute(1, 0, 2)
        neg_msg = torch.matmul(adj_neg, x_flat).reshape(G, B, d).permute(1, 0, 2)
        return pos_msg, neg_msg

    @staticmethod
    def apply_ghost_update(x, global_state, ghost_x, global_input,
                           cand_x, cand_global, ghost_gate):
        x = x + ghost_gate * (cand_x - ghost_x)
        global_state = global_state + ghost_gate * (cand_global - global_input)
        return x, global_state

    def forward(self, gene_id, cell_1, t, cell_2, perturbation_id=None, gene_id_all=None,
                    perturbation_emb=None, mode="predict_y"):
        with torch.autocast(device_type=cell_1.device.type, dtype=torch.bfloat16):
            return self._forward(gene_id, cell_1, t, cell_2, perturbation_id,
                                 gene_id_all, perturbation_emb, mode)

    def _forward(self, gene_id, cell_1, t, cell_2, perturbation_id=None, gene_id_all=None,
                    perturbation_emb=None, mode="predict_y"):
        if t.dim() == 0:
            t = t.repeat(cell_1.size(0))

        B, G = cell_1.shape
        device = cell_1.device
        gene_row = gene_id[0]

        gene_emb = self.encoder(gene_id)
        value_x = self.value_current(cell_1)
        value_c = self.value_control(cell_2)

        perturbation_emb, action_components, action_valid = self.compose_perturbation(
            perturbation_id, perturbation_emb, cell_1
        )
        t_emb = self.t_embedder(t)
        cond = self.condition_fusion(torch.cat([t_emb, perturbation_emb], dim=-1))

        x = self.input_fusion(torch.cat([gene_emb, value_x, value_c], dim=-1))

        broadcast = perturbation_emb[:, None, :].expand(-1, G, -1)
        local_action = self.local_action_field(
            perturbation_id, action_components, action_valid, gene_row, B, G, x.dtype
        )
        action = broadcast if local_action is None else broadcast + local_action
        x = x + self.action_projection(action)

        global_state = x.mean(dim=1)
        source_state = value_c

        if self.use_perturbation_interaction:
            local_nbr, edge_weight_pos, edge_weight_neg = self.encoder.local_graph(gene_row)
            adj_pos, adj_neg = self._build_adjacency(local_nbr, edge_weight_pos, edge_weight_neg)
        else:
            adj_pos, adj_neg = None, None

        for block in self.encode:
            if adj_pos is not None:
                pos_msg, neg_msg = self._neighbor_messages(x, adj_pos, adj_neg)
            else:
                pos_msg, neg_msg = torch.zeros_like(x), torch.zeros_like(x)
            x, global_state = block(x, global_state, pos_msg, neg_msg, cond, source_state)

        x_init, global_init = x, global_state
        ghost_gate = torch.sigmoid(self.ghost_gate_logit)
        ghost_cache = self.ghost.precompute(cond, source_state)
        for _ in range(self.think_steps):
            ghost_x = x + x_init
            if adj_pos is not None:
                pos_msg, neg_msg = self._neighbor_messages(ghost_x, adj_pos, adj_neg)
            else:
                pos_msg, neg_msg = torch.zeros_like(ghost_x), torch.zeros_like(ghost_x)
            cand_x, cand_global = self.ghost(
                ghost_x, global_state + global_init, pos_msg, neg_msg, cond, source_state,
                cache=ghost_cache,
            )
            global_input = global_state + global_init
            x, global_state = self.apply_ghost_update(
                x, global_state, ghost_x, global_input, cand_x, cand_global, ghost_gate
            )

        x = self.out_norm(x)

        if mode == "predict_p":
            return self.p_head(x.mean(dim=1))

        x_cat = torch.cat([x, perturbation_emb[:, None, :].expand(-1, G, -1)], dim=-1)
        vgate = torch.sigmoid(self.velocity_gate(x_cat))
        return self.final_layer(x_cat)['pred'] * vgate.squeeze(-1)
