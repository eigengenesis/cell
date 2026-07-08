#!/usr/bin/env python3
"""
Standalone VOID Cell trainer.

No scDFM imports and no gene tokenizer. Genes are direct AnnData column indices.
"""

from __future__ import annotations

from data import COMBOSCIPLEX_DEFAULT_TEST, DEFAULT_CELL_EVAL_SKIP, seed_everything, PerturbationBatchDataset, prepare_norman, prepare_combosciplex
from eval import (
    evaluate,
    make_flow_noise,
    format_duration,
    write_metrics,
    metrics_log_line,
    median_sigmas,
    mmd_multi_sigma,
)

import argparse
import csv
import json
import math
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from model import VoidCellModel, build_coexpression_geometry
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset









































def delta_noise_scale_for_step(step: int, args) -> float:
    end = float(args.delta_noise_end)
    warmup = max(int(args.delta_noise_warmup), 0)
    if warmup <= 0:
        return end
    progress = min(max(float(step) / float(warmup), 0.0), 1.0)
    blend = 0.5 - 0.5 * math.cos(math.pi * progress)
    return float(args.delta_noise_start) + (end - float(args.delta_noise_start)) * blend


def warmup_weight(weight: float, step: int, warmup: int) -> float:
    weight = float(weight)
    warmup = max(int(warmup), 0)
    if weight == 0.0 or warmup <= 0:
        return weight
    progress = min(max(float(step) / float(warmup), 0.0), 1.0)
    return weight * (0.5 - 0.5 * math.cos(math.pi * progress))


def scheduled_weight(
    weight: float,
    step: int,
    warmup: int,
    final_weight: float | None,
    transition_start: int,
    transition_steps: int,
) -> float:
    current = warmup_weight(weight, step, warmup)
    if final_weight is None:
        return current
    transition_start = max(int(transition_start), 0)
    transition_steps = max(int(transition_steps), 0)
    if transition_steps <= 0 or step < transition_start:
        return current
    progress = min(max(float(step - transition_start) / float(transition_steps), 0.0), 1.0)
    blend = 0.5 - 0.5 * math.cos(math.pi * progress)
    return current * (1.0 - blend) + float(final_weight) * blend


def deg_weights_from_condition_delta(
    condition_delta: torch.Tensor | None,
    sample_genes: torch.Tensor,
    epsilon: float,
) -> torch.Tensor | None:
    if condition_delta is None:
        return None
    lfc = condition_delta[:, sample_genes].abs().float().mean(dim=0)
    max_lfc = lfc.max().clamp_min(1e-12)
    return float(epsilon) + (1.0 - float(epsilon)) * (lfc / max_lfc)


def weighted_sqdist(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    x = x.float()
    y = y.float()
    if w is not None:
        scale = w.float().clamp_min(0.0).sqrt()
        x = x * scale
        y = y * scale
        denom = w.float().clamp_min(0.0).sum().clamp_min(1e-6)
    else:
        denom = torch.as_tensor(max(x.size(-1), 1), device=x.device, dtype=x.dtype)
    cost = torch.cdist(x, y, p=2).square()
    return cost / denom if normalize else cost


def sinkhorn_value(cost: torch.Tensor, eps: torch.Tensor | float, n_iters: int) -> torch.Tensor:
    cost = cost.float()
    eps = torch.as_tensor(eps, device=cost.device, dtype=cost.dtype).clamp_min(1e-6)
    m, n = cost.shape
    log_a = torch.full((m,), -math.log(max(m, 1)), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((n,), -math.log(max(n, 1)), device=cost.device, dtype=cost.dtype)
    f = torch.zeros(m, device=cost.device, dtype=cost.dtype)
    g = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    for _ in range(int(n_iters)):
        f = eps * (log_a - torch.logsumexp((g[None, :] - cost) / eps + log_b[None, :], dim=1))
        g = eps * (log_b - torch.logsumexp((f[:, None] - cost) / eps + log_a[:, None], dim=0))
    return torch.dot(f, log_a.exp()) + torch.dot(g, log_b.exp())


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    eps_scale: float,
    n_iters: int,
    w: torch.Tensor | None = None,
    normalize_cost: bool = True,
) -> torch.Tensor:
    with torch.no_grad():
        median_cost = torch.median(
            weighted_sqdist(x.detach(), y.detach(), w, normalize=normalize_cost)
        ).clamp_min(1e-6)
        eps = (float(eps_scale) * median_cost).clamp_min(1e-3)
    xy = sinkhorn_value(weighted_sqdist(x, y, w, normalize=normalize_cost), eps, n_iters)
    xx = sinkhorn_value(weighted_sqdist(x, x, w, normalize=normalize_cost), eps, n_iters)
    yy = sinkhorn_value(weighted_sqdist(y, y, w, normalize=normalize_cost), eps, n_iters)
    return xy - 0.5 * xx - 0.5 * yy


def endpoint_loss_kind(args) -> str:
    if args.endpoint_loss != "auto":
        return args.endpoint_loss
    return "mmd" if args.use_mmd else "none"


def compute_endpoint_loss(
    x1_hat: torch.Tensor,
    target: torch.Tensor,
    condition_delta: torch.Tensor | None,
    sample_genes: torch.Tensor,
    args,
) -> tuple[torch.Tensor, str]:
    kind = endpoint_loss_kind(args)
    if kind == "none":
        return x1_hat.new_tensor(0.0), kind

    x = x1_hat[:, sample_genes].float()
    y = target[:, sample_genes].float()
    w = None
    if kind in {"deg_mse", "deg_sinkhorn"}:
        w = deg_weights_from_condition_delta(condition_delta, sample_genes, args.deg_weight_epsilon)
        if w is None:
            kind = "mmd"

    if kind == "mmd":
        sigmas = median_sigmas(y)
        return mmd_multi_sigma(x, y, sigmas), kind
    if kind == "deg_mse":
        weighted_mse = (w[None, :] * (x - y).square()).mean()
        sigmas = median_sigmas(y, scales=(1.0, 2.0))
        return weighted_mse + float(args.deg_mse_mmd_weight) * mmd_multi_sigma(x, y, sigmas), kind
    if kind in {"sinkhorn", "deg_sinkhorn"}:
        loss = sinkhorn_divergence(
            x,
            y,
            args.sinkhorn_eps,
            args.sinkhorn_iters,
            w=w,
            normalize_cost=args.sinkhorn_normalize_cost,
        )
        if not torch.isfinite(loss):
            return x1_hat.new_tensor(0.0), f"{kind}_nonfinite"
        return loss, kind
    raise ValueError(f"Unsupported endpoint loss: {kind}")


def train_step(model, batch, args, device, step: int = 0):
    source = batch["source"].to(device=device, dtype=torch.float32)
    target = batch["target"].to(device=device, dtype=torch.float32)
    pert = batch["perturbation_id"].to(device)
    pert_gene = batch["perturbation_gene_id"].to(device)
    condition_delta = batch.get("condition_delta")
    if condition_delta is not None:
        condition_delta = condition_delta.to(device=device, dtype=torch.float32)
    b, g = source.shape
    sample_genes = torch.randperm(g, device=device)[: args.infer_top_genes]
    t = torch.rand(b, device=device)
    target_state = target - source if args.flow_target == "delta" else target
    noise_scale = delta_noise_scale_for_step(step, args) if args.flow_target == "delta" else None
    noise = make_flow_noise(source, args, delta_noise_scale=noise_scale)
    x_t = (1.0 - t[:, None]) * noise + t[:, None] * target_state
    dx = target_state - noise
    pred = model(x_t, source, t, pert, pert_gene)
    flow_loss = F.mse_loss(pred[:, sample_genes], dx[:, sample_genes])
    loss = flow_loss
    state_hat = x_t + pred * (1.0 - t[:, None])
    if args.flow_target == "delta":
        x1_hat = source + state_hat
        dir_pred = state_hat
        dir_true = target_state
    else:
        x1_hat = state_hat
        dir_pred = x1_hat - source
        dir_true = target - source
    rec_loss = F.mse_loss(x1_hat[:, sample_genes], target[:, sample_genes])
    bulk_loss = F.mse_loss(
        x1_hat[:, sample_genes].mean(dim=0),
        target[:, sample_genes].mean(dim=0),
    )
    bulk_mae_loss = F.l1_loss(
        x1_hat[:, sample_genes].mean(dim=0),
        target[:, sample_genes].mean(dim=0),
    )
    delta_bulk_weight = scheduled_weight(
        args.delta_bulk_weight,
        step,
        args.delta_bulk_warmup,
        args.delta_bulk_final_weight,
        args.loss_transition_start,
        args.loss_transition_steps,
    )
    delta_bulk_cos_weight = scheduled_weight(
        args.delta_bulk_cos_weight,
        step,
        args.delta_bulk_warmup,
        args.delta_bulk_cos_final_weight,
        args.loss_transition_start,
        args.loss_transition_steps,
    )
    delta_bulk_mae_weight = scheduled_weight(
        args.delta_bulk_mae_weight,
        step,
        args.delta_bulk_warmup,
        args.delta_bulk_mae_final_weight,
        args.loss_transition_start,
        args.loss_transition_steps,
    )
    source_anchor_weight = scheduled_weight(
        args.source_anchor_weight,
        step,
        args.source_anchor_warmup,
        args.source_anchor_final_weight,
        args.loss_transition_start,
        args.loss_transition_steps,
    )
    hetero_weight = scheduled_weight(
        args.hetero_weight,
        step,
        args.hetero_warmup,
        args.hetero_final_weight,
        args.loss_transition_start,
        args.loss_transition_steps,
    )
    recon_weight = scheduled_weight(
        args.recon_weight,
        step,
        args.recon_warmup,
        args.recon_final_weight,
        args.loss_transition_start,
        args.loss_transition_steps,
    )
    bulk_loss_weight = scheduled_weight(
        args.bulk_loss_weight,
        step,
        args.bulk_loss_warmup,
        args.bulk_loss_final_weight,
        args.loss_transition_start,
        args.loss_transition_steps,
    )
    bulk_mae_weight = scheduled_weight(
        args.bulk_mae_weight,
        step,
        args.bulk_mae_warmup,
        args.bulk_mae_final_weight,
        args.loss_transition_start,
        args.loss_transition_steps,
    )
    if condition_delta is not None and (delta_bulk_weight > 0 or delta_bulk_cos_weight > 0 or delta_bulk_mae_weight > 0):
        pred_bulk_delta = x1_hat[:, sample_genes].mean(dim=0) - source[:, sample_genes].mean(dim=0)
        target_bulk_delta = condition_delta[:, sample_genes].mean(dim=0)
        delta_bulk_loss = F.mse_loss(pred_bulk_delta, target_bulk_delta)
        delta_bulk_mae_loss = F.l1_loss(pred_bulk_delta, target_bulk_delta)
        delta_bulk_cos_loss = 1.0 - F.cosine_similarity(
            pred_bulk_delta[None],
            target_bulk_delta[None],
            dim=-1,
            eps=1e-8,
        ).mean()
        if delta_bulk_weight > 0:
            loss = loss + delta_bulk_weight * delta_bulk_loss
        if delta_bulk_cos_weight > 0:
            loss = loss + delta_bulk_cos_weight * delta_bulk_cos_loss
        if delta_bulk_mae_weight > 0:
            loss = loss + delta_bulk_mae_weight * delta_bulk_mae_loss
    else:
        delta_bulk_loss = loss.new_tensor(0.0)
        delta_bulk_cos_loss = loss.new_tensor(0.0)
        delta_bulk_mae_loss = loss.new_tensor(0.0)
    if condition_delta is not None and source_anchor_weight > 0:
        source_anchor = (source + condition_delta).clamp_min(0.0)
        source_anchor_loss = F.mse_loss(x1_hat[:, sample_genes], source_anchor[:, sample_genes])
        loss = loss + source_anchor_weight * source_anchor_loss
    else:
        source_anchor_loss = loss.new_tensor(0.0)
    if hetero_weight > 0:
        pred_resid = x1_hat[:, sample_genes] - x1_hat[:, sample_genes].mean(dim=0, keepdim=True)
        source_resid = source[:, sample_genes] - source[:, sample_genes].mean(dim=0, keepdim=True)
        hetero_loss = F.mse_loss(pred_resid, source_resid)
        loss = loss + hetero_weight * hetero_loss
    else:
        hetero_loss = loss.new_tensor(0.0)
    if args.dir_weight > 0:
        dir_loss = 1.0 - F.cosine_similarity(
            dir_pred[:, sample_genes],
            dir_true[:, sample_genes],
            dim=-1,
            eps=1e-8,
        ).mean()
        loss = loss + args.dir_weight * dir_loss
    else:
        dir_loss = loss.new_tensor(0.0)
    if recon_weight > 0:
        loss = loss + recon_weight * rec_loss
    if bulk_loss_weight > 0:
        loss = loss + bulk_loss_weight * bulk_loss
    if bulk_mae_weight > 0:
        loss = loss + bulk_mae_weight * bulk_mae_loss
    if args.action_aux_weight > 0:
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        action_aux = base_model.action_auxiliary_loss(pert, pert_gene)
        loss = loss + args.action_aux_weight * action_aux
    else:
        action_aux = loss.new_tensor(0.0)
    endpoint, endpoint_kind = compute_endpoint_loss(x1_hat, target, condition_delta, sample_genes, args)
    if endpoint_kind != "none":
        loss = loss + args.gamma * endpoint
    parts = {
        "flow": float(flow_loss.detach().item()),
        "dir": float(dir_loss.detach().item()),
        "hetero": float(hetero_loss.detach().item()),
        "action_aux": float(action_aux.detach().item()),
        "rec": float(rec_loss.detach().item()),
        "bulk": float(bulk_loss.detach().item()),
        "bulk_mae": float(bulk_mae_loss.detach().item()),
        "delta_bulk": float(delta_bulk_loss.detach().item()),
        "delta_cos": float(delta_bulk_cos_loss.detach().item()),
        "delta_bulk_mae": float(delta_bulk_mae_loss.detach().item()),
        "source_anchor": float(source_anchor_loss.detach().item()),
        "w_delta_bulk": delta_bulk_weight,
        "w_delta_bulk_mae": delta_bulk_mae_weight,
        "w_anchor": source_anchor_weight,
        "w_hetero": hetero_weight,
        "w_recon": recon_weight,
        "w_bulk": bulk_loss_weight,
        "w_bulk_mae": bulk_mae_weight,
        "endpoint": float(endpoint.detach().item()),
        "endpoint_kind": endpoint_kind,
        "mmd": float(endpoint.detach().item()) if endpoint_kind == "mmd" else 0.0,
        "noise": float(noise_scale) if noise_scale is not None else float("nan"),
    }
    return loss, parts














































def train_log_line(
    step: int,
    loss_value: float,
    elapsed: float,
    args,
    optimizer,
    loss_parts: dict | None = None,
    recent_speed: float | None = None,
) -> str:
    speed = (step + 1) / max(elapsed, 1e-9)
    remaining_steps = max(args.steps - step - 1, 0)
    eta_steps = remaining_steps / speed if speed > 0 else 0.0
    parts = [
        f"step {step}/{args.steps}",
        f"loss {loss_value:.6f}",
        f"speed {speed:.3f} step/s",
        f"recent {recent_speed:.3f} step/s" if recent_speed is not None else None,
        f"elapsed {format_duration(elapsed)}",
        f"eta {format_duration(eta_steps)}",
        f"lr {format_optimizer_lrs(optimizer)}",
    ]
    parts = [part for part in parts if part is not None]
    if loss_parts:
        parts.extend(
            [
                f"flow {loss_parts.get('flow', 0.0):.4f}",
                f"dir {loss_parts.get('dir', 0.0):.4f}",
                f"het {loss_parts.get('hetero', 0.0):.4f}",
                f"act {loss_parts.get('action_aux', 0.0):.4f}",
                f"rec {loss_parts.get('rec', 0.0):.4f}",
                f"bulk {loss_parts.get('bulk', 0.0):.4f}",
                f"bmae {loss_parts.get('bulk_mae', 0.0):.4f}",
                f"dbulk {loss_parts.get('delta_bulk', 0.0):.4f}",
                f"dbmae {loss_parts.get('delta_bulk_mae', 0.0):.4f}",
                f"dcos {loss_parts.get('delta_cos', 0.0):.4f}",
                f"anchor {loss_parts.get('source_anchor', 0.0):.4f}",
                f"w_db {loss_parts.get('w_delta_bulk', 0.0):.3f}",
                f"w_dbmae {loss_parts.get('w_delta_bulk_mae', 0.0):.3f}",
                f"w_anchor {loss_parts.get('w_anchor', 0.0):.3f}",
                f"w_het {loss_parts.get('w_hetero', 0.0):.3f}",
                f"w_rec {loss_parts.get('w_recon', 0.0):.3f}",
                f"w_bulk {loss_parts.get('w_bulk', 0.0):.3f}",
                f"w_bmae {loss_parts.get('w_bulk_mae', 0.0):.3f}",
            ]
        )
        if math.isfinite(loss_parts.get("noise", float("nan"))):
            parts.append(f"noise {loss_parts.get('noise', 0.0):.3f}")
        endpoint_kind = str(loss_parts.get("endpoint_kind", "none"))
        if endpoint_kind != "none":
            parts.append(f"endpoint {endpoint_kind}:{loss_parts.get('endpoint', 0.0):.4f}")
    if args.max_hours > 0:
        limit = args.max_hours * 3600.0
        parts.append(f"wall_left {format_duration(max(limit - elapsed, 0.0))}")
    return "[train] " + " | ".join(parts)




def model_state_dict(model):
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    return base.state_dict()


def save_checkpoint(path: Path, model, step: int, args) -> None:
    torch.save({"model": model_state_dict(model), "step": step, "args": vars(args)}, path)


def maybe_save_best_checkpoints(out_dir: Path, model, step: int, args, metrics: dict, best: dict) -> None:
    specs = [
        ("MAE", "min", "best_mae.pt"),
        ("L2", "min", "best_l2.pt"),
        ("Pearson_Delta", "max", "best_pearson_delta.pt"),
        ("DE-Spearman", "max", "best_de_spearman.pt"),
    ]
    values = {}
    for key, mode, filename in specs:
        try:
            value = float(metrics.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        old = best.get(key)
        improved = old is None or (value < old if mode == "min" else value > old)
        if improved:
            best[key] = value
            save_checkpoint(out_dir / filename, model, step, args)
            values[key] = value

    try:
        pearson = float(metrics.get("Pearson_Delta", float("nan")))
        spear = float(metrics.get("DE-Spearman", float("nan")))
        mae = float(metrics.get("MAE", float("nan")))
        src_hat = float(metrics.get("Pearson_Source_Hat", float("nan")))
    except (TypeError, ValueError):
        return
    if all(math.isfinite(v) for v in (pearson, spear, mae, src_hat)):
        # A compact screen score for this project: prioritize direction, then calibration,
        # while lightly rewarding source residual preservation.
        balanced = pearson + spear + 0.25 * src_hat - mae
        old = best.get("balanced_score")
        if old is None or balanced > old:
            best["balanced_score"] = balanced
            save_checkpoint(out_dir / "best_balanced.pt", model, step, args)
            values["balanced_score"] = balanced

    if values:
        summary = " | ".join(f"{k} {v:.4f}" for k, v in values.items())
        print(f"[best] step {step} | {summary}", flush=True)


def load_model_weights(model, checkpoint_path: str, device) -> int | None:
    if not checkpoint_path:
        return None
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(str(k).startswith("_orig_mod.") for k in state.keys()):
        state = {str(k).removeprefix("_orig_mod."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[init] loaded {checkpoint_path} with missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    else:
        print(f"[init] loaded {checkpoint_path}", flush=True)
    if isinstance(ckpt, dict):
        return ckpt.get("step")
    return None


def load_data(args):
    path = Path(args.data_path)
    if args.dataset == "norman":
        return prepare_norman(path, args.n_top_genes, args.split, args.fold, args.seed, args.store_dtype)
    if args.dataset == "combosciplex":
        return prepare_combosciplex(path, args.n_top_genes, args.combosciplex_test_conditions, args.store_dtype)
    raise ValueError(args.dataset)


def train(args):
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    data = load_data(args)
    train_mask = (data.modes == "train") | data.is_control
    neighbor_cache = out_dir / f"geometry_top{args.graph_k}_m{args.manifold_dim}.pt"
    train_rows = np.where(train_mask)[0]
    geometry = build_coexpression_geometry(
        data.x,
        args.graph_k,
        args.manifold_dim,
        neighbor_cache,
        row_idx=train_rows,
        max_cells=args.graph_cells,
        seed=args.seed,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = VoidCellModel(
        n_genes=data.x.shape[1],
        n_perturbations=len(data.perturbation_names),
        neighbors=geometry["neighbors"],
        edge_weights=geometry["weights"],
        manifold_coords=geometry["coords"],
        dim=args.dim,
        hidden=args.hidden,
        encode_blocks=args.encode_blocks,
        think_steps=args.think_steps,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        neighbor_chunk=args.neighbor_chunk,
        neighbor_gate=args.neighbor_gate,
        directional_shifts=args.directional_shifts,
        directional_residual_gate=args.directional_residual_gate,
        directional_gate_init=args.directional_gate_init,
        shift_dims=args.shift_dims,
        shift_stencil=args.shift_stencil,
        shift_temperature=args.shift_temperature,
        shift_code_strength=args.shift_code_strength,
        spatial_grid_shifts=args.spatial_grid_shifts,
        spatial_grid_dims=args.spatial_grid_dims,
        spatial_grid_side=args.spatial_grid_side,
        spatial_shift_stencil=args.spatial_shift_stencil,
        spatial_shift_code_strength=args.spatial_shift_code_strength,
        graph_message_weight=args.graph_message_weight,
        spatial_message_weight=args.spatial_message_weight,
        self_weight=args.self_weight,
        neighbor_weight=args.neighbor_weight,
        global_weight=args.global_weight,
        source_memory=args.source_memory,
        source_weight=args.source_weight,
        action_field=args.action_field,
        action_strength=args.action_strength,
        action_init_std=args.action_init_std,
        share_pert_gene_embedding=args.share_pert_gene_embedding,
        dynamic_edge_gate=args.dynamic_edge_gate,
        edge_gate_init=args.edge_gate_init,
        checkpoint_blocks=args.checkpoint_blocks,
    ).to(device)
    init_step = load_model_weights(model, args.init_from, device)
    if init_step is not None:
        print(f"[init] checkpoint_step={init_step}; starting new training schedule", flush=True)
    eval_model = model
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
        if args.compile_eval:
            eval_model = model
    base_model = eval_model._orig_mod if hasattr(eval_model, "_orig_mod") else eval_model
    extra = ""
    if getattr(base_model, "spatial_grid_shifts", False):
        extra = f" spatial_grid={base_model.grid_shape} spatial_dirs={base_model.n_spatial_dirs}"
    if getattr(base_model, "directional_residual_gate", False):
        gate = torch.sigmoid(base_model.directional_gate_logit.detach()).cpu().item()
        extra += f" directional_gate={gate:.4f}"
    extra += (
        f" source_memory={getattr(base_model, 'source_memory', 'none')}"
        f" action_field={getattr(base_model, 'action_field', 'none')}"
    )
    if getattr(base_model, "share_pert_gene_embedding", False):
        extra += " shared_pert_gene_emb=on"
    if getattr(base_model, "dynamic_edge_gate", False):
        extra += " dynamic_edge_gate=on"
    print(
        f"genes={data.x.shape[1]} perturbations={len(data.perturbation_names)} "
        f"params={sum(p.numel() for p in model.parameters())}{extra}"
    )
    if args.eval_only:
        eval_step = int(init_step) if init_step is not None else 0
        eval_t0 = time.time()
        metrics = evaluate(eval_model, data, args, device, out_dir=out_dir, step=eval_step)
        if metrics:
            metrics["Step"] = eval_step
            metrics["Elapsed"] = format_duration(0.0)
            write_metrics(out_dir / "metrics.csv", metrics)
            print(metrics_log_line(metrics), flush=True)
        print(f"[eval] wall_time {format_duration(time.time() - eval_t0)}", flush=True)
        return

    ds = PerturbationBatchDataset(
        data,
        args.batch_size,
        mixed_conditions=args.mixed_condition_batch,
        condition_cycle=args.condition_cycle,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: lr_multiplier(s, args))
    use_amp = device.type == "cuda" and args.precision != "fp32"
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.precision == "fp16")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.precision == "fp16")

    step = 0
    t0 = time.time()
    last_log_time = t0
    last_log_step = 0
    best_metrics = {}
    while step < args.steps:
        for batch in loader:
            batch = {k: v.squeeze(0) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, loss_parts = train_step(model, batch, args, device, step)
            if not torch.isfinite(loss):
                bad_param = None
                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                for name, param in base_model.named_parameters():
                    if not torch.isfinite(param).all():
                        bad_param = name
                        break
                print(
                    f"[train] non-finite loss at step {step}; "
                    f"bad_param={bad_param}; "
                    f"parts={json.dumps(loss_parts, sort_keys=True)}",
                    flush=True,
                )
                save_checkpoint(out_dir / "nonfinite.pt", model, step, args)
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            if scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= old_scale
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer_stepped = True
            if optimizer_stepped:
                scheduler.step()
            if step % args.log_every == 0:
                now = time.time()
                elapsed = max(now - t0, 1e-9)
                recent_steps = max(step - last_log_step, 1)
                recent_elapsed = max(now - last_log_time, 1e-9)
                recent_speed = recent_steps / recent_elapsed
                print(
                    train_log_line(
                        step,
                        loss.item(),
                        elapsed,
                        args,
                        optimizer,
                        loss_parts,
                        recent_speed=recent_speed,
                    ),
                    flush=True,
                )
                last_log_time = now
                last_log_step = step
            if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
                save_checkpoint(out_dir / "last.pt", model, step, args)
                eval_t0 = time.time()
                try:
                    metrics = evaluate(eval_model, data, args, device, out_dir=out_dir, step=step)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    print(f"[eval] skipped at step {step}: CUDA out of memory during eval", flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    metrics = None
                if metrics:
                    metrics["Step"] = step
                    metrics["Elapsed"] = format_duration(time.time() - t0)
                    write_metrics(out_dir / "metrics.csv", metrics)
                    print(metrics_log_line(metrics), flush=True)
                    maybe_save_best_checkpoints(out_dir, model, step, args, metrics, best_metrics)
                eval_elapsed = time.time() - eval_t0
                print(f"[eval] wall_time {format_duration(eval_elapsed)}", flush=True)
                last_log_time = time.time()
                last_log_step = step
                save_checkpoint(out_dir / "last.pt", model, step, args)
            step += 1
            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / "last.pt", model, step, args)
            if args.max_hours > 0 and (time.time() - t0) >= args.max_hours * 3600.0:
                save_checkpoint(out_dir / "last.pt", model, step, args)
                save_checkpoint(out_dir / "final.pt", model, step, args)
                print(f"Reached max_hours={args.max_hours:.2f}; saved checkpoint at step={step}.")
                return
            if step >= args.steps:
                break
    save_checkpoint(out_dir / "final.pt", model, step, args)


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if g.numel() == 0:
        return g
    original_dtype = g.dtype
    use_bf16 = g.is_cuda and torch.cuda.is_bf16_supported()
    x = g.bfloat16() if use_bf16 else g.float()
    norm = x.norm()
    if float(norm) < eps:
        return torch.zeros_like(g)
    x = x / (norm + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(int(steps), 1)):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x

    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


def polar_quintic(
    g: torch.Tensor,
    steps: int = 12,
    eps: float = 1e-7,
    compute_dtype: str = "float32",
) -> torch.Tensor:
    """Approximate the matrix polar factor with the Aurora simple-quintic map."""
    if g.numel() == 0:
        return g
    original_dtype = g.dtype
    if compute_dtype == "bf16":
        x = g.bfloat16()
    else:
        x = g.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = 2.0, -1.5, 0.5
    for _ in range(max(int(steps), 1)):
        aa = x @ x.mT
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.mT
    if compute_dtype == "bf16":
        return x
    return x.to(dtype=original_dtype)


class HybridMuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_aurora", False):
                self._aurora_step(group)
            elif group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group):
        lr = group["lr"]
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]
        ns_steps = group["ns_steps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients.")
            if weight_decay != 0:
                p.mul_(1.0 - lr * weight_decay)
            grad = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)

            matrix = buf.reshape(buf.size(0), -1)
            update = zeropower_via_newtonschulz5(matrix, steps=ns_steps)
            rows, cols = matrix.shape
            scale = math.sqrt(max(1.0, rows / max(cols, 1)))
            p.add_(update.reshape_as(p), alpha=-lr * scale)

    def _aurora_update(self, update: torch.Tensor, group) -> torch.Tensor:
        rows, cols = update.shape
        original_rows, original_cols = rows, cols
        polar_steps = group["polar_steps"]
        eps = group["eps"]
        compute_dtype = group["compute_dtype"]
        if rows == cols:
            out = polar_quintic(update, steps=polar_steps, eps=eps, compute_dtype=compute_dtype)
        else:
            transposed = rows < cols
            if transposed:
                update = update.T
                rows, cols = update.shape

            g32 = update.float()
            target_row_sq = float(cols) / float(rows)
            row_norm = g32.norm(dim=-1, keepdim=True).clamp_min(eps)
            d = 1.0 / row_norm
            pp_iterations = max(int(group["pp_iterations"]), 1)
            pp_beta = float(group["pp_beta"])
            out = None
            for pp_idx in range(pp_iterations):
                out = polar_quintic(d * g32, steps=polar_steps, eps=eps, compute_dtype=compute_dtype)
                if pp_idx < pp_iterations - 1:
                    row_sq = out.float().pow(2).sum(dim=-1, keepdim=True).clamp_min(eps * eps)
                    d = d * (target_row_sq / row_sq).pow(pp_beta)
            if transposed:
                out = out.T

        scale = math.sqrt(max(1.0, original_rows / max(original_cols, 1)))
        return out * scale

    def _aurora_step(self, group):
        lr = group["lr"]
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]
        nesterov = group["nesterov"]
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("Aurora does not support sparse gradients.")
            if p.ndim != 2:
                raise RuntimeError(f"Aurora expects 2D parameters, got shape {tuple(p.shape)}.")
            grad = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.lerp_(grad, 1.0 - momentum)
            update_src = grad.lerp(buf, momentum) if nesterov else buf
            update = self._aurora_update(update_src.reshape(update_src.size(0), -1), group)
            if not torch.isfinite(update).all():
                raise RuntimeError(f"Aurora produced non-finite update for parameter {tuple(p.shape)}.")
            if weight_decay != 0:
                p.mul_(1.0 - lr * weight_decay)
            p.add_(update.reshape_as(p), alpha=-lr)

    def _adamw_step(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients.")
            if weight_decay != 0:
                p.mul_(1.0 - lr * weight_decay)
            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1
            step = state["step"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

def use_muon_for_parameter(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim < 2:
        return False
    lowered = name.lower()
    adamw_only = (
        "embedding",
        "velocity",
        "out_norm",
        "norm",
        "bias",
        # Dense-field bridge maps are very large rectangular input/output
        # projections. Orthogonalized Muon/Aurora updates are too aggressive
        # for them early in training and can NaN the cancer-style backbone.
        "expression_to_field",
        "target_action_to_field",
        "drug_action_to_field",
        "field_to_gene",
        "field_readout",
    )
    return not any(token in lowered for token in adamw_only)

def build_optimizer(model, args):
    betas = (args.adam_beta1, args.adam_beta2)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=betas,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw_fused":
        try:
            return torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=betas,
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
                fused=torch.cuda.is_available(),
            )
        except TypeError:
            return torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=betas,
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
    if args.optimizer not in {"muon", "aurora"}:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if use_muon_for_parameter(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    groups = []
    if muon_params:
        if args.optimizer == "aurora":
            groups.append(
                {
                    "params": muon_params,
                    "lr": args.muon_lr,
                    "momentum": args.muon_momentum,
                    "weight_decay": args.weight_decay,
                    "polar_steps": args.aurora_polar_steps,
                    "pp_iterations": args.aurora_pp_iterations,
                    "pp_beta": args.aurora_pp_beta,
                    "eps": args.aurora_eps,
                    "compute_dtype": args.aurora_compute_dtype,
                    "nesterov": args.aurora_nesterov,
                    "use_aurora": True,
                    "name": "aurora",
                }
            )
        else:
            groups.append(
                {
                    "params": muon_params,
                    "lr": args.muon_lr,
                    "momentum": args.muon_momentum,
                    "weight_decay": args.weight_decay,
                    "ns_steps": args.muon_ns_steps,
                    "use_muon": True,
                    "name": "muon",
                }
            )
    if adamw_params:
        groups.append(
            {
                "params": adamw_params,
                "lr": args.lr,
                "betas": betas,
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
                "use_muon": False,
                "name": "adamw",
            }
        )
    return HybridMuonAdamW(groups)

def format_optimizer_lrs(optimizer) -> str:
    if len(optimizer.param_groups) == 1:
        return f"{optimizer.param_groups[0]['lr']:.2e}"
    return ",".join(
        f"{group.get('name', i)}={group['lr']:.2e}"
        for i, group in enumerate(optimizer.param_groups)
    )

def lr_multiplier(step: int, args) -> float:
    if args.lr_schedule == "none":
        return 1.0
    warmup = max(int(args.warmup_steps), 0)
    if warmup > 0 and step < warmup:
        return max((step + 1) / float(warmup), 1e-3)
    denom = max(args.steps - warmup, 1)
    progress = min(max((step - warmup) / float(denom), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["norman", "combosciplex"], default="norman")
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-from", default="")
    p.add_argument("--split", choices=["additive", "combinations", "unseen"], default="additive")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--n-top-genes", type=int, default=5000)
    p.add_argument("--infer-top-genes", type=int, default=1000)
    p.add_argument("--store-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--combosciplex-test-conditions", nargs="*", default=None)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adamw", "adamw_fused", "muon", "aurora"], default="adamw")
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument("--aurora-polar-steps", type=int, default=12)
    p.add_argument("--aurora-pp-iterations", type=int, default=2)
    p.add_argument("--aurora-pp-beta", type=float, default=0.5)
    p.add_argument("--aurora-eps", type=float, default=1e-7)
    p.add_argument("--aurora-compute-dtype", choices=["float32", "bf16"], default="float32")
    p.add_argument("--aurora-nesterov", dest="aurora_nesterov", action="store_true", default=True)
    p.add_argument("--no-aurora-nesterov", dest="aurora_nesterov", action="store_false")
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--encode-blocks", type=int, default=4)
    p.add_argument("--think-steps", type=int, default=8)
    p.add_argument("--graph-k", type=int, default=30)
    p.add_argument("--graph-cells", type=int, default=8192)
    p.add_argument("--manifold-dim", type=int, default=4)
    p.add_argument("--neighbor-chunk", type=int, default=256)
    p.add_argument("--neighbor-gate", action="store_true")
    p.add_argument("--directional-shifts", action="store_true")
    p.add_argument("--directional-residual-gate", action="store_true")
    p.add_argument("--directional-gate-init", type=float, default=-4.0)
    p.add_argument("--shift-dims", type=int, default=0)
    p.add_argument("--shift-stencil", choices=["axis", "cube"], default="axis")
    p.add_argument("--shift-temperature", type=float, default=4.0)
    p.add_argument("--shift-code-strength", type=float, default=1.0)
    p.add_argument("--spatial-grid-shifts", action="store_true")
    p.add_argument("--spatial-grid-dims", type=int, default=3)
    p.add_argument("--spatial-grid-side", type=int, default=0)
    p.add_argument("--spatial-shift-stencil", choices=["axis", "cube"], default="cube")
    p.add_argument("--spatial-shift-code-strength", type=float, default=1.0)
    p.add_argument("--graph-message-weight", type=float, default=1.0)
    p.add_argument("--spatial-message-weight", type=float, default=1.0)
    p.add_argument("--self-weight", type=float, default=1.0)
    p.add_argument("--neighbor-weight", type=float, default=1.0)
    p.add_argument("--global-weight", type=float, default=1.0)
    p.add_argument("--source-memory", choices=["none", "static"], default="static")
    p.add_argument("--source-weight", type=float, default=1.0)
    p.add_argument("--action-field", choices=["none", "target_gene", "drug_manifold"], default="target_gene")
    p.add_argument("--action-strength", type=float, default=1.0)
    p.add_argument("--action-init-std", type=float, default=1e-4)
    p.add_argument("--share-pert-gene-embedding", action="store_true")
    p.add_argument("--dynamic-edge-gate", action="store_true")
    p.add_argument("--edge-gate-init", type=float, default=2.0)
    p.add_argument("--checkpoint-blocks", action="store_true")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--residual-scale", type=float, default=0.1)
    p.add_argument("--noise", choices=["gaussian", "poisson"], default="gaussian")
    p.add_argument("--flow-target", choices=["cell", "delta"], default="cell")
    p.add_argument("--delta-noise-scale", type=float, default=1.0)
    p.add_argument("--delta-noise-start", type=float, default=0.25)
    p.add_argument("--delta-noise-end", type=float, default=1.0)
    p.add_argument("--delta-noise-warmup", type=int, default=2000)
    p.add_argument("--eval-delta-noise-scale", type=float, default=None)
    p.add_argument("--use-mmd", action="store_true")
    p.add_argument("--endpoint-loss", choices=["auto", "none", "mmd", "sinkhorn", "deg_mse", "deg_sinkhorn"], default="auto")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--deg-weight-epsilon", type=float, default=0.1)
    p.add_argument("--deg-mse-mmd-weight", type=float, default=0.1)
    p.add_argument("--sinkhorn-eps", type=float, default=0.1)
    p.add_argument("--sinkhorn-iters", type=int, default=50)
    p.add_argument("--sinkhorn-normalize-cost", dest="sinkhorn_normalize_cost", action="store_true", default=True)
    p.add_argument("--no-sinkhorn-normalize-cost", dest="sinkhorn_normalize_cost", action="store_false")
    p.add_argument("--dir-weight", type=float, default=0.0)
    p.add_argument("--hetero-weight", type=float, default=0.0)
    p.add_argument("--action-aux-weight", type=float, default=0.01)
    p.add_argument("--recon-weight", type=float, default=0.5)
    p.add_argument("--recon-warmup", type=int, default=0)
    p.add_argument("--recon-final-weight", type=float, default=None)
    p.add_argument("--bulk-loss-weight", type=float, default=2.0)
    p.add_argument("--bulk-loss-warmup", type=int, default=0)
    p.add_argument("--bulk-loss-final-weight", type=float, default=None)
    p.add_argument("--bulk-mae-weight", type=float, default=0.0)
    p.add_argument("--bulk-mae-warmup", type=int, default=0)
    p.add_argument("--bulk-mae-final-weight", type=float, default=None)
    p.add_argument("--delta-bulk-weight", type=float, default=0.0)
    p.add_argument("--delta-bulk-cos-weight", type=float, default=0.0)
    p.add_argument("--delta-bulk-mae-weight", type=float, default=0.0)
    p.add_argument("--delta-bulk-warmup", type=int, default=0)
    p.add_argument("--delta-bulk-final-weight", type=float, default=None)
    p.add_argument("--delta-bulk-cos-final-weight", type=float, default=None)
    p.add_argument("--delta-bulk-mae-final-weight", type=float, default=None)
    p.add_argument("--source-anchor-weight", type=float, default=0.0)
    p.add_argument("--source-anchor-warmup", type=int, default=0)
    p.add_argument("--source-anchor-final-weight", type=float, default=None)
    p.add_argument("--hetero-warmup", type=int, default=0)
    p.add_argument("--hetero-final-weight", type=float, default=None)
    p.add_argument("--loss-transition-start", type=int, default=0)
    p.add_argument("--loss-transition-steps", type=int, default=0)
    p.add_argument("--mixed-condition-batch", action="store_true")
    p.add_argument("--condition-cycle", action="store_true")
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--ode-method", choices=["euler", "heun", "rk4"], default="euler")
    p.add_argument("--eval-top-genes", type=int, default=1000)
    p.add_argument("--eval-gene-selection", choices=["mean", "hvg_test", "hvg_train"], default="mean")
    p.add_argument("--eval-cells", type=int, default=128)
    p.add_argument("--eval-seed", type=int, default=None)
    p.add_argument("--eval-control-cells", type=int, default=0)
    p.add_argument("--eval-cell-noise", choices=["train", "source", "poisson", "gaussian"], default="train")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-progress", action="store_true")
    p.add_argument("--eval-max-conditions", type=int, default=0)
    p.add_argument("--cell-eval", dest="cell_eval", action="store_true", default=True)
    p.add_argument("--no-cell-eval", dest="cell_eval", action="store_false")
    p.add_argument("--cell-eval-threads", type=int, default=4)
    p.add_argument("--cell-eval-batch-size", type=int, default=100)
    p.add_argument("--cell-eval-profile", choices=["full", "minimal", "de", "anndata", "vcc"], default="full")
    p.add_argument("--cell-eval-skip-metrics", default=DEFAULT_CELL_EVAL_SKIP)
    p.add_argument("--official-eval-no-skip", action="store_true")
    p.add_argument("--save-eval-anndata", action="store_true")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-eval", action="store_true")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
