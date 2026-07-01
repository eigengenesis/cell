#!/usr/bin/env python3
"""
Standalone VOID Cell trainer.

No scDFM imports and no gene tokenizer. Genes are direct AnnData column indices.
"""

from __future__ import annotations

from data import COMBOSCIPLEX_DEFAULT_TEST, DEFAULT_CELL_EVAL_SKIP, seed_everything, PerturbationBatchDataset, prepare_norman, prepare_combosciplex
from eval import evaluate, make_flow_noise, format_duration, write_metrics, metrics_log_line, median_sigmas, mmd_multi_sigma

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









































def train_step(model, batch, args, device):
    source = batch["source"].to(device=device, dtype=torch.float32)
    target = batch["target"].to(device=device, dtype=torch.float32)
    pert = batch["perturbation_id"].to(device)
    pert_gene = batch["perturbation_gene_id"].to(device)
    b, g = source.shape
    sample_genes = torch.randperm(g, device=device)[: args.infer_top_genes]
    t = torch.rand(b, device=device)
    target_state = target - source if args.flow_target == "delta" else target
    noise = make_flow_noise(source, args)
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
    if args.hetero_weight > 0:
        pred_resid = x1_hat[:, sample_genes] - x1_hat[:, sample_genes].mean(dim=0, keepdim=True)
        source_resid = source[:, sample_genes] - source[:, sample_genes].mean(dim=0, keepdim=True)
        hetero_loss = F.mse_loss(pred_resid, source_resid)
        loss = loss + args.hetero_weight * hetero_loss
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
    if args.recon_weight > 0:
        loss = loss + args.recon_weight * rec_loss
    if args.bulk_loss_weight > 0:
        loss = loss + args.bulk_loss_weight * bulk_loss
    if args.use_mmd:
        sigmas = median_sigmas(target[:, sample_genes])
        mmd = mmd_multi_sigma(x1_hat[:, sample_genes], target[:, sample_genes], sigmas)
        loss = loss + args.gamma * mmd
    else:
        mmd = loss.new_tensor(0.0)
    parts = {
        "flow": float(flow_loss.detach().item()),
        "dir": float(dir_loss.detach().item()),
        "hetero": float(hetero_loss.detach().item()),
        "rec": float(rec_loss.detach().item()),
        "bulk": float(bulk_loss.detach().item()),
        "mmd": float(mmd.detach().item()),
    }
    return loss, parts














































def train_log_line(step: int, loss_value: float, elapsed: float, args, optimizer, loss_parts: dict | None = None) -> str:
    speed = (step + 1) / max(elapsed, 1e-9)
    remaining_steps = max(args.steps - step - 1, 0)
    eta_steps = remaining_steps / speed if speed > 0 else 0.0
    parts = [
        f"step {step}/{args.steps}",
        f"loss {loss_value:.6f}",
        f"speed {speed:.3f} step/s",
        f"elapsed {format_duration(elapsed)}",
        f"eta {format_duration(eta_steps)}",
        f"lr {format_optimizer_lrs(optimizer)}",
    ]
    if loss_parts:
        parts.extend(
            [
                f"flow {loss_parts.get('flow', 0.0):.4f}",
                f"dir {loss_parts.get('dir', 0.0):.4f}",
                f"het {loss_parts.get('hetero', 0.0):.4f}",
                f"rec {loss_parts.get('rec', 0.0):.4f}",
                f"bulk {loss_parts.get('bulk', 0.0):.4f}",
            ]
        )
        if args.use_mmd:
            parts.append(f"mmd {loss_parts.get('mmd', 0.0):.4f}")
    if args.max_hours > 0:
        limit = args.max_hours * 3600.0
        parts.append(f"wall_left {format_duration(max(limit - elapsed, 0.0))}")
    return "[train] " + " | ".join(parts)




def model_state_dict(model):
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    return base.state_dict()


def save_checkpoint(path: Path, model, step: int, args) -> None:
    torch.save({"model": model_state_dict(model), "step": step, "args": vars(args)}, path)


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
        checkpoint_blocks=args.checkpoint_blocks,
    ).to(device)
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
    print(
        f"genes={data.x.shape[1]} perturbations={len(data.perturbation_names)} "
        f"params={sum(p.numel() for p in model.parameters())}{extra}"
    )

    ds = PerturbationBatchDataset(data, args.batch_size, mixed_conditions=args.mixed_condition_batch)
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
    while step < args.steps:
        for batch in loader:
            batch = {k: v.squeeze(0) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, loss_parts = train_step(model, batch, args, device)
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
                elapsed = max(time.time() - t0, 1e-9)
                print(train_log_line(step, loss.item(), elapsed, args, optimizer, loss_parts), flush=True)
            if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
                save_checkpoint(out_dir / "last.pt", model, step, args)
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
            if group.get("use_muon", False):
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
    adamw_only = ("embedding", "velocity", "out_norm", "norm", "bias")
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
    if args.optimizer != "muon":
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
    p.add_argument("--split", choices=["additive", "combinations", "unseen"], default="additive")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--n-top-genes", type=int, default=5000)
    p.add_argument("--infer-top-genes", type=int, default=1000)
    p.add_argument("--store-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--combosciplex-test-conditions", nargs="*", default=None)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adamw", "adamw_fused", "muon"], default="adamw")
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--muon-ns-steps", type=int, default=5)
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
    p.add_argument("--checkpoint-blocks", action="store_true")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--residual-scale", type=float, default=0.1)
    p.add_argument("--noise", choices=["gaussian", "poisson"], default="gaussian")
    p.add_argument("--flow-target", choices=["cell", "delta"], default="cell")
    p.add_argument("--delta-noise-scale", type=float, default=1.0)
    p.add_argument("--eval-delta-noise-scale", type=float, default=None)
    p.add_argument("--use-mmd", action="store_true")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--dir-weight", type=float, default=0.0)
    p.add_argument("--hetero-weight", type=float, default=0.0)
    p.add_argument("--recon-weight", type=float, default=0.5)
    p.add_argument("--bulk-loss-weight", type=float, default=2.0)
    p.add_argument("--mixed-condition-batch", action="store_true")
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--eval-top-genes", type=int, default=1000)
    p.add_argument("--eval-gene-selection", choices=["mean", "hvg_test", "hvg_train"], default="mean")
    p.add_argument("--eval-cells", type=int, default=128)
    p.add_argument("--eval-seed", type=int, default=None)
    p.add_argument("--eval-control-cells", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-progress", action="store_true")
    p.add_argument("--eval-max-conditions", type=int, default=0)
    p.add_argument("--cell-eval", dest="cell_eval", action="store_true", default=True)
    p.add_argument("--no-cell-eval", dest="cell_eval", action="store_false")
    p.add_argument("--cell-eval-threads", type=int, default=4)
    p.add_argument("--cell-eval-batch-size", type=int, default=100)
    p.add_argument("--cell-eval-profile", choices=["full", "minimal", "de", "anndata", "vcc"], default="full")
    p.add_argument("--cell-eval-skip-metrics", default=DEFAULT_CELL_EVAL_SKIP)
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
