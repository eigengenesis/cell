import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import torch
import torch.nn as nn
import tyro
from config.config_flow import FlowConfig as Config
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from src.data_process.data import Data, PerturbationDataset
from src.flow_matching.ot import OTPlanSampler
from src.flow_matching.path import AffineProbPath
from src.models.instantiate_model import instantiate_model
import tqdm
from src.flow_matching.path.scheduler import CondOTScheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
import torchdiffeq
from tqdm import trange
import numpy as np
from cell_eval import MetricsEvaluator
import anndata as ad
import pandas as pd
import math
import re
from src.utils.utils import (save_checkpoint, load_checkpoint, make_lognorm_poisson_noise,
                              pick_eval_score, process_vocab, set_requires_grad_for_p_only)
from src.models.origin.layers import build_void_geometry

ot_sampler = OTPlanSampler(method="exact")
path = AffineProbPath(scheduler=CondOTScheduler())


def autocast_ctx():
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                          enabled=(device.type == 'cuda'))


def build_grn_neighbor_mask(edge_list, gene_name_to_token, ntoken, k):
    import collections
    nbrs = collections.defaultdict(list)
    for e in edge_list:
        a, b = e[0], e[1]
        if a in gene_name_to_token and b in gene_name_to_token:
            ta, tb = gene_name_to_token[a], gene_name_to_token[b]
            nbrs[ta].append(tb)
            nbrs[tb].append(ta)
    mask = torch.full((ntoken, k), -1, dtype=torch.long)
    for tok, lst in nbrs.items():
        lst = list(dict.fromkeys(lst))[:k]
        if lst:
            mask[tok, :len(lst)] = torch.tensor(lst, dtype=torch.long)
    return mask


def pairwise_sq_dists(X, Y):
    return torch.cdist(X, Y, p=2) ** 2


@torch.no_grad()
def median_sigmas(X, scales=(0.5, 1.0, 2.0, 4.0)):
    D2 = pairwise_sq_dists(X, X)
    tri = D2[~torch.eye(D2.size(0), dtype=bool, device=D2.device)]
    m = torch.median(tri).clamp_min(1e-12)
    s2 = torch.tensor(scales, device=X.device) * m
    return [float(s.item()) for s in torch.sqrt(s2)]


def mmd2_unbiased_multi_sigma(X, Y, sigmas):
    m, n = X.size(0), Y.size(0)
    Dxx = pairwise_sq_dists(X, X)
    Dyy = pairwise_sq_dists(Y, Y)
    Dxy = pairwise_sq_dists(X, Y)
    vals = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * (sigma ** 2) + 1e-12)
        Kxx = torch.exp(-beta * Dxx)
        Kyy = torch.exp(-beta * Dyy)
        Kxy = torch.exp(-beta * Dxy)
        term_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
        term_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
        term_xy = Kxy.mean()
        vals.append(term_xx + term_yy - 2.0 * term_xy)
    return torch.stack(vals).mean()


def compute_deg_weights(train_sampler, epsilon=0.1):
    adata = train_sampler.adata
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    cond = adata.obs['perturbation_covariates'].astype(str).values
    pair_ids = np.asarray(train_sampler.perturbation_covariates_id)
    is_ctrl = cond == 'control+control'
    if is_ctrl.sum() == 0:
        is_ctrl = np.array(['control' in c.lower() for c in cond])
    ctrl_mean = X[is_ctrl].mean(0) if is_ctrl.sum() > 0 else X.mean(0)
    string_to_pair = {}
    for i, name in enumerate(cond):
        if name not in string_to_pair:
            string_to_pair[name] = tuple(sorted(int(v) for v in pair_ids[i]))
    weights = {}
    for name in np.unique(cond):
        if name == 'control+control':
            continue
        mask = cond == name
        if mask.sum() == 0:
            continue
        lfc = np.abs(X[mask].mean(0) - ctrl_mean)
        w_max = max(float(lfc.max()), 1e-12)
        w = epsilon + (1.0 - epsilon) * (lfc / w_max)
        weights[string_to_pair[name]] = torch.from_numpy(w.astype(np.float32))
    return weights


def weighted_sqdist(X, Y, w=None):
    if w is None:
        return torch.cdist(X, Y, p=2) ** 2
    s = w.clamp_min(0).sqrt()
    return torch.cdist(X * s, Y * s, p=2) ** 2


def sinkhorn_value(C, eps, n_iters):
    m, n = C.shape
    log_a = torch.full((m,), -math.log(m), device=C.device, dtype=C.dtype)
    log_b = torch.full((n,), -math.log(n), device=C.device, dtype=C.dtype)
    f = torch.zeros(m, device=C.device, dtype=C.dtype)
    g = torch.zeros(n, device=C.device, dtype=C.dtype)
    for _ in range(n_iters):
        f = eps * (log_a - torch.logsumexp((g[None, :] - C) / eps + log_b[None, :], dim=1))
        g = eps * (log_b - torch.logsumexp((f[:, None] - C) / eps + log_a[:, None], dim=0))
    return torch.dot(f, log_a.exp()) + torch.dot(g, log_b.exp())


def sinkhorn_divergence(X, Y, eps, n_iters, w=None):
    return (sinkhorn_value(weighted_sqdist(X, Y, w), eps, n_iters)
            - 0.5 * sinkhorn_value(weighted_sqdist(X, X, w), eps, n_iters)
            - 0.5 * sinkhorn_value(weighted_sqdist(Y, Y, w), eps, n_iters))


def perturbation_gene_ids_from_tokens(perturbation_id, panel_pos, device):
    rows = perturbation_id.detach().cpu().tolist()
    mapped = [[panel_pos.get(int(tok), -1) for tok in row] for row in rows]
    return torch.tensor(mapped, dtype=torch.long, device=device)


def train_step(source, target, perturbation_id, vf, accelerator,
               noise_type='Poisson', mode="predict_y", cfg_dropout=0.0,
               deg_weights=None, condition_idx=None, perturbation_gene_id=None):
    B = source.shape[0]
    device = accelerator.device
    is_void = config.fusion_method == 'void'
    flow_target = getattr(config, 'flow_target', 'cell')

    gene = gene_ids.repeat(B, 1).to(device)
    if is_void:
        gene_input = gene
        sample_genes = None
        loss_gene_index = torch.arange(source.shape[-1], device=device)
    else:
        input_gene_ids = torch.randperm(source.shape[-1], device=device)[:config.infer_top_gene]
        source = source[:, input_gene_ids]
        target = target[:, input_gene_ids]
        gene_input = gene[:, input_gene_ids]
        sample_genes = None
        loss_gene_index = input_gene_ids

    def sl(a):
        return a[:, sample_genes] if sample_genes is not None else a

    if mode == "predict_y":
        t = torch.rand(B, device=device)
        target_state = target - source if flow_target == 'delta' else target
        if flow_target == 'delta':
            x0 = torch.randn_like(source) * getattr(config, 'delta_noise_scale', 1.0)
        elif noise_type == "Gaussian":
            x0 = torch.randn_like(source)
        else:
            x0 = make_lognorm_poisson_noise(
                target_log=source,
                alpha=getattr(config, "poisson_alpha", 0.8),
                per_cell_L=getattr(config, "poisson_target_sum", 1e4),
            )
        path_x1 = path.sample(t=t, x_0=x0, x_1=target_state)

        use_null = cfg_dropout > 0.0 and torch.rand(1).item() < cfg_dropout
        with autocast_ctx():
            if use_null:
                base_vf = vf.module if hasattr(vf, 'module') else vf
                null_pemb = base_vf.p_mask_embed.unsqueeze(0).expand(B, -1).to(source.device)
                predicted_velocity = vf(gene_input, path_x1.x_t, path_x1.t, source,
                                        None, gene_input, perturbation_emb=null_pemb,
                                        perturbation_gene_id=perturbation_gene_id, mode=mode)
            else:
                predicted_velocity = vf(gene_input, path_x1.x_t, path_x1.t, source,
                                        perturbation_id, gene_input,
                                        perturbation_gene_id=perturbation_gene_id, mode=mode)
        predicted_velocity = predicted_velocity.float()

        loss = ((sl(predicted_velocity) - sl(path_x1.dx_t)) ** 2).mean()

        state_hat = path_x1.x_t + predicted_velocity * (1 - t).unsqueeze(-1)
        if flow_target == 'delta':
            x1_hat = source + state_hat
            dir_pred = state_hat
            dir_true = target_state
        else:
            x1_hat = state_hat
            dir_pred = x1_hat - source
            dir_true = target - source

        rec_w = getattr(config, 'recon_weight', 0.5)
        bulk_w = getattr(config, 'bulk_loss_weight', 2.0)
        dir_w = getattr(config, 'dir_weight', 0.0)
        if rec_w > 0:
            loss = loss + rec_w * ((sl(x1_hat) - sl(target)) ** 2).mean()
        if bulk_w > 0:
            loss = loss + bulk_w * ((sl(x1_hat).mean(0) - sl(target).mean(0)) ** 2).mean()
        if dir_w > 0:
            cos = F.cosine_similarity(sl(dir_pred), sl(dir_true), dim=-1, eps=1e-8).mean()
            loss = loss + dir_w * (1.0 - cos)

        endpoint_type = getattr(config, 'endpoint_loss', 'mmd')
        if endpoint_type != 'none' and config.gamma > 0:
            x1_e = sl(x1_hat)
            tgt_e = sl(target)
            w = None
            if (deg_weights is not None and condition_idx is not None and not use_null
                    and endpoint_type in ('deg_mse', 'deg_sinkhorn')):
                ci = condition_idx
                if ci.dim() > 1:
                    ci = ci[0]
                key = tuple(sorted(int(v) for v in ci.reshape(-1).tolist()))
                w_full = deg_weights.get(key, None)
                if w_full is not None:
                    w = w_full[loss_gene_index.cpu()].to(device)

            if endpoint_type == 'mmd' or (endpoint_type in ('deg_mse', 'deg_sinkhorn') and w is None):
                sigmas = median_sigmas(tgt_e, scales=(0.5, 1.0, 2.0, 4.0))
                endpoint_loss = mmd2_unbiased_multi_sigma(x1_e, tgt_e, sigmas)
            elif endpoint_type == 'deg_mse':
                weighted_mse = (w * (x1_e - tgt_e) ** 2).mean()
                sigmas = median_sigmas(tgt_e, scales=(1.0, 2.0))
                endpoint_loss = weighted_mse + 0.1 * mmd2_unbiased_multi_sigma(x1_e, tgt_e, sigmas)
            elif endpoint_type in ('sinkhorn', 'deg_sinkhorn'):
                eps_scale = getattr(config, 'sinkhorn_eps', 0.1)
                n_iters = getattr(config, 'sinkhorn_iters', 50)
                with torch.no_grad():
                    med = torch.median(weighted_sqdist(x1_e, tgt_e, w)).clamp_min(1e-6)
                    eps = (eps_scale * med).clamp_min(1e-3)
                endpoint_loss = sinkhorn_divergence(x1_e, tgt_e, eps, n_iters, w=w)
                if not torch.isfinite(endpoint_loss):
                    endpoint_loss = torch.zeros((), device=device)
            else:
                raise ValueError(f"Unknown endpoint_loss: {endpoint_type}")

            loss = loss + endpoint_loss * config.gamma

    elif mode == "predict_p":
        t_p = torch.ones(B, device=device)
        with autocast_ctx():
            predicted_p_embed = vf(gene_input, target, t_p, source, perturbation_id, gene_input,
                                   perturbation_gene_id=perturbation_gene_id, mode=mode)
        predicted_p_embed = predicted_p_embed.float()
        base_vf = vf.module if hasattr(vf, "module") else vf
        p_embed_gt = base_vf.get_perturbation_emb(perturbation_id=perturbation_id, cell_1=source)
        pred = F.normalize(predicted_p_embed, dim=-1)
        tgt = F.normalize(p_embed_gt.detach().float(), dim=-1)
        loss = 1 - (pred * tgt).sum(dim=-1).mean()

    return loss


def wrapped_vf(target, t, source, perturbation_id, vf, gene_ids, gene_all,
               perturbation_emb=None, perturbation_gene_id=None):
    gene = gene_ids.repeat(source.shape[0], 1).to(device)
    with autocast_ctx():
        out = vf(gene, target, t, source, perturbation_id, gene_all,
                 perturbation_emb=perturbation_emb, perturbation_gene_id=perturbation_gene_id)
    return out.float()


@torch.no_grad()
def generate_sample(wrapped_vf, source, condition_vec=None, vf=None, gene_ids=None,
                    gene_all=None, steps=None, method=None, cfg_scale=0.0,
                    perturbation_gene_id=None):
    is_void = config.fusion_method == 'void'
    flow_target = getattr(config, 'flow_target', 'cell')
    noise_type = config.noise_type
    steps = config.ode_steps if steps is None else steps
    method = config.ode_method if method is None else method
    n_gene = source.shape[-1] if is_void else config.infer_top_gene

    if flow_target == 'delta':
        target_noise = torch.randn(source.shape[0], n_gene, device=source.device) \
            * getattr(config, 'delta_noise_scale', 1.0)
    elif noise_type == "Gaussian":
        target_noise = torch.randn(source.shape[0], n_gene, device=source.device)
    else:
        target_noise = make_lognorm_poisson_noise(
            target_log=source,
            alpha=getattr(config, "poisson_alpha", 0.8),
            per_cell_L=getattr(config, "poisson_target_sum", 1e4),
        )

    null_pemb = None
    if cfg_scale > 0.0:
        base_vf = vf.module if hasattr(vf, 'module') else vf
        null_pemb = base_vf.p_mask_embed.unsqueeze(0).expand(source.shape[0], -1).to(source.device)

    def field(t, x):
        v_cond = wrapped_vf(x, t, source, condition_vec, vf, gene_ids, gene_all,
                            perturbation_gene_id=perturbation_gene_id)
        if cfg_scale > 0.0:
            v_uncond = wrapped_vf(x, t, source, None, vf, gene_ids, gene_all,
                                  perturbation_emb=null_pemb,
                                  perturbation_gene_id=perturbation_gene_id)
            return v_cond + cfg_scale * (v_cond - v_uncond)
        return v_cond

    if is_void:
        x = target_noise
        dt = 1.0 / float(steps)
        for i in range(steps):
            t = torch.full((source.shape[0],), i / float(steps), device=source.device)
            x = x + dt * field(t, x)
        last = x
    else:
        last = torchdiffeq.odeint(
            field, target_noise, torch.linspace(0, 1, steps).to(source.device),
            atol=1e-4, rtol=1e-4, method=method,
        )[-1]

    out = source + last if flow_target == 'delta' else last
    return torch.clamp(out, min=0)


@torch.inference_mode()
def test(data_sampler, vf, accelerator, batch_size=128, path='./', vocab=None, scheme='mse',
         panel_pos=None):
    is_void = config.fusion_method == 'void'

    if is_void:
        panel_size = len(panel_pos)
        pos_to_token = [0] * panel_size
        for tok, p in panel_pos.items():
            pos_to_token[p] = int(tok)
        gene_ids_test = torch.tensor(pos_to_token, dtype=torch.long, device=device)

        valid_tokens = vocab.encode(list(data_sampler.adata.var_names))
        valid_tok_to_col = {int(t): i for i, t in enumerate(valid_tokens)}
        sel = torch.tensor([valid_tok_to_col.get(t, -1) for t in pos_to_token], dtype=torch.long)
        sel_valid = sel >= 0

        def to_panel(arr):
            t = arr if torch.is_tensor(arr) else torch.as_tensor(np.asarray(arr))
            t = t.float()
            out = torch.zeros(t.shape[0], panel_size, dtype=torch.float32)
            out[:, sel_valid] = t[:, sel[sel_valid]]
            return out
    else:
        gene_ids_test = vocab.encode(list(data_sampler.adata.var_names))
        gene_ids_test = torch.tensor(gene_ids_test, dtype=torch.long, device=device)

        def to_panel(arr):
            return arr

    perturbation_name_list = data_sampler._perturbation_covariates
    control_data = data_sampler.get_control_data()
    control_src = to_panel(control_data['src_cell_data'])

    if is_void:
        n_eval = min(config.infer_top_gene, control_src.shape[1])
        cmean = control_src.float().mean(0)
        eval_gene_idx = torch.argsort(cmean, descending=True)[:n_eval].sort().values.cpu().numpy()
    else:
        eval_gene_idx = None

    def report(arr):
        return arr[:, eval_gene_idx] if eval_gene_idx is not None else arr

    control_src_np = control_src.cpu().numpy() if torch.is_tensor(control_src) else np.asarray(control_src)

    all_pred_expressions = [report(control_src_np)]
    obs_perturbation_name_pred = ['control'] * control_src_np.shape[0]
    all_target_expressions = [report(control_src_np)]
    obs_perturbation_name_real = ['control'] * control_src_np.shape[0]

    print('perturbation_name_list:', len(perturbation_name_list))
    for perturbation_name in perturbation_name_list:
        perturbation_data = data_sampler.get_perturbation_data(perturbation_name)
        target = to_panel(perturbation_data['tgt_cell_data'])
        target = target.cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
        perturbation_id = perturbation_data['condition_id'].to(device)
        source = control_src.to(device)
        if config.perturbation_function == 'crisper':
            names = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
            if all(name in vocab.stoi for name in names):
                perturbation_id = torch.tensor(vocab.encode(names), dtype=torch.long, device=device)
                perturbation_id = perturbation_id.repeat(source.shape[0], 1)

        idx = torch.randperm(source.shape[0])
        source = source[idx]
        N = 128
        source = source[:N]

        pred_expressions = []
        for i in trange(0, N, batch_size):
            src_b = source[i:i + batch_size]
            batch_perturbation_id = perturbation_id[0].repeat(src_b.shape[0], 1).to(accelerator.device)
            pgid = None
            if is_void and panel_pos is not None:
                pgid = perturbation_gene_ids_from_tokens(batch_perturbation_id, panel_pos, accelerator.device)
            pred_expression = generate_sample(
                wrapped_vf, src_b, batch_perturbation_id,
                vf, gene_ids=gene_ids_test, gene_all=gene_ids_test, cfg_scale=0.0,
                perturbation_gene_id=pgid,
            )
            pred_expressions.append(pred_expression)

        pred_expressions = torch.cat(pred_expressions, dim=0).cpu().numpy()
        all_pred_expressions.append(report(pred_expressions))
        all_target_expressions.append(report(target))
        obs_perturbation_name_pred.extend([perturbation_name] * pred_expressions.shape[0])
        obs_perturbation_name_real.extend([perturbation_name] * target.shape[0])

    all_pred_expressions = np.concatenate(all_pred_expressions, axis=0)
    n_nan = np.isnan(all_pred_expressions).sum()
    n_inf = np.isinf(all_pred_expressions).sum()
    print(f"[eval] pred non-finite: nan={n_nan} inf={n_inf} "
          f"min={np.nanmin(all_pred_expressions):.4f} max={np.nanmax(all_pred_expressions):.4f}")
    all_pred_expressions = np.nan_to_num(all_pred_expressions, nan=0.0, posinf=0.0, neginf=0.0)
    all_target_expressions = np.concatenate(all_target_expressions, axis=0)
    obs_pred = pd.DataFrame({'perturbation': obs_perturbation_name_pred})
    obs_real = pd.DataFrame({'perturbation': obs_perturbation_name_real})
    pred = ad.AnnData(X=all_pred_expressions, obs=obs_pred)
    real = ad.AnnData(X=all_target_expressions, obs=obs_real)

    eval_score = None
    if accelerator.is_main_process:
        evaluator = MetricsEvaluator(adata_pred=pred, adata_real=real,
                                     control_pert="control", pert_col="perturbation", num_threads=32, outdir=path)
        (results, agg_results) = evaluator.compute()
        results.write_csv(os.path.join(path, 'results.csv'))
        agg_results.write_csv(os.path.join(path, 'agg_results.csv'))
        pred.write_h5ad(os.path.join(path, 'pred.h5ad'))
        real.write_h5ad(os.path.join(path, 'real.h5ad'))
        eval_score = pick_eval_score(agg_results, scheme)
        print(f"Current evaluation score: {eval_score:.4f}")

    return eval_score


def find_latest_checkpoint(save_path: str) -> str | None:
    if not os.path.isdir(save_path):
        return None
    candidates = []
    for name in os.listdir(save_path):
        match = re.match(r"iteration_(\d+)$", name)
        if match:
            ckpt_path = os.path.join(save_path, name, "checkpoint.pt")
            if os.path.isfile(ckpt_path):
                candidates.append((int(match.group(1)), ckpt_path))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


if __name__ == "__main__":
    config = tyro.cli(Config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(42)

    if accelerator.is_main_process:
        print(config)
        save_path = config.make_path()
        os.makedirs(save_path, exist_ok=True)
    device = accelerator.device

    data_manager = Data('./data')
    data_manager.load_data(config.data_name)
    data_manager.process_data(
        n_top_genes=config.n_top_genes, infer_top_gene=config.infer_top_gene,
        split_method=config.split_method, fold=config.fold,
        use_negative_edge=config.use_negative_edge, k=config.topk,
    )
    train_sampler, valid_sampler, test_dl = data_manager.load_flow_data(batch_size=config.batch_size)

    train_dataset = PerturbationDataset(
        train_sampler, config.batch_size,
        cell_type_col='cell_type',
        plate_col=config.plate_col if config.plate_col else None,
    )
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    if config.use_negative_edge:
        mask_path = os.path.join(
            data_manager.data_path, data_manager.data_name,
            'mask_fold_' + str(config.fold) + 'topk_' + str(config.topk) + config.split_method + '_negative_edge' + '.pt')
    else:
        mask_path = os.path.join(
            data_manager.data_path, data_manager.data_name,
            'mask_fold_' + str(config.fold) + 'topk_' + str(config.topk) + config.split_method + '.pt')

    vocab = process_vocab(data_manager, config)
    is_void = config.fusion_method == 'void'

    if not is_void:
        _adj_shape = torch.load(mask_path).shape[0]
        assert _adj_shape == len(vocab), \
            f"co-expression mask rows {_adj_shape} != vocab size {len(vocab)}"

    model_wire_path = None
    if config.use_wire and not is_void:
        wire_path = mask_path.replace('.pt', '_wire_eigvecs.pt')
        if not os.path.exists(wire_path):
            from src.data_process.precompute_wire import compute_laplacian_eigvecs
            eigvec_dim = getattr(config, 'wire_eigvec_dim', 32)
            hvg_gene_names = list(data_manager.adata.var_names)
            gene_token_ids = vocab.encode(hvg_gene_names)
            ntoken_local = max(gene_token_ids) + 1
            X = data_manager.adata_train.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = X.astype(np.float32)
            coords_hvg = compute_laplacian_eigvecs(X, eigvec_dim, k=config.topk)
            spectral_coords = torch.zeros(ntoken_local, eigvec_dim, dtype=torch.float32)
            for pos, token_id in enumerate(gene_token_ids):
                spectral_coords[token_id] = torch.from_numpy(coords_hvg[pos])
            torch.save(spectral_coords, wire_path)
        model_wire_path = wire_path

    if is_void:
        gene_ids = vocab.encode(list(train_sampler.adata.var_names))
    else:
        gene_ids = vocab.encode(list(data_manager.adata.var_names))
    gene_ids = torch.tensor(gene_ids, dtype=torch.long, device=device)
    panel_pos = {int(t): i for i, t in enumerate(gene_ids.cpu().tolist())}

    grn_mask_path_arg = None
    if getattr(config, 'grn_mask_path', '') != '' and not is_void:
        grn_cache = mask_path.replace('.pt', '_grn.pt')
        if not os.path.exists(grn_cache):
            edges = pd.read_csv(config.grn_mask_path).itertuples(index=False, name=None)
            name_to_token = {n: vocab.encode([n])[0] for n in data_manager.adata.var_names if n in vocab.stoi}
            grn_mask = build_grn_neighbor_mask(edges, name_to_token, ntoken=len(vocab), k=config.topk)
            torch.save(grn_mask, grn_cache)
        grn_mask_path_arg = grn_cache

    vf = instantiate_model(
        config.model_type,
        ntoken=len(vocab),
        d_model=config.d_model,
        d_perturbation=config.d_model,
        fusion_method=config.fusion_method,
        perturbation_function=config.perturbation_function,
        mask_path=mask_path,
        wire_path=model_wire_path,
        use_wire=config.use_wire,
        use_repo=config.use_repo,
        grn_mask_path=grn_mask_path_arg,
        void_encode_blocks=config.void_encode_blocks,
        void_think_steps=config.void_think_steps,
        void_hidden=config.void_hidden,
        void_manifold_dim=config.manifold_dim,
        void_neighbor_chunk=config.void_neighbor_chunk,
        void_checkpoint=config.void_checkpoint,
    )

    if is_void:
        void_panel = list(train_sampler.adata.var_names)
        X_panel = train_sampler.adata.X
        if hasattr(X_panel, 'toarray'):
            X_panel = X_panel.toarray()
        X_panel = np.asarray(X_panel, dtype=np.float32)
        assert X_panel.shape[1] == len(void_panel) == train_sampler.adata.shape[1], \
            f"void geometry width {X_panel.shape[1]} != sampler panel {len(void_panel)}"
        nbr, w, coords = build_void_geometry(
            X_panel, k=config.topk, manifold_dim=config.manifold_dim, seed=42)
        vf.set_geometry(nbr, w, coords)

    save_path = config.make_path()

    optimizer = torch.optim.Adam(vf.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps, eta_min=config.eta_min)

    if config.checkpoint_path == '':
        config.checkpoint_path = find_latest_checkpoint(save_path) or ''

    start_iteration = 0
    if config.checkpoint_path != '':
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {config.checkpoint_path}")
        load_checkpoint(config.checkpoint_path, vf, optimizer, scheduler)
        match = re.search(r"iteration_(\d+)", config.checkpoint_path)
        if match:
            start_iteration = int(match.group(1)) + 1

    if start_iteration >= config.steps:
        if accelerator.is_main_process:
            print(f"Checkpoint already at iteration {start_iteration - 1}, nothing to do.")
        start_iteration = config.steps

    deg_weights = None
    if getattr(config, 'endpoint_loss', 'mmd') in ('deg_mse', 'deg_sinkhorn') and config.gamma > 0:
        if accelerator.is_main_process:
            print("Precomputing DEG weights...")
        deg_weights = compute_deg_weights(train_sampler, epsilon=getattr(config, 'deg_epsilon', 0.1))
        if accelerator.is_main_process:
            print(f"DEG weight conditions: {len(deg_weights)}; sample keys: {list(deg_weights)[:3]}")

    vf = accelerator.prepare(vf)
    if is_void and config.devices == "1":
        vf = torch.compile(vf, dynamic=True)
    optimizer, scheduler, dataloader = accelerator.prepare(optimizer, scheduler, dataloader)
    inverse_dict = {v: str(k) for k, v in data_manager.perturbation_dict.items()}
    pbar = tqdm.tqdm(total=config.steps, initial=start_iteration)
    iteration = start_iteration

    while iteration < config.steps:
        for batch_data in dataloader:
            if iteration >= config.steps:
                break
            source = batch_data['src_cell_data'].squeeze(0)
            target = batch_data['tgt_cell_data'].squeeze(0)
            condition_idx_raw = batch_data['condition_id'].squeeze(0)
            perturbation_id = batch_data['condition_id'].squeeze(0).to(device)
            if config.perturbation_function == 'crisper':
                names = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
                if all(name in vocab.stoi for name in names):
                    perturbation_id = torch.tensor(vocab.encode(names), dtype=torch.long, device=device)
                    perturbation_id = perturbation_id.repeat(source.shape[0], 1)

            perturbation_gene_id = None
            if is_void:
                perturbation_gene_id = perturbation_gene_ids_from_tokens(perturbation_id, panel_pos, device)

            set_requires_grad_for_p_only(vf, p_only=config.mode)

            loss = train_step(source, target, perturbation_id, vf, accelerator,
                              noise_type=config.noise_type, mode=config.mode, cfg_dropout=0.0,
                              deg_weights=deg_weights, condition_idx=condition_idx_raw,
                              perturbation_gene_id=perturbation_gene_id)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(vf.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if iteration > 0 and iteration % config.print_every == 0:
                save_path_ = os.path.join(save_path, f'iteration_{iteration}')
                os.makedirs(save_path_, exist_ok=True)
                if accelerator.is_main_process:
                    print(f"saving {iteration}'s checkpoint...")
                    save_checkpoint(
                        model=accelerator.unwrap_model(vf), optimizer=optimizer, scheduler=scheduler,
                        iteration=iteration, eval_score=None, save_path=save_path_, is_best=False)
                eval_score = test(valid_sampler, vf, accelerator,
                                  batch_size=config.batch_size, path=save_path_, vocab=vocab,
                                  panel_pos=panel_pos)

            accelerator.wait_for_everyone()
            pbar.update(1)
            pbar.set_description(f'loss: {loss.item():.4f}, iteration: {iteration}')
            iteration += 1