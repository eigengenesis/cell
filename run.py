import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import hashlib
import inspect
from pathlib import Path
import torch
import torch.nn as nn
import tyro
from config_flow import FlowConfig as Config
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from src.data_process.data import Data, PerturbationDataset
from src.flow_matching.ot import OTPlanSampler
from src.flow_matching.path import AffineProbPath
from instantiate_model import instantiate_model
from sampling import (
    action_aware_gene_sample,
    build_gene_column_lookup,
    build_neighbor_column_table,
)
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

ot_sampler = OTPlanSampler(method="exact")
path = AffineProbPath(scheduler=CondOTScheduler())


def log_source_fingerprint(label, obj):
    source_path = Path(inspect.getsourcefile(obj)).resolve()
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    print(f"[source] {label} path={source_path} sha256={digest}", flush=True)


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
    """
    Per-combination per-gene weight from training log-fold-change, keyed by the
    sorted gene-id pair tuple (same space as batch_data['condition_id']).
    w_i = epsilon + (1 - epsilon) * |lfc_i| / max_j|lfc_j|.
    Returns {(id_a, id_b): (n_genes,) CPU tensor}.
    """
    adata = train_sampler.adata
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    cond = adata.obs['perturbation_covariates'].astype(str).values
    pair_ids = np.asarray(train_sampler.perturbation_covariates_id)        # (N, 2)

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


def train_step(source, target, perturbation_id, vf, accelerator,
               noise_type='Poisson', mode="predict_y", cfg_dropout=0.0,
               deg_weights=None, condition_idx=None,
               gene_column_by_token=None, sampling_neighbor_columns=None):
    B = source.shape[0]
    device = accelerator.device

    if gene_column_by_token is not None and sampling_neighbor_columns is not None:
        input_gene_ids, target_count, mandatory_count, target_coverage = action_aware_gene_sample(
            source.shape[-1], config.infer_top_gene, perturbation_id,
            gene_column_by_token, sampling_neighbor_columns, device,
        )
    else:
        input_gene_ids = torch.randperm(source.shape[-1], device=device)[:config.infer_top_gene]
        target_count, mandatory_count, target_coverage = 0, 0, float('nan')
    source = source[:, input_gene_ids]
    target = target[:, input_gene_ids]
    gene = gene_ids.repeat(B, 1).to(device)
    gene_input = gene[:, input_gene_ids]

    if mode == "predict_y":
        t = torch.rand(B, device=device)
        if noise_type == "Gaussian":
            target_noise = torch.randn_like(source)
        elif noise_type == "Poisson":
            target_noise = make_lognorm_poisson_noise(
                target_log=source,
                alpha=getattr(config, "poisson_alpha", 0.8),
                per_cell_L=getattr(config, "poisson_target_sum", 1e4),
            )
        path_x1 = path.sample(t=t, x_0=target_noise, x_1=target)

        use_null = cfg_dropout > 0.0 and torch.rand(1).item() < cfg_dropout
        if use_null:
            base_vf = vf.module if hasattr(vf, 'module') else vf
            null_pemb = base_vf.p_mask_embed.unsqueeze(0).expand(B, -1).to(source.device)
            predicted_velocity = vf(gene_input, path_x1.x_t, path_x1.t, source,
                                    None, gene_input, perturbation_emb=null_pemb, mode=mode)
        else:
            predicted_velocity = vf(gene_input, path_x1.x_t, path_x1.t, source,
                                    perturbation_id, gene_input, mode=mode)

        loss = ((predicted_velocity - path_x1.dx_t) ** 2).mean()
        x1_hat = path_x1.x_t + predicted_velocity * (1 - t).unsqueeze(-1)

        endpoint_type = getattr(config, 'endpoint_loss', 'mmd')
        if endpoint_type != 'none':
            w = None
            if (deg_weights is not None and condition_idx is not None and not use_null
                    and endpoint_type in ('deg_mse', 'deg_sinkhorn')):
                ci = condition_idx
                if ci.dim() > 1:
                    ci = ci[0]
                key = tuple(sorted(int(v) for v in ci.reshape(-1).tolist()))
                w_full = deg_weights.get(key, None)
                if w_full is not None:
                    w = w_full[input_gene_ids.cpu()].to(device)

            if endpoint_type == 'mmd' or (endpoint_type in ('deg_mse', 'deg_sinkhorn') and w is None):
                sigmas = median_sigmas(target, scales=(0.5, 1.0, 2.0, 4.0))
                endpoint_loss = mmd2_unbiased_multi_sigma(x1_hat, target, sigmas)
            elif endpoint_type == 'deg_mse':
                weighted_mse = (w * (x1_hat - target) ** 2).mean()
                sigmas = median_sigmas(target, scales=(1.0, 2.0))
                endpoint_loss = weighted_mse + 0.1 * mmd2_unbiased_multi_sigma(x1_hat, target, sigmas)
            elif endpoint_type in ('sinkhorn', 'deg_sinkhorn'):
                eps_scale = getattr(config, 'sinkhorn_eps', 0.1)
                n_iters = getattr(config, 'sinkhorn_iters', 50)
                with torch.no_grad():
                    med = torch.median(weighted_sqdist(x1_hat, target, w)).clamp_min(1e-6)
                    eps = (eps_scale * med).clamp_min(1e-3)
                endpoint_loss = sinkhorn_divergence(x1_hat, target, eps, n_iters, w=w)
                if not torch.isfinite(endpoint_loss):
                    endpoint_loss = torch.zeros((), device=device)
            else:
                raise ValueError(f"Unknown endpoint_loss: {endpoint_type}")

            loss = loss + endpoint_loss * config.gamma

    elif mode == "predict_p":
        t_p = torch.ones(B, device=device)
        predicted_p_embed = vf(gene_input, target, t_p, source, perturbation_id, gene_input, mode=mode)
        base_vf = vf.module if hasattr(vf, "module") else vf
        p_embed_gt = base_vf.get_perturbation_emb(perturbation_id=perturbation_id, cell_1=source)
        pred = F.normalize(predicted_p_embed, dim=-1)
        tgt = F.normalize(p_embed_gt.detach(), dim=-1)
        loss = 1 - (pred * tgt).sum(dim=-1).mean()

    sampling_stats = {
        'target_count': target_count,
        'mandatory_count': mandatory_count,
        'target_coverage': target_coverage,
    }
    return loss, sampling_stats


def wrapped_vf(target, t, source, perturbation_id, vf, gene_ids, gene_all, perturbation_emb=None):
    gene = gene_ids.repeat(source.shape[0], 1).to(device)
    out = vf(gene, target, t, source, perturbation_id, gene_all, perturbation_emb=perturbation_emb)
    return out.clone()


@torch.no_grad()
def generate_sample(wrapped_vf, source, condition_vec=None, vf=None, gene_ids=None,
                    gene_all=None, steps=20, method="rk4", cfg_scale=0.0):
    noise_type = config.noise_type
    if noise_type == "Gaussian":
        target_noise = torch.randn(source.shape[0], config.infer_top_gene, device=source.device)
    elif noise_type == "Poisson":
        target_noise = make_lognorm_poisson_noise(
            target_log=source,
            alpha=getattr(config, "poisson_alpha", 0.8),
            per_cell_L=getattr(config, "poisson_target_sum", 1e4),
        )

    null_pemb = None
    if cfg_scale > 0.0:
        base_vf = vf.module if hasattr(vf, 'module') else vf
        null_pemb = base_vf.p_mask_embed.unsqueeze(0).expand(source.shape[0], -1).to(source.device)

    def ode_fn(t, x):
        v_cond = wrapped_vf(x, t, source, condition_vec, vf, gene_ids, gene_all)
        if cfg_scale > 0.0:
            v_uncond = wrapped_vf(x, t, source, None, vf, gene_ids, gene_all, perturbation_emb=null_pemb)
            return v_cond + cfg_scale * (v_cond - v_uncond)
        return v_cond

    traj = torchdiffeq.odeint(
        ode_fn, target_noise, torch.linspace(0, 1, steps).to(source.device),
        atol=1e-4, rtol=1e-4, method=method,
    )
    return torch.clamp(traj[-1], min=0)


@torch.inference_mode()
def test(data_sampler, vf, accelerator, batch_size=128, path='./', vocab=None, scheme='mse'):
    gene_ids_test = vocab.encode(list(data_sampler.adata.var_names))
    gene_ids_test = torch.tensor(gene_ids_test, dtype=torch.long, device=device)
    perturbation_name_list = data_sampler._perturbation_covariates
    control_data = data_sampler.get_control_data()
    all_pred_expressions = [control_data['src_cell_data']]
    obs_perturbation_name_pred = ['control'] * control_data['src_cell_data'].shape[0]
    all_target_expressions = [control_data['src_cell_data']]
    obs_perturbation_name_real = ['control'] * control_data['src_cell_data'].shape[0]

    print('perturbation_name_list:', len(perturbation_name_list))
    for perturbation_name in perturbation_name_list:
        perturbation_data = data_sampler.get_perturbation_data(perturbation_name)
        target = perturbation_data['tgt_cell_data']
        perturbation_id = perturbation_data['condition_id'].to(device)
        source = control_data['src_cell_data'].to(device)
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
            batch_perturbation_id = perturbation_id[0].repeat(source[i:i + batch_size].shape[0], 1).to(accelerator.device)
            pred_expression = generate_sample(
                wrapped_vf, source[i:i + batch_size], batch_perturbation_id,
                vf, gene_ids=gene_ids_test, gene_all=gene_ids_test, cfg_scale=0.0 #cfg_scale=1.5
            )
            pred_expressions.append(pred_expression)

        pred_expressions = torch.cat(pred_expressions, dim=0).cpu().numpy()
        all_pred_expressions.append(pred_expressions)
        all_target_expressions.append(target)
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
    torch.set_float32_matmul_precision("high")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[ddp_kwargs])

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
        log_source_fingerprint('run', train_step)
        log_source_fingerprint('config', Config)
        log_source_fingerprint('instantiate_model', instantiate_model)
        log_source_fingerprint('sampling', action_aware_gene_sample)
        log_source_fingerprint('scdfm_data', Data)
        log_source_fingerprint('scdfm_dataset', PerturbationDataset)
    device = accelerator.device

    data_manager = Data('./data')
    data_manager.load_data(config.data_name)
    data_manager.process_data(
        n_top_genes=config.n_top_genes, infer_top_gene=config.infer_top_gene,
        split_method=config.split_method, fold=config.fold,
        use_negative_edge=config.use_negative_edge, k=config.topk,
    )
    train_sampler, valid_sampler, test_dl = data_manager.load_flow_data(batch_size=config.batch_size)

    dataset_parameters = inspect.signature(PerturbationDataset).parameters
    dataset_kwargs = {}
    if 'cell_type_col' in dataset_parameters:
        dataset_kwargs['cell_type_col'] = 'cell_type'
    if 'plate_col' in dataset_parameters:
        dataset_kwargs['plate_col'] = config.plate_col if config.plate_col else None
    train_dataset = PerturbationDataset(train_sampler, config.batch_size, **dataset_kwargs)
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

    _adj_shape = torch.load(mask_path).shape[0]
    assert _adj_shape == len(vocab), \
        f"co-expression mask rows {_adj_shape} != vocab size {len(vocab)}; " \
        f"mask row order must equal vocab token-id order"

    if config.use_wire:
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
    else:
        model_wire_path = None

    gene_ids = vocab.encode(list(data_manager.adata.var_names))
    gene_ids = torch.tensor(gene_ids, dtype=torch.long, device=device)

    grn_mask_path_arg = None
    if getattr(config, 'grn_mask_path', '') != '':
        grn_cache = mask_path.replace('.pt', '_grn.pt')
        if not os.path.exists(grn_cache):
            edges = pd.read_csv(config.grn_mask_path).itertuples(index=False, name=None)
            name_to_token = {n: vocab.encode([n])[0] for n in data_manager.adata.var_names if n in vocab.stoi}
            grn_mask = build_grn_neighbor_mask(edges, name_to_token, ntoken=len(vocab), k=config.topk)
            torch.save(grn_mask, grn_cache)
        grn_mask_path_arg = grn_cache

    corr_path_arg = None
    if getattr(config, "use_signed_edges", False):
        corr_path = mask_path.replace('.pt', '_corr.pt')
        if not os.path.exists(corr_path):
            hvg_gene_names = list(data_manager.adata.var_names)
            gene_token_ids = vocab.encode(hvg_gene_names)
            X = data_manager.adata_train.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
            corr_hvg = np.corrcoef(X.T).astype(np.float32)
            corr_hvg = np.nan_to_num(corr_hvg, nan=0.0, posinf=0.0, neginf=0.0)
            corr_full = torch.zeros(len(vocab), len(vocab), dtype=torch.float32)
            idx = torch.tensor(gene_token_ids, dtype=torch.long)
            corr_full[idx.unsqueeze(1), idx.unsqueeze(0)] = torch.from_numpy(corr_hvg)
            torch.save(corr_full, corr_path)
        corr_path_arg = corr_path

    vf = instantiate_model(
        config.model_type,
        ntoken=len(vocab),
        d_model=config.d_model,
        d_hid=config.d_hid,
        encode_blocks=config.encode_blocks,
        think_steps=config.think_steps,
        d_perturbation=config.d_model,
        fusion_method=config.fusion_method,
        perturbation_function=config.perturbation_function,
        control_token_id=vocab.stoi.get('control'),
        mask_path=mask_path,
        wire_path=model_wire_path,
        use_wire=config.use_wire,
        use_repo=config.use_repo,
        grn_mask_path=grn_mask_path_arg,
        corr_path=corr_path_arg,
    )

    gene_column_by_token = build_gene_column_lookup(gene_ids, len(vocab)).to(device)
    sampling_neighbor_columns = build_neighbor_column_table(
        mask_path=mask_path,
        gene_token_ids=gene_ids,
        vocab_size=len(vocab),
        topk=config.topk,
        corr_path=corr_path_arg,
    ).to(device)
    if accelerator.is_main_process:
        log_source_fingerprint('model', type(vf))
        print(
            f"[model] params={sum(p.numel() for p in vf.parameters())} "
            f"d_model={config.d_model} d_hid={config.d_hid} "
            f"encode_blocks={config.encode_blocks} think_steps={config.think_steps}",
            flush=True,
        )
        print(
            f"[field] genes={config.infer_top_gene} target_neighbors={config.topk} "
            "sampler=action_aware",
            flush=True,
        )

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
            print(f"Checkpoint already at iteration {start_iteration - 1}, "
                  f"target steps={config.steps} reached. Nothing to do.")
        start_iteration = config.steps

    deg_weights = None
    if getattr(config, 'endpoint_loss', 'mmd') in ('deg_mse', 'deg_sinkhorn'):
        if accelerator.is_main_process:
            print("Precomputing DEG weights...")
        deg_weights = compute_deg_weights(train_sampler, epsilon=getattr(config, 'deg_epsilon', 0.1))
        if accelerator.is_main_process:
            print(f"DEG weight conditions: {len(deg_weights)}; sample keys: {list(deg_weights)[:3]}")
            _w_len = next(iter(deg_weights.values())).shape[0]
            _n_genes = train_sampler.adata.shape[1]
            assert _w_len == _n_genes, f"DEG weight length {_w_len} != adata n_genes {_n_genes}"

    vf = accelerator.prepare(vf)
    vf = torch.compile(vf, mode="max-autotune")
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
            condition_idx_raw = batch_data['condition_id'].squeeze(0)        # before CRISPR token conversion
            perturbation_id = batch_data['condition_id'].squeeze(0).to(device)
            if config.perturbation_function == 'crisper':
                names = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
                if all(name in vocab.stoi for name in names):
                    perturbation_id = torch.tensor(vocab.encode(names), dtype=torch.long, device=device)
                    perturbation_id = perturbation_id.repeat(source.shape[0], 1)

            set_requires_grad_for_p_only(vf, p_only=config.mode)

            loss, sampling_stats = train_step(
                source, target, perturbation_id, vf, accelerator,
                noise_type=config.noise_type, mode=config.mode, cfg_dropout=0.0,
                deg_weights=deg_weights, condition_idx=condition_idx_raw,
                gene_column_by_token=gene_column_by_token,
                sampling_neighbor_columns=sampling_neighbor_columns,
            )

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
                                  batch_size=config.batch_size, path=save_path_, vocab=vocab)

            if accelerator.is_main_process and (iteration == start_iteration or iteration % 1000 == 0):
                print(
                    f"[field] iteration={iteration} target_coverage={sampling_stats['target_coverage']:.3f} "
                    f"targets={sampling_stats['target_count']} mandatory_genes={sampling_stats['mandatory_count']}",
                    flush=True,
                )

            accelerator.wait_for_everyone()
            pbar.update(1)
            pbar.set_description(f'loss: {loss.item():.4f}, iteration: {iteration}')
            iteration += 1
