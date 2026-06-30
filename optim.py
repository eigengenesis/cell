from __future__ import annotations
import math
import torch


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

