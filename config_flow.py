from dataclasses import dataclass
import os
@dataclass
class FlowConfig:
    # Flow model type
    model_type: str = 'origin'
    # Flow Matching specific parameters
    batch_size: int = 32
    ntoken: int = 512
    d_model: int = 512
    lr: float = 1e-5
    steps: int = 5000 # iterations
    eta_min: float = 1e-7
    devices: str = "1"
    test_only: bool = False
    # Perturbation related parameters
    data_name: str = "combosciplex"
    perturbation_function: str = 'crisper'
    noise_type: str = "Gaussian"
    poisson_alpha: float = 0.8
    poisson_target_sum: int = -1
    print_every: int = 5000
    mode: str = 'predict_y'  # predict_y, predict_p
    result_path: str = './result'
    perturbation_fusion_method: str = 'sum'  # mlp, sum
    fusion_method: str = 'cross'  # cross , concat, add
    infer_top_gene: int = 1000
    n_top_genes: int = 5000
    checkpoint_path: str = ''
    gamma: float = 0.0
    split_method: str = 'additive'
    plate_col: str = ''  # Donor/plate column for batch-controlled pairing. Empty disables it.
    use_mmd_loss: bool = False
    fold: int = 0
    use_negative_edge: bool = False
    topk: int = 15
    use_wire: bool = False
    use_repo: bool = False
    # Endpoint distribution loss selector (replaces use_mmd_loss as the active gate).
    # 'mmd' | 'deg_mse' | 'sinkhorn' | 'deg_sinkhorn' | 'none'. Requires gamma > 0 to take effect.
    endpoint_loss: str = 'mmd'
    deg_epsilon: float = 0.1
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 50
    # Curated GRN graph (STRING / Reactome / GO / GRNdb) as a 2-column gene-name-pair CSV.
    # Empty string disables the second graph pass.
    grn_mask_path: str = ''
    use_signed_edges: bool = False
    neighbor_gate: bool = False
    d_hid: int = 2048
    encode_blocks: int = 8
    think_steps: int = 8
    def __post_init__(self):
        if self.data_name == 'norman_umi_go_filtered':
            self.n_top_genes = 5054
        if self.data_name == 'norman':
            self.n_top_genes = 5000
        path = self.make_path()
    def make_path(self):
        exp_name = '-'.join(['flow',
                             f'fusion_{self.fusion_method}',
                             f'{self.data_name}',
                             self.model_type,
                             self.mode,
                             f'gamma_{self.gamma}',
                             f'pertfunc_{self.perturbation_function}',
                             f'lr_{self.lr}',
                             f'dim_{self.d_model}',
                             f'infertop_{self.infer_top_gene}',
                             f'split_{self.split_method}',
                             f'mmd_{self.use_mmd_loss}',
                             f'fold_{self.fold}',
                             f'negedge_{self.use_negative_edge}',
                             f'topk_{self.topk}',
                             f'wire_{self.use_wire}',
                             f'repo_{self.use_repo}',
                             f'endpoint_{self.endpoint_loss}',
                             f'grn_{self.grn_mask_path != ""}',
                             ])
        return os.path.join(self.result_path, exp_name)