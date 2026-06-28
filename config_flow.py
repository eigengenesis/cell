from dataclasses import dataclass
import os
@dataclass
class FlowConfig:
    model_type: str = 'origin'
    batch_size: int = 32
    ntoken: int = 512
    d_model: int = 512
    lr: float = 1e-5
    steps: int = 5000
    eta_min: float = 1e-7
    devices: str = "1"
    test_only: bool = False
    data_name: str = "combosciplex"
    perturbation_function: str = 'crisper'
    noise_type: str = "Gaussian"
    poisson_alpha: float = 0.8
    poisson_target_sum: int = -1
    print_every: int = 5000
    mode: str = 'predict_y'
    result_path: str = './result'
    perturbation_fusion_method: str = 'sum'
    fusion_method: str = 'cross'
    infer_top_gene: int = 1000
    n_top_genes: int = 5000
    checkpoint_path: str = ''
    gamma: float = 0.0
    split_method: str = 'additive'
    plate_col: str = ''
    use_mmd_loss: bool = False
    fold: int = 0
    use_negative_edge: bool = False
    topk: int = 15
    use_wire: bool = False
    use_repo: bool = False
    endpoint_loss: str = 'mmd'
    deg_epsilon: float = 0.1
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 50
    grn_mask_path: str = ''

    # void geometry / architecture
    manifold_dim: int = 8
    void_encode_blocks: int = 4
    void_think_steps: int = 8
    void_hidden: int = 512
    void_neighbor_chunk: int = 0      # 0 = gather all genes in one kernel
    void_checkpoint: bool = False

    # void flow / loss
    flow_target: str = 'cell'         # 'cell' | 'delta'
    delta_noise_scale: float = 1.0
    recon_weight: float = 0.5
    bulk_loss_weight: float = 2.0
    dir_weight: float = 0.0

    # ODE sampling
    ode_steps: int = 20
    ode_method: str = 'euler'

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