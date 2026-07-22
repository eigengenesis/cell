from src.models.scGPT.model import TransformerModel
from src.models.perturbation.model import Model as FlowModel
from src.models.perturbation.model import TimedTransformer
from src.models.origin.model import model as OriginModel
import torch
def instantiate_model(model_type: str, **kwargs):
    if model_type == 'origin':
        m = OriginModel(
            ntoken=kwargs['ntoken'],
            d_model=kwargs['d_model'],
            d_hid=kwargs.get('d_hid', 2048),
            nlayers=kwargs.get('encode_blocks', 8),
            think_steps=kwargs.get('think_steps', 8),
            fusion_method=kwargs['fusion_method'],
            perturbation_function=kwargs['perturbation_function'],
            control_token_id=kwargs.get('control_token_id', None),
            mask_path=kwargs['mask_path'],
            wire_path=kwargs.get('wire_path', None),
            use_wire=kwargs.get('use_wire', False),
            use_repo=kwargs.get('use_repo', False),
            grn_mask_path=kwargs.get('grn_mask_path', None),
            corr_path=kwargs.get('corr_path', None),
            neighbor_gate=kwargs.get('neighbor_gate', False),
        )
        return m
    else:
        raise ValueError(f"Invalid model type: {model_type}")