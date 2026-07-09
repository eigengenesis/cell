from src.models.scGPT.model import TransformerModel
from src.models.perturbation.model import Model as FlowModel
from src.models.perturbation.model import TimedTransformer
from src.models.origin.model import model as OriginModel
import torch
def instantiate_model(model_type: str, **kwargs):
    if model_type == 'origin':
        if kwargs['fusion_method'] == 'differential_transformer':
            layers = 8
        elif kwargs['fusion_method'] == 'differential_perceiver':
            layers = 4
        else:
            layers = 8
        m = OriginModel(
            ntoken=kwargs['ntoken'],
            d_model=kwargs['d_model'],
            fusion_method=kwargs['fusion_method'],
            nlayers=layers,
            perturbation_function=kwargs['perturbation_function'],
            mask_path=kwargs['mask_path'],
            wire_path=kwargs.get('wire_path', None),
            use_wire=kwargs.get('use_wire', False),
            use_repo=kwargs.get('use_repo', False),
            grn_mask_path=kwargs.get('grn_mask_path', None),
            corr_path=kwargs.get('corr_path', None),
            num_factors=kwargs.get('num_factors', 64),
        )
        return m
    else:
        raise ValueError(f"Invalid model type: {model_type}")
if __name__ == "__main__":
    model = instantiate_model("punet128")
    x = torch.randn(32, 128, 128)
    t = torch.randn(32)
    out = model(x, t)
    print(out.shape)