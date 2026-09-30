"""Strict target-specific checkpoints for the optional ASTRA fine-tuned component."""
from pathlib import Path
import torch

from astra.model.model import Direct8Model
from astra.fine_tuning.fp32 import enable_fp32_training


def load_fine_tuned(path, kwargs, metadata, *, sample_id, task, device='cpu'):
    payload = torch.load(Path(path),map_location='cpu',weights_only=True)
    if (payload.get('format') != 'ASTRA-fine-tuned-v1' or payload.get('kwargs') != kwargs
            or payload.get('base_checkpoint_sha256') != metadata['checkpoint_sha256']
            or payload.get('source_checkpoint_sha256') != metadata['source_checkpoint_sha256']):
        raise ValueError('fine-tuned checkpoint does not match this ASTRA base, panel or architecture')
    scope = payload['fine_tuning']
    if scope.get('sample_id') != sample_id or scope.get('task') != task:
        raise ValueError('fine-tuned checkpoint is specific to its recorded target sample and task')
    if scope.get('image_preprocessing') != 'raw' or scope.get('query_fine_labels_used_for_fitting') is not False:
        raise ValueError('fine-tuned checkpoint has incompatible input/label boundaries')
    model = Direct8Model(**kwargs).to(device).eval()
    model.load_state_dict(payload['model'],strict=True)
    model = enable_fp32_training(model)
    return model,payload


def save_fine_tuned(path, model, metadata, scope):
    state = {k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in model.state_dict().items()}
    payload = dict(format='ASTRA-fine-tuned-v1',kwargs=model.model_kwargs,model=state,
        base_checkpoint_sha256=metadata['checkpoint_sha256'],
        source_checkpoint_sha256=metadata['source_checkpoint_sha256'],fine_tuning=scope)
    with Path(path).open('xb') as stream:
        torch.save(payload,stream)
