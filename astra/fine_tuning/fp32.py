"""Registered FP32 fine-tuning with the existing observed-only training contract."""
import inspect
from pathlib import Path
import textwrap
import types

import torch

from astra.model.fp32 import CONSERVATION_TOLERANCE, allocate_fp32, compact_pool, compile_like, enable_fp32, replace_once
from astra.fine_tuning import hd_losses as adaptation_losses, observations
from astra.fine_tuning.hd_data import TargetBatch
from astra.fine_tuning import hd_objective as hd, spot_objective as spot, objectives as train


def fp32_function(function, **globals_):
    source = textwrap.dedent(inspect.getsource(function))
    source = source.replace('.double()', '.float()').replace('torch.float64', 'torch.float32')
    return compile_like(function, source, **globals_)


def enable_fp32_training(model):
    """Use differentiable operations with gradients; use storage reuse under no_grad."""
    model = enable_fp32(model)
    original = type(model).forward
    code = textwrap.dedent(inspect.getsource(original))
    start = code.index('    global_density = counts.sum(1)')
    end = code.index('    cell_residual, raw_exposure = self._prior_residual(hidden)')
    code = code[:start] + '''    global_density = counts.sum(1) / geometry.area_children.sum(1).clamp_min(1)[:, None]
    parent_slots = density.shape[1]
    if gap_prior_mode == "global":
        density_table = torch.cat((density, global_density[:, None]), dim=1)
        gap_index = torch.full_like(owner, parent_slots)
    else:
        query_y, query_x = shape[1] // 4, shape[2] // 4
        density_table = torch.cat((density, context["local_density"].flatten(1, 2), global_density[:, None]), dim=1)
        local_index = torch.arange(query_y * query_x, device=owner.device).reshape(1, query_y, query_x) + parent_slots
        gap_index = torch.where(support, local_index, parent_slots + query_y * query_x)
        gap_index = gap_index.repeat_interleave(4, 1).repeat_interleave(4, 2)
    prior_index = torch.where(owner >= 0, owner, gap_index)
    base_mass = self._gather_parent(density_table, prior_index).reshape(*shape, self.num_genes).clamp_min(self.minimum_prior_mass)
    del density_table, prior_index, gap_index
''' + code[end:]
    code = code.replace('torch.float64', 'torch.float32')
    code = replace_once(code,
        '    prior = base_mass * torch.exp(cell_residual.to(torch.float32) + exposure[None, None, None].to(torch.float32))',
        '    prior = base_mass * torch.exp(cell_residual + exposure[None, None, None])\n    del base_mass, cell_residual')
    code = replace_once(code, '    segment_prior = pool_segment_mass(prior, layout)',
        '    segment_prior = pool_segment_mass(prior, layout)\n    del prior')
    training_forward = compile_like(original, code, pool_segment_mass=compact_pool, allocate=allocate_fp32)
    inference_forward = model.forward.__func__

    def forward(self, **inputs):
        if self.gap_prior_mode != 'local':
            raise ValueError('This implementation supports the registered local GAP prior')
        inputs['parent_counts'] = inputs['parent_counts'].float()
        return (training_forward if torch.is_grad_enabled() else inference_forward)(self, **inputs)

    model.forward = types.MethodType(forward, model)
    return model


def poisson_divergence(teacher, student):
    """FP32 count KL with a stable evaluation around equal teacher/student counts."""
    assert teacher.dtype == student.dtype == torch.float32
    delta = student - teacher
    near = (teacher >= 1e-12) & (student >= 1e-12) & (delta.abs() <= .5 * teacher)
    ratio = torch.where(near, delta, 0.) / torch.where(near, teacher, 1.)
    # r-log1p(r) loses significant digits near zero. For |r|<=1/8, the
    # degree-10 series remainder is <=|r|**11/(11*(1-|r|)), below FP32 resolution.
    polynomial = 1./10
    for degree in range(9, 1, -1):
        polynomial = (-1. if degree % 2 else 1.) / degree + ratio * polynomial
    series = ratio.square() * polynomial
    local = teacher * torch.where(ratio.abs() <= .125, series, ratio - torch.log1p(ratio))
    direct = teacher * (teacher.clamp_min(1e-12).log() - student.clamp_min(1e-12).log()) + delta
    return torch.where(near, local, direct)


class TargetBatchFP32(TargetBatch):
    """Keep observed-parent aggregation in FP32 as well as model arithmetic."""


def canonical_hd16_owner(*, field_shape=(128,128), device='cpu'):
    # On the registered aligned 2um raster, each 16um square is exactly 8x8
    # integer cells. Avoid rebuilding the generic floating-point square program.
    rows,columns=field_shape
    if rows%8 or columns%8 or min(rows,columns)<=0:
        raise ValueError('The FP32 adapter requires an aligned 16um parent grid')
    yy=torch.arange(rows,device=device,dtype=torch.long)[:,None]//8
    xx=torch.arange(columns,device=device,dtype=torch.long)[None]//8
    return types.SimpleNamespace(owner_map=yy*(columns//8)+xx)


TargetBatchFP32.model_inputs = fp32_function(TargetBatch.model_inputs,
    build_target_observations=fp32_function(observations.build_target_observations,
        canonical_hd16_owner=canonical_hd16_owner))
target_from_packet = fp32_function(hd.target_from_packet, TargetBatch=TargetBatchFP32,
    canonical_hd16_owner=canonical_hd16_owner)
poisson_per_umi = fp32_function(adaptation_losses._poisson_per_umi)
target_self_supervision = fp32_function(adaptation_losses.target_self_supervision, _poisson_per_umi=poisson_per_umi)

code = textwrap.dedent(inspect.getsource(adaptation_losses.source_allocation_kl)).replace('.double()', '.float()')
code = replace_once(code,
    '    terms = teacher * (teacher.clamp_min(1e-12).log() - student.clamp_min(1e-12).log()) + student - teacher',
    '    terms = poisson_divergence(teacher, student)')
source_allocation_kl = compile_like(adaptation_losses.source_allocation_kl, code, poisson_divergence=poisson_divergence)
hd_objective = fp32_function(hd.objective, target_self_supervision=target_self_supervision)
hd_preservation = fp32_function(hd.preservation_objective, source_allocation_kl=source_allocation_kl)
spot_objective = fp32_function(spot.objective)
code = textwrap.dedent(inspect.getsource(spot.preservation_objective)).replace('.double()', '.float()')
code = replace_once(code,
    '    terms=q*(q.clamp_min(1e-12).log()-p.clamp_min(1e-12).log())+p-q',
    '    terms=poisson_divergence(q,p)')
spot_preservation = compile_like(spot.preservation_objective, code, poisson_divergence=poisson_divergence)
data_objective = fp32_function(train.data_objective, target_from_packet=target_from_packet,
    hd_objective=hd_objective, spot_objective=spot_objective)
protection = fp32_function(train.protection, target_from_packet=target_from_packet,
    hd_preservation=hd_preservation, spot_preservation=spot_preservation)
selection_score = fp32_function(train.selection_score, data_objective=data_objective)
