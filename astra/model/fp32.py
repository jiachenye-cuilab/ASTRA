"""FP32 frozen inference with unchanged weights, geometry and allocation formula."""
import inspect
import textwrap
import types

import torch

from astra.model.allocation import allocate

CONSERVATION_TOLERANCE = 5e-6


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Source changed; review the optimization before running: ' + before[:100])
    return source.replace(before, after)


def compile_like(original, source, **additions):
    namespace = dict(original.__globals__, **additions)
    exec(compile(source, __file__, 'exec'), namespace)
    return namespace[original.__name__]


def compact_pool(values, layout):
    assert values.dtype == torch.float32
    flat = values.reshape(-1, values.shape[-1])
    selected = flat if layout.valid_cell_indices.numel() == flat.shape[0] else flat.index_select(0, layout.valid_cell_indices)
    inverse = layout.cell_to_segment.flatten().index_select(0, layout.valid_cell_indices)
    return values.new_zeros((layout.segments, values.shape[-1])).index_add_(0, inverse, selected)


allocate_fp32 = compile_like(allocate, inspect.getsource(allocate).replace('torch.float64', 'torch.float32'))


def enable_fp32(model):
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # The checkpoint's historical FP64 tolerance is not representable as a general FP32 bound.
    model.conservation_tolerance = CONSERVATION_TOLERANCE
    inputs = model._inputs.__func__
    model._inputs = types.MethodType(compile_like(inputs,
        textwrap.dedent(inspect.getsource(inputs)).replace('torch.float64', 'torch.float32')), model)
    original = model.forward.__func__
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
    base_mass = self._gather_parent(density_table, prior_index).reshape(*shape, self.num_genes)
    del density_table, prior_index, gap_index
    base_mass.clamp_min_(self.minimum_prior_mass)
''' + code[end:]
    code = replace_once(code,
        '    cell_residual = self.residual_bound * torch.tanh(cell_residual / self.residual_bound)',
        '    cell_residual.div_(self.residual_bound).tanh_().mul_(self.residual_bound)')
    code = replace_once(code,
        '    prior = base_mass * torch.exp(cell_residual.to(torch.float64) + exposure[None, None, None].to(torch.float64))\n    prior = torch.where(valid_field[..., None] & available[:, None, None], prior, 0.0)',
        '    prior = cell_residual\n    del cell_residual\n    prior.add_(exposure[None, None, None]).exp_().mul_(base_mass)\n    del base_mass\n    prior.masked_fill_(~(valid_field[..., None] & available[:, None, None]), 0.0)')
    code = replace_once(code, '    segment_prior = pool_segment_mass(prior, layout)',
        '    segment_prior = pool_segment_mass(prior, layout)\n    del prior')
    forward = compile_like(original, code, pool_segment_mass=compact_pool, allocate=allocate_fp32)

    def inference_only(self, **inputs):
        if torch.is_grad_enabled():
            raise RuntimeError('FP32 inference storage reuse requires no_grad; training is a separate validation')
        if self.gap_prior_mode != 'local':
            raise ValueError('ASTRA requires the local gap prior')
        inputs['parent_counts'] = inputs['parent_counts'].float()
        return forward(self, **inputs)

    model.forward = types.MethodType(inference_only, model)
    return model
