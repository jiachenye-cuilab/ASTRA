"""ASTRA FP32 inference from prepared whole-section inputs."""
from contextlib import contextmanager
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import numpy as np
import torch

from astra.model.fp32 import enable_fp32
from astra.model.model import Direct8Model
from astra.inference.section_cache import Fields, SpotInputFields, target_packets, spot_packets
from astra.inference.section_export import ExportPlan
from astra.inference.export_diagnostics import section_grid_diagnostics
from astra.fine_tuning.fp32 import target_from_packet
from astra.fine_tuning.spot_objective import model_inputs


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


class Timing:
    def __init__(self, device):
        self.device, self.seconds = torch.device(device), {}

    @contextmanager
    def phase(self, name, cuda=False):
        if cuda and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if cuda and self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter()-started


def open_fields(root, genes, metadata, device):
    root = Path(root).resolve()
    info = read(root/'inputs.json')
    if (info.get('uni_checkpoint_sha256') != metadata['uni_checkpoint_sha256']
            or info.get('preprocessing_mode') != 'raw'
            or info.get('contains_fine_expression_arrays') is not False
            or info.get('protocol_id') not in (0, 1)):
        raise ValueError('section cache must contain raw H&E, matching UNI1, assay identity and coarse observations only')
    task = {'observed16_input':'HD16', 'observed_spot55_input':'Spot55'}.get(info.get('role'))
    if task is None:
        raise ValueError('unsupported prepared section role')
    geometry = info.get('inference_geometry')
    allowed = [dict(field_um=256, core_um=160, stride_um=160, context_margin_um=48)] if task == 'HD16' else [
        dict(field_um=256, core_um=160, stride_um=144, context_margin_um=48),
        dict(field_um=256, core_um=144, stride_um=144, context_margin_um=56)]
    if geometry not in allowed:
        raise ValueError('cache inference_geometry must identify a supported core and stride')
    panels = SimpleNamespace(input_gene_ids=genes, output_gene_ids=genes, output_indices=np.arange(len(genes)))
    fields = (Fields if task == 'HD16' else SpotInputFields)(root, panels, device=device)
    if not fields.fields or not fields.available.any():
        raise ValueError('empty section or no measured panel genes')
    fields.task = task
    return fields


def items_for(fields):
    return [dict(field_index=i, sample=fields.resource_id, protocol_id=fields.protocol,
                 start_yx_2um=start.tolist()) for i, start in enumerate(fields.starts)]


class FineTuningFields:
    """Use exactly the inference cache, with whole FOVs on either side of a fixed boundary."""
    def __init__(self, manifest_path, genes, metadata, device, *, smoke=False):
        path = Path(manifest_path).resolve()
        manifest = read(path)
        cache = Path(manifest['prepared_section'])
        if cache.is_absolute():
            raise ValueError('prepared_section must be relative to the manifest')
        self.cache = open_fields(path.parent/cache, genes, metadata, device)
        self.task, self.sample_id = self.cache.task, self.cache.resource_id
        if manifest.get('sample_id') != self.sample_id or manifest.get('task') != self.task:
            raise ValueError('manifest sample/task differs from the prepared section')
        self.device, self.metadata = self.cache.device, self.cache.metadata
        expected = dict(field_um=256,core_um=160,stride_um=160 if self.task=='HD16' else 144,context_margin_um=48)
        if self.metadata['inference_geometry'] != expected:
            raise ValueError('latest fine-tuning requires HD16 160/160 or Spot55 160/144 geometry')
        self.geometry = expected
        starts = self.cache.starts
        # Require the physical split explicitly: a new grid must not move a previous split.
        boundary = manifest['boundary_x_2um']
        if type(boundary) is not int or boundary < 0:
            raise ValueError('boundary_x_2um must be an explicit nonnegative integer')
        self.boundary_x_2um = boundary
        if self.task == 'HD16':
            known = self.cache.observed
            usable = known.reshape(-1,8,2,8,2).sum((2,4)) >= 2
            active = known & usable.repeat(2,axis=1).repeat(2,axis=2)
            eligible = active[:,3:13,3:13].any((1,2))
        else:
            eligible = np.asarray([len(p) >= 2 for p in self.metadata['neighbor_pairs']])
        indices = dict(support=np.flatnonzero(eligible & (starts[:,1]+128 <= boundary)),
                       selection=np.flatnonzero(eligible & (starts[:,1] >= boundary)))
        items = items_for(self.cache)
        self.groups = {role:[items[int(i)] for i in ids] for role,ids in indices.items()}
        minimum = 16 if smoke else 64
        if len(self.groups['support']) < minimum or not self.groups['selection']:
            raise ValueError(f'physical split needs at least {minimum} eligible support FOVs and a nonempty selection region')

    def prepare(self, items, role='target_input'):
        return self.cache.prepare(items, role)

    def encode(self, prepared):
        return self.cache.encode(prepared)


def predict_section(inputs, output, package, *, device='cpu', batch_size=4, pipeline='device', fine_tuned=None):
    from astra.runtime import sha256
    from astra.fine_tuning.checkpoint import load_fine_tuned
    package, output = Path(package), Path(output)
    if read(Path(inputs)/'inputs.json').get('role') == 'observed_st100_input':
        if fine_tuned:
            raise ValueError('ST100 paper workflow uses frozen weights; target adaptation is not validated')
        from astra.inference.st100 import predict
        return predict(inputs,output,package,device=device,batch_size=batch_size)
    if output.exists():
        raise FileExistsError('output directory exists; choose a new path')
    if batch_size < 1 or pipeline not in ('cpu','device'):
        raise ValueError('invalid batch size or prefetch pipeline')
    started = time.perf_counter()
    timer = Timing(device)
    metadata = read(package/'model/metadata.json')
    with timer.phase('model_and_inputs', cuda=True):
        for name in ('checkpoint.pt','config.json','input_gene_ids.json','output_gene_ids.json'):
            if sha256(package/'model'/name) != metadata[Path(name).stem+'_sha256']:
                raise ValueError(f'package identity mismatch: {name}')
        genes = read(package/'model/input_gene_ids.json')
        if genes != read(package/'model/output_gene_ids.json'):
            raise ValueError('section export requires the registered identical input/output panel')
        fields = open_fields(inputs, genes, metadata, device)
        config = read(package/'model/config.json')
        if fine_tuned:
            model, payload = load_fine_tuned(fine_tuned, config['kwargs'], metadata,
                sample_id=fields.resource_id, task=fields.task, device=device)
            scope = payload['fine_tuning']
            if scope.get('geometry') is not None and scope['geometry'] != fields.metadata['inference_geometry']:
                raise ValueError('fine-tuning and inference geometry differ')
        else:
            model = Direct8Model(**config['kwargs']).to(device).eval()
            model.load_state_dict(torch.load(package/'model/checkpoint.pt', map_location='cpu', weights_only=True), strict=True)
            model = enable_fp32(model)
            scope = None
        items = items_for(fields)
        indices = list(range(fields.fields))
        plan = ExportPlan(fields.task, fields, indices, batch_size=batch_size)
        coordinates_name = 'parent_yx_16um.npy' if fields.task == 'HD16' else 'query_yx_8um.npy'
        coordinates = np.load(fields.root/coordinates_name, allow_pickle=False)
        if not np.array_equal(plan.unique, np.arange(len(coordinates))):
            raise ValueError('export assignments do not cover every requested coordinate')
        if fields.task == 'Spot55' and not np.allclose(plan.expected_weight, 1., atol=1e-12, rtol=0):
            raise ValueError('overlapping core weights must sum to one')
    output.mkdir(parents=True)
    prediction = np.lib.format.open_memmap(output/'prediction.npy', mode='w+', dtype=np.float32, shape=plan.shape(len(genes)))
    prediction[:] = 0
    accumulated = np.zeros(len(plan.unique))
    error = 0.
    source = dict(training=dict(batch_pipeline=pipeline))
    packets = target_packets if fields.task == 'HD16' else spot_packets
    with timer.phase('prediction_and_export', cuda=True), torch.no_grad():
        with packets(fields,None,source,items,batch_size=batch_size,role='target_input',task=fields.task) as batches:
            for selected,packet,available,protocol,semantic in batches:
                model_input = (target_from_packet(packet,available,protocol,semantic).model_inputs(parent_bin_um=16)
                               if fields.task == 'HD16' else model_inputs(packet,available,protocol,semantic))
                result = model(**model_input)
                error = max(error,plan.compact(result,model_input,packet,model,
                    [i['field_index'] for i in selected],prediction,accumulated))
                del result,model_input,packet,available,protocol,semantic
        if fields.task == 'Spot55' and not np.array_equal(accumulated, plan.expected_weight):
            raise ValueError('not every overlapping core contribution was exported')
        prediction.flush()
        np.save(output/coordinates_name, coordinates)
        np.save(output/'gene_available.npy', fields.available)
        write(output/'gene_ids.json', genes)
    with timer.phase('serialized_export_diagnostics'):
        serialized_export = section_grid_diagnostics(output, fields)
    report = dict(status='complete',model_name='ASTRA',sample_id=fields.resource_id,task=fields.task,
        arithmetic_dtype='float32',preprocessing_mode='raw',geometry=fields.metadata['inference_geometry'],
        fields=fields.fields,batch_size=batch_size,cpu_threads=torch.get_num_threads(),batch_pipeline=pipeline,
        output_shape=list(prediction.shape),output_dtype=str(prediction.dtype),device=str(device),
        base_checkpoint_sha256=metadata['checkpoint_sha256'],fine_tuning=scope,
        checkpoint_sha256=sha256(fine_tuned) if fine_tuned else metadata['checkpoint_sha256'],
        parent_conservation_max_scaled=error,parent_conservation_scope='pre_export_allocation',
        export_conservation_tolerance=1e-5 if fields.task == 'HD16' else None,
        serialized_export=serialized_export,
        elapsed_seconds=time.perf_counter()-started,phase_seconds=timer.seconds,
        timing_scope='Prepared-cache loading, model, batched inference, compact export, array flush and serialized-grid diagnostics; excludes cache preparation and fine-tuning.',
        preparation_report='inputs/preparation.json (if built by python -m astra prepare-section); report its cost separately and once',
        peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        query_fine_labels_used_for_fitting=False)
    if torch.device(device).type == 'cuda':
        report.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
    write(output/'report.json',report)
    return report
