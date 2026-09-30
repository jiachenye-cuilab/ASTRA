"""Prepared FOV input adapter; no target fine-expression fields are read."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from astra.model.ownership import validate_owner_batch
from astra.fine_tuning.geometry import canonical_hd16_owner
from astra.data.batching import prefetched_batches


@dataclass(frozen=True)
class SpotPacket:
    parent_counts: torch.Tensor
    owner_map: torch.Tensor
    parent_valid: torch.Tensor
    image_features_2um: torch.Tensor
    field_valid: torch.Tensor
    observed_16um: torch.Tensor | None = None


KEYS = ('parent_counts', 'gene_ids', 'gene_available', 'owner_map', 'parent_valid',
        'field_valid', 'protocol_id', 'image_features_2um', 'pathology_features',
        'pathology_valid', 'uni_checkpoint_sha256', 'parent_centers_yx_um')


class Fields:
    def __init__(self, manifest_path, genes, metadata, device, *, smoke=False):
        from astra.data.inputs import load_input, semantic_input
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text(encoding='utf-8'))
        self.task = manifest['task']
        self.sample_id = manifest['sample_id']
        if self.task not in ('HD16', 'Spot55') or not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError('manifest needs a nonempty sample_id and task HD16 or Spot55')
        if not manifest.get('support') or not manifest.get('selection'):
            raise ValueError('provide separate spatial support and selection FOVs')
        self.device = torch.device(device)
        self.paths, self.origins, self.groups = [], [], {}
        for role in ('support', 'selection'):
            self.groups[role] = []
            for item in manifest[role]:
                value = Path(item['input'])
                if value.is_absolute():
                    raise ValueError('manifest input paths must be relative to the manifest')
                self.paths.append((path.parent/value).resolve())
                origin = np.asarray(item['origin_yx_um'], dtype=np.float64)
                if origin.shape != (2,) or not np.isfinite(origin).all():
                    raise ValueError('each FOV needs its registered global origin_yx_um')
                self.origins.append(origin)
                self.groups[role].append(dict(field_index=len(self.paths)-1, sample=self.sample_id))
        minimum = 16 if smoke else 64
        if len(self.groups['support']) < minimum or len(set(self.paths)) != len(self.paths):
            raise ValueError(f'need at least {minimum} distinct support inputs, with no repeated file paths')
        support_origins = np.asarray(self.origins[:len(self.groups['support'])])
        for origin in self.origins[len(self.groups['support']):]:
            if np.any(np.all(np.abs(support_origins-origin) < 256, axis=1)):
                raise ValueError('support and selection FOVs overlap; split physical regions before fitting')
        self.genes, self.identity = genes, metadata
        self.metadata = dict(resource_id=self.sample_id, neighbor_pairs=[])
        first = None
        for i in range(len(self.paths)):
            data = load_input(self.paths[i], genes, keys=KEYS)
            if 'pathology_features' not in data:
                raise ValueError('fine-tuning requires precomputed frozen UNI features for every FOV')
            semantic_input(data, torch.device('cpu'), metadata, None)
            validate_owner_batch(torch.from_numpy(data['owner_map'])[None],
                torch.from_numpy(data['parent_valid'])[None], torch.from_numpy(data['field_valid'])[None])
            observed = data['parent_valid'][:, None] & data['gene_available'][None]
            counts = data['parent_counts'][observed]
            if not np.equal(counts, np.round(counts)).all():
                raise ValueError('fine-tuning needs raw integer UMI counts, not normalized expression')
            identity = (int(data['protocol_id']), data['gene_available'])
            if first is not None and (identity[0] != first[0] or not np.array_equal(identity[1], first[1])):
                raise ValueError('all target FOVs must share one assay and measured gene panel')
            first = identity
            pairs = self.validate_geometry(data)
            self.metadata['neighbor_pairs'].append(pairs)

    def validate_geometry(self, data):
        if self.task == 'HD16':
            canonical = canonical_hd16_owner().owner_map.numpy()
            if data['parent_counts'].shape[0] != 256:
                raise ValueError('HD16 requires 256 row-major parent slots on the aligned 16um grid')
            valid = data['parent_valid'].reshape(16,16)
            complete = data['field_valid'].reshape(16,8,16,8).all(axis=(1,3))
            if np.any(valid & ~complete):
                raise ValueError('HD16 observed parents must have complete image support')
            expected = np.where(np.repeat(np.repeat(valid,8,axis=0),8,axis=1), canonical, -1)
            if not np.array_equal(data['owner_map'], expected):
                raise ValueError('HD16 ownership must follow the measured canonical 16um grid')
            return []
        centers = data.get('parent_centers_yx_um')
        if centers is None or centers.shape != (len(data['parent_valid']),2):
            raise ValueError('Spot55 requires local parent_centers_yx_um[P,2] from measured geometry')
        keep = data['parent_valid']
        if not np.isfinite(centers[keep]).all() or np.any((centers[keep] < 27.5) | (centers[keep] > 228.5)):
            raise ValueError('only complete 55um spots may supply fitting counts')
        axis = (np.arange(128)+.5)*2
        expected = np.full((128,128),-1,dtype=np.int64)
        for i in np.flatnonzero(keep):
            mask = (axis[:,None]-centers[i,0])**2 + (axis[None,:]-centers[i,1])**2 <= 27.5**2
            if np.any(expected[mask] >= 0):
                raise ValueError('measured spot supports overlap')
            expected[mask] = i
        if not np.array_equal(data['owner_map'],expected):
            raise ValueError('Spot55 owners must match complete measured spot supports')
        distance = np.linalg.norm(centers[:,None]-centers[None],axis=-1)
        eligible = (distance > 0) & (distance <= 110) & keep[:,None] & keep[None]
        pairs = np.argwhere(np.triu(eligible,k=1)).tolist()
        if len(pairs) < 2:
            raise ValueError('Spot55 needs at least two geometry-only neighboring pairs per FOV')
        return pairs

    def prepare(self, items, role='target_input'):
        from astra.data.inputs import load_input
        data = [load_input(self.paths[i['field_index']],self.genes,keys=KEYS) for i in items]
        slots = max(len(d['parent_valid']) for d in data)
        for d in data:
            pad = slots-len(d['parent_valid'])
            d['parent_counts'] = np.pad(d['parent_counts'],((0,pad),(0,0)))
            d['parent_valid'] = np.pad(d['parent_valid'],(0,pad))
        names = ('parent_counts','owner_map','parent_valid','image_features_2um','field_valid',
                 'gene_available','protocol_id','pathology_features','pathology_valid')
        tensors = {k:torch.from_numpy(np.stack([d[k] for d in data])) for k in names}
        return {k:t.pin_memory() if self.device.type=='cuda' else t for k,t in tensors.items()}

    def encode(self, prepared):
        t = {k:v.to(self.device,non_blocking=True) for k,v in prepared.items()}
        batch_size = len(t['parent_counts'])
        observed = t['parent_valid'].reshape(batch_size,16,16) if self.task=='HD16' else None
        # HD target_from_packet checks canonical ownership; missing bins are carried separately.
        owner = (canonical_hd16_owner(device=self.device).owner_map[None].expand(batch_size,-1,-1)
                 if self.task=='HD16' else t['owner_map'])
        packet = SpotPacket(t['parent_counts'],owner,t['parent_valid'],t['image_features_2um'].float(),
                            t['field_valid'],observed)
        return packet,t['gene_available'],t['protocol_id'],(t['pathology_features'],t['pathology_valid'])

    def batch(self, items):
        return self.encode(self.prepare(items))


@contextmanager
def packets(fields, features, config, items, *, batch_size, role, task):
    if features is not None or role != 'target_input' or task != fields.task:
        raise ValueError('fine-tuning accepts only target coarse observations and frozen UNI features')
    def prepare(begin):
        selected = items[begin:begin+batch_size]
        return selected,fields.prepare(selected,role)
    def encode(prepared):
        selected,values = prepared
        return selected,*fields.encode(values)
    with prefetched_batches(range(0,len(items),batch_size),prepare,encode,
            pipeline=config.get('training',{}).get('batch_pipeline','device'),device=fields.device) as batches:
        iterator = (batch.value for batch in batches)
        try:
            yield iterator
        finally:
            iterator.close()
