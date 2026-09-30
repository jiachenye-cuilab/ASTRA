from contextlib import contextmanager
from dataclasses import dataclass
import json
import numpy as np
from scipy.sparse import csr_matrix
import torch
from astra.data.batching import prefetched_batches
from astra.data.image_features import image_features_from_raw
from astra.fine_tuning.fp32 import canonical_hd16_owner
from astra.fine_tuning.inputs import SpotPacket

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

@dataclass(frozen=True)
class ObservedParentPacket:
    parent_counts: torch.Tensor
    owner_map: torch.Tensor
    image_features_2um: torch.Tensor
    field_valid: torch.Tensor


class TargetInputFields:
    def __init__(self, root, panels, *, device="cpu"):
        self.root, self.panels, self.device = root, panels, torch.device(device)
        self.metadata = read_json(root / "inputs.json")
        m = self.metadata
        if (m["status"] != "complete" or m["input_resolution_um"] != 16
                or m["role"] != "observed16_input" or m["input_gene_ids"] != list(panels.input_gene_ids)
                or m["output_gene_ids"] != list(panels.output_gene_ids)):
            raise ValueError("observed16 cache role, completion or ordered gene identities differ")
        self.resource_id, self.protocol = m["resource_id"], m["protocol_id"]
        self.fields = m["fields"]
        arrays = dict(indptr="counts16_indptr.npy", indices="counts16_indices.npy", data="counts16_data.npy",
            starts="starts_yx.npy", density="density_float32.npy", area="area_uint16.npy", valid="field_valid_bool.npy",
            pathology="features_float32.npy", pathology_valid="pathology_valid_bool.npy")
        for name, filename in arrays.items():
            setattr(self, name, np.load(root / filename, mmap_mode="r", allow_pickle=False))
        n, g = self.fields, len(panels.input_gene_ids)
        if (self.starts.shape != (n, 2) or np.any(self.starts % 8) or self.indptr.shape != (n * 256 + 1,)
                or self.indices.shape != self.data.shape or self.indptr[0] != 0 or self.indptr[-1] != len(self.data)
                or np.any(np.diff(self.indptr) < 0) or self.indices.dtype != np.int32 or self.data.dtype != np.int64
                or np.any(self.indices < 0) or np.any(self.indices >= g) or np.any(self.data < 0)
                or self.density.shape != (n, 128, 128, 3) or self.density.dtype != np.float32
                or self.area.shape != (n, 128, 128) or self.area.dtype != np.uint16
                or self.valid.shape != (n, 128, 128) or self.valid.dtype != np.bool_
                or self.pathology.shape != (n, 1024, 14, 14) or self.pathology.dtype != np.float32
                or self.pathology_valid.shape != (n, 14, 14) or self.pathology_valid.dtype != np.bool_):
            raise ValueError("observed16 sparse/image/UNI geometry or dtype differs")
        self.available = np.asarray(m["gene_available"], dtype=bool)
        if self.available.shape != (g,):
            raise ValueError("availability must follow the complete input gene panel")
        self.counts = csr_matrix((self.data, self.indices, self.indptr), shape=(n * 256, g), copy=False)

    def _indices(self, items, role):
        if role != "target_input":
            raise PermissionError("observed16 cache cannot be reinterpreted as a fitting or fine-label cache")
        indices = []
        for item in items:
            i = item["field_index"]
            if (type(i) is not int or not 0 <= i < self.fields or item["sample"] != self.resource_id
                    or item["protocol_id"] != self.protocol or list(item["start_yx_2um"]) != self.starts[i].tolist()):
                raise PermissionError("requested observed16 field identity differs")
            indices.append(i)
        return np.asarray(indices, dtype=np.int64)

    def prepare(self, items, role):
        indices = self._indices(items, role)
        rows = (indices[:, None] * 256 + np.arange(256)[None]).ravel()
        arrays = (self.counts[rows].toarray().reshape(len(items), 16, 16, -1), self.density[indices],
            self.area[indices], self.valid[indices], np.broadcast_to(self.available, (len(items), len(self.available))).copy(),
            np.full(len(items), self.protocol, dtype=np.int64), self.pathology[indices], self.pathology_valid[indices])
        tensors = [torch.from_numpy(a) for a in arrays]
        return tuple(t.pin_memory() if self.device.type == "cuda" else t for t in tensors)

    def encode(self, prepared):
        counts, density, area, valid, available, protocol, pathology, pathology_valid = (
            t.to(self.device, non_blocking=True) for t in prepared)
        image = image_features_from_raw(density, area, valid).permute(0, 3, 1, 2).contiguous()
        owner = canonical_hd16_owner(device=self.device).owner_map[None].expand(len(counts), -1, -1)
        return ObservedParentPacket(counts.flatten(1, 2), owner, image, valid), available, protocol, (pathology, pathology_valid)


@contextmanager
def target_packets(fields, features, config, items, *, batch_size, role, task):
    if not isinstance(fields, TargetInputFields) or features is not None or task != "HD16":
        raise TypeError("target-only packets require observed16 fields and their own stored image inputs")

    def prepare(begin):
        selected = items[begin:begin + batch_size]
        return selected, fields.prepare(selected, role)

    def encode(prepared):
        selected, values = prepared
        return (selected, *fields.encode(values))

    with prefetched_batches(range(0, len(items), batch_size), prepare, encode,
            pipeline=config["training"].get("batch_pipeline", "cpu"), device=fields.device) as packets:
        iterator = (packet.value for packet in packets)
        try:
            yield iterator
        finally:
            iterator.close()


@dataclass(frozen=True)
class MeasuredPacket(ObservedParentPacket):
    observed_16um: torch.Tensor


class Fields(TargetInputFields):
    def __init__(self, root, panels, *, device):
        super().__init__(root, panels, device=device)
        self.observed = np.load(root / 'observed16_bool.npy', mmap_mode='r', allow_pickle=False)
        assert self.observed.shape == (self.fields, 16, 16) and self.observed.dtype == np.bool_

    def prepare(self, items, role):
        mask = torch.from_numpy(np.array(self.observed[self._indices(items, role)], copy=True))
        return (*super().prepare(items, role), mask.pin_memory() if self.device.type == 'cuda' else mask)

    def encode(self, prepared):
        packet, available, protocol, semantic = super().encode(prepared[:-1])
        packet = MeasuredPacket(packet.parent_counts, packet.owner_map, packet.image_features_2um,
                                packet.field_valid, prepared[-1].to(self.device, non_blocking=True))
        return packet, available, protocol, semantic


class SpotInputFields:
    _indices = TargetInputFields._indices

    def __init__(self, root, panels, *, device='cpu'):
        self.root,self.panels,self.device=root,panels,torch.device(device)
        m=self.metadata=read_json(root/'inputs.json')
        if (m['status']!='complete' or m['role']!='observed_spot55_input'
                or m['input_gene_ids']!=list(panels.input_gene_ids)
                or m['output_gene_ids']!=list(panels.output_gene_ids)):
            raise ValueError('Spot55-only input role or gene ordering differs')
        self.resource_id,self.protocol,self.fields=m['resource_id'],m['protocol_id'],m['fields']
        self.slots=m['parent_slots']
        for name,filename in dict(indptr='counts55_indptr',indices='counts55_indices',data='counts55_data',
                starts='starts_yx',owner='spot_owner_int16',parent_valid='parent_valid_bool',
                density='density_float32',area='area_uint16',valid='field_valid_bool',
                pathology='features_float32',pathology_valid='pathology_valid_bool').items():
            setattr(self,name,np.load(root/(filename+'.npy'),mmap_mode='r',allow_pickle=False))
        n,p,g=self.fields,self.slots,len(panels.input_gene_ids)
        assert self.indptr.shape==(n*p+1,) and self.indptr[0]==0 and self.indptr[-1]==len(self.data)
        assert self.indices.shape==self.data.shape and np.all(np.diff(self.indptr)>=0)
        assert self.data.dtype==np.int64 and self.indices.dtype==np.int32
        assert np.all(self.data>=0) and np.all((self.indices>=0)&(self.indices<g))
        assert self.owner.shape==(n,128,128) and self.parent_valid.shape==(n,p)
        assert self.owner.dtype==np.int16 and self.parent_valid.dtype==np.bool_
        assert np.all((self.owner>=-1)&(self.owner<p)) and self.starts.shape==(n,2)
        assert self.density.shape==(n,128,128,3) and self.density.dtype==np.float32
        assert self.area.shape==self.valid.shape==(n,128,128) and self.valid.dtype==np.bool_
        assert self.pathology.shape==(n,1024,14,14) and self.pathology.dtype==np.float32
        assert self.pathology_valid.shape==(n,14,14)
        self.available=np.asarray(m['gene_available'],dtype=bool)
        assert self.available.shape==(g,)
        self.counts=csr_matrix((self.data,self.indices,self.indptr),shape=(n*p,g),copy=False)

    def prepare(self, items, role):
        indices=self._indices(items,role)
        rows=(indices[:,None]*self.slots+np.arange(self.slots)[None]).ravel()
        arrays=(self.counts[rows].toarray().reshape(len(items),self.slots,-1),
            self.owner[indices].astype(np.int64),self.parent_valid[indices],self.density[indices],
            self.area[indices],self.valid[indices],np.broadcast_to(self.available,(len(items),len(self.available))).copy(),
            np.full(len(items),self.protocol,dtype=np.int64),self.pathology[indices],self.pathology_valid[indices])
        tensors=[torch.from_numpy(x) for x in arrays]
        return tuple(t.pin_memory() if self.device.type=='cuda' else t for t in tensors)

    def encode(self, prepared):
        counts,owner,pv,density,area,valid,available,protocol,uni,uni_valid=(t.to(self.device,non_blocking=True) for t in prepared)
        image=image_features_from_raw(density,area,valid).permute(0,3,1,2).contiguous()
        return SpotPacket(counts,owner,pv,image,valid),available,protocol,(uni,uni_valid)


@contextmanager
def spot_packets(fields, features, config, items, *, batch_size, role, task):
    if not isinstance(fields,SpotInputFields) or features is not None or task!='Spot55':
        raise TypeError('Spot55 adaptation accepts only spot-only input packets')
    def prepare(begin):
        selected=items[begin:begin+batch_size]
        return selected,fields.prepare(selected,role)
    def encode(prepared):
        selected,values=prepared
        return selected,*fields.encode(values)
    with prefetched_batches(range(0,len(items),batch_size),prepare,encode,
            pipeline=config['training'].get('batch_pipeline','cpu'),device=fields.device) as packets:
        iterator=(packet.value for packet in packets)
        try: yield iterator
        finally: iterator.close()
