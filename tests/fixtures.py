"""Deterministic synthetic engineering inputs; no biological or UNI measurements."""
import json
from pathlib import Path
import shutil

import numpy as np
from scipy import sparse
import torch

from astra.assets import ROOT
from astra.model.model import Direct8Model
from astra.runtime import sha256
from astra.inference.section_geometry import tile_plan, prediction_tiles, spot_counts


def write_json(path, value):
    Path(path).write_text(json.dumps(value) + '\n', encoding='utf-8')


def synthetic_package(root):
    root = Path(root)
    (root / 'model').mkdir(parents=True)
    genes = ['synthetic_A', 'synthetic_B']
    config = json.loads((ROOT / 'model/config.json').read_text())
    config['kwargs'].update(input_gene_ids=genes, output_gene_ids=genes)
    torch.manual_seed(930)
    model = Direct8Model(**config['kwargs'])
    torch.save(model.state_dict(), root / 'model/checkpoint.pt')
    for name, value in [('config.json', config), ('input_gene_ids.json', genes), ('output_gene_ids.json', genes)]:
        write_json(root / 'model' / name, value)
    shutil.copyfile(ROOT/'model/uni_config.json',root/'model/uni_config.json')
    metadata = dict(uni_checkpoint_sha256='synthetic_features_not_UNI',
                    **{Path(n).stem+'_sha256': sha256(root / 'model' / n) for n in
                       ['checkpoint.pt', 'config.json', 'input_gene_ids.json', 'output_gene_ids.json','uni_config.json']})
    write_json(root / 'model/metadata.json', metadata)
    return root, genes, metadata


def synthetic_section(root, genes, metadata, task, *, core=160, partial=False):
    root = Path(root)
    root.mkdir()
    shape = np.array([128, 128])
    geometry = dict(field_um=256, core_um=core, stride_um=160 if task == 'HD16' else 144,
                    context_margin_um=(256-core)//2)
    if task == 'HD16':
        coords = np.array([[0,0], [0,9], [0,10], [9,9], [15,15]], np.int64)
        counts = np.arange(1, len(coords)*len(genes)+1).reshape(len(coords), len(genes))
        starts, keep, assignment, local = tile_plan(coords, shape, core_um=core)
        arrays = dict(parent_yx_16um=coords[keep], assignment=assignment, local_parent=local)
        rows = np.zeros((len(starts), 256, len(genes)), np.int64)
        observed = np.zeros((len(starts), 16, 16), bool)
        lookup = {tuple(point): i for i, point in enumerate(coords)}
        for f, origin in enumerate(starts):
            for y in range(16):
                for x in range(16):
                    slot = lookup.get(tuple(origin//8 + [y,x]))
                    if slot is not None:
                        rows[f, y*16+x] = counts[slot]
                        observed[f,y,x] = True
        matrix = sparse.csr_matrix(rows.reshape(-1, len(genes)))
        arrays.update(observed16_bool=observed, counts16_indptr=matrix.indptr.astype(np.int64),
                      counts16_indices=matrix.indices.astype(np.int32), counts16_data=matrix.data.astype(np.int64))
        extra = dict(input_resolution_um=16)
    else:
        yy, xx = np.indices((32,32))
        queries = np.column_stack((yy.ravel(), xx.ravel()))
        if partial:
            queries = queries[(queries[:,0] < 14) & (queries[:,1] < 14)]
        starts, query, assignment, weights = prediction_tiles(queries, shape, geometry)
        arrays = dict(query_yx_8um=queries, query_index=query, query_assignment=assignment, query_weight=weights)
        extra = {}
    n = len(starts)
    yy, xx = np.indices((128,128))
    valid = np.stack([(yy+y >= 0) & (yy+y < shape[0]) & (xx+x >= 0) & (xx+x < shape[1]) for y,x in starts])
    arrays.update(starts_yx=starts, field_valid_bool=valid,
                  density_float32=np.ones((n,128,128,3),np.float32), area_uint16=np.full((n,128,128),65535,np.uint16),
                  features_float32=np.zeros((n,1024,14,14),np.float32), pathology_valid_bool=np.ones((n,14,14),bool))
    for name, value in arrays.items():
        np.save(root / (name + '.npy'), value)
    available = np.ones(len(genes), bool)
    if task == 'Spot55':
        obs = root / 'observations'
        obs.mkdir()
        np.save(obs / 'spot_yx_um.npy', np.array([[72.,72.], [144.,144.], [5.,5.], [600.,600.]]))
        np.save(obs / 'parent_counts.npy', np.array([[19,37],[53,71],[7,11],[13,17]], np.int64))
        np.save(obs / 'gene_available.npy', available)
        _, slots, pairs = spot_counts(obs, root, starts)
        extra.update(parent_slots=slots, neighbor_pairs=pairs)
    write_json(root / 'inputs.json', dict(status='complete', fields=n, resource_id='synthetic_section', protocol_id=1,
        role='observed16_input' if task == 'HD16' else 'observed_spot55_input', input_gene_ids=genes, output_gene_ids=genes,
        gene_available=available.tolist(), uni_checkpoint_sha256=metadata['uni_checkpoint_sha256'],
        contains_fine_expression_arrays=False, preprocessing_mode='raw', inference_geometry=geometry, **extra))
    return root
