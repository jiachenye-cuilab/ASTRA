"""Prepare ASTRA whole-section inputs from measured coarse observations."""
from pathlib import Path
import resource
import time

import numpy as np
from scipy import sparse
import torch

from astra.inference.section import Timing, read, write
from astra.inference.section_geometry import tile_plan, prediction_tiles, spot_counts
from astra.inference.section_prepare_images import prepare_images
from astra.data.image_readers import registered_transform


def coordinates(path, *, integer):
    values = np.load(path, allow_pickle=False)
    if (values.ndim != 2 or values.shape[1] != 2 or not len(values)
            or not np.isfinite(values).all() or np.any(values < 0)
            or (integer and values.dtype.kind not in 'iu')
            or len(np.unique(values,axis=0)) != len(values)):
        raise ValueError('coordinates must be unique finite nonnegative [N,2] y,x positions')
    return values.astype(np.int64 if integer else np.float64)


def counts_in_panel(path, gene_path, panel):
    genes = read(gene_path)
    if not isinstance(genes,list) or len(set(genes)) != len(genes) or not all(isinstance(g,str) for g in genes):
        raise ValueError('gene_ids must be a JSON array of unique strings')
    counts = sparse.load_npz(path) if path.suffix == '.npz' else sparse.csr_matrix(np.load(path,mmap_mode='r',allow_pickle=False))
    counts = counts.tocsr()
    if (counts.shape[1] != len(genes) or not np.isfinite(counts.data).all()
            or np.any(counts.data < 0) or not np.equal(counts.data,np.round(counts.data)).all()
            or np.any(counts.data >= np.iinfo(np.int64).max)):
        raise ValueError('counts must be nonnegative integer raw UMI values in gene_ids order')
    counts = counts.astype(np.int64)
    lookup = {g:i for i,g in enumerate(genes)}
    available = np.asarray([g in lookup for g in panel])
    if not available.any():
        raise ValueError('no measured genes intersect the ASTRA panel')
    columns = np.asarray([lookup[g] for g in panel if g in lookup])
    measured = counts[:,columns].tocoo()
    result = sparse.coo_matrix((measured.data,(measured.row,np.flatnonzero(available)[measured.col])),
                              shape=(counts.shape[0],len(panel))).tocsr()
    result.eliminate_zeros()
    return result,available


def prepare(config_path, output, package, *, device='cpu', uni_batch_size=16):
    from astra.runtime import sha256
    config_path,output,package = Path(config_path).resolve(),Path(output).resolve(),Path(package)
    if output.exists():
        raise FileExistsError('prepared output exists; choose a new directory')
    cfg = read(config_path)
    if cfg['task'] not in ('HD16','Spot55') or not isinstance(cfg['sample_id'],str) or not cfg['sample_id'].strip():
        raise ValueError('provide task HD16 or Spot55 and a nonempty sample_id')
    if type(cfg['protocol_id']) is not int or cfg['protocol_id'] not in (0,1) or uni_batch_size < 1:
        raise ValueError('invalid protocol_id or UNI batch size')
    shape = np.asarray(cfg['capture_shape_yx_2um'])
    transform = registered_transform(cfg)
    if (shape.shape != (2,) or shape.dtype.kind not in 'iu' or np.any(shape < 128)
            or transform.shape != (3,3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-12):
        raise ValueError('provide capture shape [rows,columns] and an invertible 3x3 registration matrix')
    def path(key):
        value = Path(cfg[key])
        return (config_path.parent/value).resolve()
    metadata = read(package/'model/metadata.json')
    for name in ('input_gene_ids.json','output_gene_ids.json','uni_config.json'):
        if sha256(package/'model'/name) != metadata[Path(name).stem+'_sha256']:
            raise ValueError(f'package identity mismatch: {name}')
    if 'image_cache' not in cfg and sha256(path('uni_checkpoint')) != metadata['uni_checkpoint_sha256']:
        raise ValueError('local UNI1 checkpoint identity differs from the published model')
    genes = read(package/'model/input_gene_ids.json')
    core = cfg.get('core_um',160)
    stride = cfg.get('stride_um',160 if cfg['task']=='HD16' else 144)
    if (cfg['task']=='HD16' and (core,stride)!=(160,160)) or (cfg['task']=='Spot55' and (core,stride) not in ((160,144),(144,144))):
        raise ValueError('supported geometry is HD16 160/160 or Spot55 160/144 (historical option 144/144)')
    geometry = dict(field_um=256,core_um=core,stride_um=stride,context_margin_um=(256-core)//2)
    timer = Timing(device)
    started = time.perf_counter()
    with timer.phase('input_preparation'):
        counts,available = counts_in_panel(path('counts'),path('gene_ids'),genes)
        coords = coordinates(path('parent_yx_16um' if cfg['task']=='HD16' else 'spot_yx_um'),integer=cfg['task']=='HD16')
        if len(coords) != counts.shape[0]:
            raise ValueError('count rows and observed coordinates differ')
        if cfg['task']=='HD16':
            starts,keep,assignment,local = tile_plan(coords,shape,core_um=core)
            if len(keep) != len(coords):
                raise ValueError('every input HD16 parent must be completely inside the capture')
            arrays = dict(parent_yx_16um=coords[keep],assignment=assignment,local_parent=local,starts_yx=starts)
            lookup = {tuple(point):i for i,point in enumerate(coords)}
            offsets = np.stack(np.meshgrid(np.arange(16),np.arange(16),indexing='ij'),axis=-1).reshape(-1,2)
            slots = np.asarray([[lookup.get(tuple(p),-1) for p in origin//8+offsets] for origin in starts])
            selected = counts[np.maximum(slots.ravel(),0)].multiply((slots.ravel()>=0)[:,None]).tocsr()
            selected.eliminate_zeros()
            arrays.update(counts16_indptr=selected.indptr.astype(np.int64),counts16_indices=selected.indices.astype(np.int32),
                          counts16_data=selected.data.astype(np.int64),observed16_bool=(slots>=0).reshape(-1,16,16))
        else:
            queries = coordinates(path('query_yx_8um'),integer=True)
            starts,query,assignment,weights = prediction_tiles(queries,shape,geometry)
            arrays = dict(query_yx_8um=queries,query_index=query,query_assignment=assignment,query_weight=weights)
        output.mkdir(parents=True)
        for name,value in arrays.items():
            np.save(output/(name+'.npy'),value)
    raw = dict(tissue_image=path('tissue_image') if 'tissue_image' in cfg else None,capture_shape_yx_2um=shape.tolist(),spot_to_image=transform.tolist())
    settings = dict(pad_capture=True,require_full_FOV_HE=False,he_density_cache_mib=512,
                    he_integration_rows=8,uni_batch_size=uni_batch_size,device=str(device))
    source = dict(image_preprocessing=dict(mode='raw'),uni_checkpoint_sha256=metadata['uni_checkpoint_sha256'],
                  uni_checkpoint=path('uni_checkpoint') if 'uni_checkpoint' in cfg else None,uni_config=package/'model/uni_config.json')
    if 'image_cache' in cfg:
        from astra.data.section_inputs import copy_image_cache
        uni_sha = copy_image_cache(path('image_cache'),output,starts,metadata,transform)
    else:
        uni_sha = prepare_images(settings,source,raw,output,starts,timer,config_directory=config_path.parent)
    with timer.phase('input_preparation'):
        extra = {}
        if cfg['task']=='Spot55':
            # Retain measured coarse counts, never target fine-expression arrays.
            observed = output/'observations'
            observed.mkdir()
            np.save(observed/'spot_yx_um.npy',coords)
            np.save(observed/'parent_counts.npy',counts[:,available].toarray())
            np.save(observed/'gene_available.npy',available)
            available,slots,pairs = spot_counts(observed,output,starts)
            valid = np.load(output/'field_valid_bool.npy',mmap_mode='r')
            cells8 = valid.reshape(len(starts),32,4,32,4).all((2,4))
            local = queries[query] - starts[assignment]//4
            if not cells8[assignment,local[:,0],local[:,1]].all():
                raise ValueError('a requested Spot55 output cell lacks complete registered image support')
            extra.update(parent_slots=slots,neighbor_pairs=pairs)
        else:
            valid = np.load(output/'field_valid_bool.npy',mmap_mode='r')
            complete = valid.reshape(len(starts),16,8,16,8).all((2,4))
            known = arrays['observed16_bool']
            if np.any(known & ~complete):
                raise ValueError('an observed HD16 parent lacks complete registered image support')
            extra['input_resolution_um'] = 16
        write(output/'inputs.json',dict(status='complete',role='observed16_input' if cfg['task']=='HD16' else 'observed_spot55_input',
            resource_id=cfg['sample_id'],protocol_id=cfg['protocol_id'],fields=len(starts),
            input_gene_ids=genes,output_gene_ids=read(package/'model/output_gene_ids.json'),gene_available=available.tolist(),
            uni_checkpoint_sha256=uni_sha,contains_fine_expression_arrays=False,preprocessing_mode='raw',
            inference_geometry=geometry,shared_train_inference_cache=True,**extra))
    report = dict(status='complete',sample_id=cfg['sample_id'],task=cfg['task'],fields=len(starts),geometry=geometry,
        device=str(device),uni_batch_size=uni_batch_size,arithmetic_dtype='float32',phase_seconds=timer.seconds,
        elapsed_seconds=time.perf_counter()-started,peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        timing_scope='One fresh coarse packing, registered raw H&E integration and frozen FP32 UNI cache build; reuse once for fit and inference.')
    if torch.device(device).type=='cuda':
        report.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
    write(output/'preparation.json',report)
    return report
