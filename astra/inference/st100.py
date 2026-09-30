"""Frozen ST100 grid9 adapter: one 100um anchor per field, count-space core means."""
from pathlib import Path
import time
import numpy as np
import torch
from astra.data.assay_geometry import st100_geometry, st100_owner, st100_field_indices
from astra.data.image_features import image_features_from_raw
from astra.data.image_readers import registered_transform
from astra.data.section_inputs import copy_image_cache
from astra.inference.section import read, write, Timing
from astra.inference.section_preparation import counts_in_panel, coordinates
from astra.inference.section_prepare_images import prepare_images
from astra.inference.export_diagnostics import spot_grid_weights, count_errors, _merge
from astra.model.model import Direct8Model
from astra.model.fp32 import enable_fp32
from astra.runtime import sha256


GEOMETRY=dict(field_um=256,core_um=160,context_margin_um=48,positions_per_anchor=9,
              offsets_yx_um=[-24,0,24],diameter_um=100,pitch_um=150,grid_um=8,stitch='mean_all_cores')


def prepare(config_path,output,package,*,device='cpu',uni_batch_size=4):
    path,output,package=Path(config_path).resolve(),Path(output),Path(package)
    if output.exists(): raise FileExistsError('choose a new ST100 input directory')
    c=read(path)
    if (c.get('task')!='ST100' or (c.get('diameter_um'),c.get('pitch_um'))!=(100,150)
            or type(c['protocol_id']) is not int or c['protocol_id']!=0):
        raise ValueError('ST100 requires registered 100um/150um geometry and 3prime protocol_id=0')
    genes=read(package/'model/input_gene_ids.json');metadata=read(package/'model/metadata.json')
    for name in ('input_gene_ids.json','output_gene_ids.json','uni_config.json'):
        if sha256(package/'model'/name)!=metadata[Path(name).stem+'_sha256']:
            raise ValueError(f'package identity mismatch: {name}')
    counts,available=counts_in_panel(path.parent/c['counts'],path.parent/c['gene_ids'],genes)
    centers=coordinates(path.parent/c['spot_yx_um'],integer=False)
    if len(centers)!=counts.shape[0]: raise ValueError('ST100 centers and count rows differ')
    g=st100_geometry(centers);shape=np.asarray(c['capture_shape_yx_2um'])
    if (shape.shape!=(2,) or shape.dtype.kind not in 'iu' or np.any(shape<128) or np.any(g['starts']<0)
            or np.any(g['starts']+128>shape)):
        raise ValueError('all nine ST100 fields need full capture/image support; supply a calibrated padded frame')
    transform=registered_transform(c);output.mkdir(parents=True)
    for name,value in [('starts_yx',g['starts']),('field_anchor',g['anchors']),('query_yx_8um',g['query_yx']),
                       ('query_coverage',g['coverage']),('query_parent',g['parent']),('query_spot_area_2um',g['areas']),
                       ('spot_yx_um',centers),('gene_available',available)]: np.save(output/(name+'.npy'),value)
    from scipy import sparse
    sparse.save_npz(output/'counts.npz',counts)
    timer=Timing(device)
    if 'image_cache' in c:
        uni_sha=copy_image_cache(path.parent/c['image_cache'],output,g['starts'],metadata,transform)
    else:
        uni_path=path.parent/c['uni_checkpoint']
        if sha256(uni_path)!=metadata['uni_checkpoint_sha256']: raise ValueError('local UNI identity differs')
        raw=dict(tissue_image=path.parent/c['tissue_image'],capture_shape_yx_2um=shape.tolist(),spot_to_image=transform.tolist())
        source=dict(image_preprocessing=dict(mode='raw'),uni_checkpoint=uni_path,uni_config=package/'model/uni_config.json',
                    uni_checkpoint_sha256=metadata['uni_checkpoint_sha256'])
        uni_sha=prepare_images(dict(device=str(device),uni_batch_size=uni_batch_size,require_full_FOV_HE=True,
                                   he_density_cache_mib=64),source,raw,output,g['starts'],timer,config_directory=path.parent)
    valid=np.load(output/'field_valid_bool.npy',mmap_mode='r')
    if not valid.all(): raise ValueError('ST100 paper adapter requires complete H&E in every FOV')
    if 'tissue_mask_8um' in c:
        mask=np.load(path.parent/c['tissue_mask_8um'],mmap_mode='r',allow_pickle=False)
        if mask.dtype!=np.bool_ or mask.shape!=tuple(shape//4): raise ValueError('ST100 tissue mask must follow capture 8um grid')
        np.save(output/'tissue_mask_8um.npy',mask)
    write(output/'inputs.json',dict(status='complete',role='observed_st100_input',resource_id=c['sample_id'],protocol_id=0,
        fields=len(g['starts']),input_gene_ids=genes,output_gene_ids=genes,gene_available=available.tolist(),
        uni_checkpoint_sha256=uni_sha,preprocessing_mode='raw',contains_fine_expression_arrays=False,
        capture_shape_yx_2um=shape.tolist(),inference_geometry=GEOMETRY))
    report=dict(status='complete',sample_id=c['sample_id'],task='ST100',fields=len(g['starts']),geometry=GEOMETRY,
                raw_UNI_extraction='image_cache' not in c,phase_seconds=timer.seconds)
    write(output/'preparation.json',report);return report


def serialized_diagnostics(output,centers,counts,available):
    values=np.load(output/'prediction.npy',mmap_mode='r',allow_pickle=False)
    queries=np.load(output/'query_yx_8um.npy',allow_pickle=False)
    from astra.inference.benchmark import query_lookup
    errors=[]
    for i,center in enumerate(centers):
        cells,weights=spot_grid_weights(center,diameter_um=100)
        index=query_lookup(queries,cells)
        total=np.einsum('n,ng->g',weights,values[index].astype(float))[available]
        errors.append(count_errors(total,counts[i].toarray()[0,available]))
    return dict(operator='uniform_density_8um_on_center_rasterized_100um_circle',exact_representation=False,
                full_support_spots=len(centers),**_merge(errors))


def fill_gaps(output,mask):
    """Original 4-neighbor/32um tissue-path IDW, separate from direct predictions."""
    from scipy.spatial import cKDTree
    queries=np.load(output/'query_yx_8um.npy');pred=np.load(output/'prediction.npy',mmap_mode='r')
    source=np.zeros(mask.shape,'u1');source[queries[:,0],queries[:,1]]=1
    core=np.flatnonzero(mask[queries[:,0],queries[:,1]])
    candidates=np.argwhere(mask&(source==0))
    kept=[];neighbors_list=[];weights_list=[]
    if len(core):
        tree=cKDTree(queries[core]*8)
        for first in range(0,len(candidates),2048):
            points=candidates[first:first+2048]
            distances,neighbors=tree.query(points*8,k=min(4,len(core)))
            if distances.ndim==1: distances,neighbors=distances[:,None],neighbors[:,None]
            neighbors=core[neighbors];allowed=distances<=32
            for alpha in (.25,.5,.75):
                along=np.rint(points[:,None]*(1-alpha)+queries[neighbors]*alpha).astype(int)
                allowed &= mask[along[...,0],along[...,1]]
            weights=np.where(allowed,1/np.maximum(distances,1)**2,0)
            good=weights.sum(1)>0
            kept.append(points[good]);neighbors_list.append(neighbors[good]);weights_list.append(weights[good]/weights[good].sum(1,keepdims=True))
    q=np.concatenate(kept) if kept else np.empty((0,2),'i8')
    neighbors=np.concatenate(neighbors_list) if neighbors_list else np.empty((0,min(4,len(core))),'i8')
    weights=np.concatenate(weights_list) if weights_list else np.empty_like(neighbors,dtype=float)
    values=np.lib.format.open_memmap(output/'prediction_interpolated.npy',mode='w+',dtype='f4',shape=(len(q),pred.shape[1]))
    for first in range(0,len(q),128):
        values[first:first+128]=np.einsum('nk,nkg->ng',weights[first:first+128],pred[neighbors[first:first+128]],optimize=True)
    values.flush();source[q[:,0],q[:,1]]=2
    np.save(output/'interpolated_yx_8um.npy',q);np.save(output/'interpolation_neighbors.npy',neighbors)
    np.save(output/'interpolation_weights.npy',weights.astype('f4'));np.save(output/'prediction_source.npy',source)
    return dict(interpolated_bins=len(q),neighbors=4,maximum_distance_um=32,source_labels={'0':'unsupported','1':'direct','2':'interpolated'},
                primary_quantitative_support='direct complete-circle bins; exclude interpolated output')


def predict(inputs,output,package,*,device='cpu',batch_size=4):
    from scipy import sparse
    inputs,output,package=Path(inputs),Path(output),Path(package)
    if output.exists(): raise FileExistsError('choose a new ST100 prediction directory')
    if batch_size<1: raise ValueError('batch_size must be positive')
    begun=time.perf_counter();info=read(inputs/'inputs.json');metadata=read(package/'model/metadata.json')
    genes=read(package/'model/input_gene_ids.json')
    if (info.get('role')!='observed_st100_input' or info.get('inference_geometry')!=GEOMETRY
            or info.get('protocol_id')!=0 or info.get('preprocessing_mode')!='raw'
            or info.get('contains_fine_expression_arrays') is not False
            or info.get('input_gene_ids')!=genes or info.get('output_gene_ids')!=genes
            or info.get('uni_checkpoint_sha256')!=metadata['uni_checkpoint_sha256']):
        raise ValueError('ST100 cache identity/geometry differs from the paper adapter')
    for name in ('checkpoint.pt','config.json','input_gene_ids.json','output_gene_ids.json'):
        if sha256(package/'model'/name)!=metadata[Path(name).stem+'_sha256']: raise ValueError(f'package identity mismatch: {name}')
    centers=np.load(inputs/'spot_yx_um.npy');g=st100_geometry(centers)
    for name,key in [('starts_yx','starts'),('field_anchor','anchors'),('query_yx_8um','query_yx'),('query_coverage','coverage')]:
        if not np.array_equal(np.load(inputs/(name+'.npy')),g[key]): raise ValueError(f'ST100 cache geometry differs: {name}')
    available=np.load(inputs/'gene_available.npy',allow_pickle=False);counts=sparse.load_npz(inputs/'counts.npz').tocsr()
    if (available.shape!=(len(genes),) or available.dtype!=np.bool_ or not available.any()
            or counts.shape!=(len(centers),len(genes)) or np.any(counts.data<0) or counts.dtype.kind not in 'iu'):
        raise ValueError('invalid ST100 count/availability panel')
    arrays={name:np.load(inputs/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            for name in ('density_float32','area_uint16','field_valid_bool','features_float32','pathology_valid_bool')}
    from astra.data.section_inputs import IMAGE_ARRAYS
    for name,(shape,dtype) in IMAGE_ARRAYS.items():
        if arrays[name].shape!=(len(g['starts']),*shape) or arrays[name].dtype!=dtype: raise ValueError(f'invalid ST100 image array: {name}')
    model=Direct8Model(**read(package/'model/config.json')['kwargs']).to(device).eval()
    model.load_state_dict(torch.load(package/'model/checkpoint.pt',map_location='cpu',weights_only=True),strict=True)
    model=enable_fp32(model);output.mkdir(parents=True)
    result_grid=np.lib.format.open_memmap(output/'prediction.npy',mode='w+',dtype='f4',shape=(len(g['query_yx']),len(genes)))
    result_grid[:]=0;seen=np.zeros(len(g['query_yx']),'i4');error=0.
    with torch.no_grad():
        for first in range(0,len(g['starts']),batch_size):
            stop=min(first+batch_size,len(g['starts']));starts=g['starts'][first:stop];anchors=g['anchors'][first:stop];n=len(starts)
            def tensor(name): return torch.from_numpy(np.array(arrays[name][first:stop],copy=True)).to(device)
            density,area,valid=tensor('density_float32'),tensor('area_uint16'),tensor('field_valid_bool')
            if not valid.all() or not torch.isfinite(density).all() or torch.any(density<0) or torch.any(area!=65535):
                raise ValueError('ST100 needs complete finite raw image support')
            owner=torch.from_numpy(np.stack([st100_owner(start,centers[a]) for start,a in zip(starts,anchors)])).to(device).long()
            kwargs=dict(parent_counts=torch.as_tensor(counts[anchors].toarray()[:,None],dtype=torch.float32,device=device),
                owner_map=owner,parent_valid=torch.ones((n,1),dtype=torch.bool,device=device),
                image_features_2um=image_features_from_raw(density,area,valid).permute(0,3,1,2).contiguous(),field_valid=valid,
                gene_available=torch.as_tensor(available,device=device)[None].expand(n,-1),protocol_id=torch.zeros(n,dtype=torch.long,device=device),
                pathology_features=tensor('features_float32'),pathology_valid=tensor('pathology_valid_bool'))
            result=model(**kwargs)
            error=max(error,float((result['parent_conservation_residual'].abs()/(1+kwargs['parent_counts'].abs())).max()))
            if error>5e-6: raise ValueError('ST100 individual-FOV conservation exceeds FP32 tolerance')
            cores=result['pred_count_8um'][:,6:26,6:26].cpu().numpy().reshape(n,400,len(genes))
            if not np.isfinite(cores).all() or np.any(cores<0): raise ValueError('invalid ST100 core prediction')
            for start,core in zip(starts,cores):
                idx=st100_field_indices(g,start);result_grid[idx]+=core;seen[idx]+=1
    if not np.array_equal(seen,g['coverage']): raise ValueError('incomplete ST100 core contribution coverage')
    for first in range(0,len(result_grid),128): result_grid[first:first+128]/=seen[first:first+128,None]
    result_grid.flush()
    for name,value in [('query_yx_8um',g['query_yx']),('gene_available',available),('query_parent',g['parent']),
                       ('query_spot_area_2um',g['areas']),('query_coverage',seen),('xy_um',g['query_yx'][:,::-1]*8+4)]: np.save(output/(name+'.npy'),value)
    write(output/'gene_ids.json',genes)
    diagnostics=serialized_diagnostics(output,centers,counts,available)
    gap_report=None
    if (inputs/'tissue_mask_8um.npy').exists(): gap_report=fill_gaps(output,np.load(inputs/'tissue_mask_8um.npy'))
    report=dict(status='complete',sample_id=info['resource_id'],task='ST100',geometry=GEOMETRY,
        arithmetic_dtype='float32',preprocessing_mode='raw',checkpoint_sha256=metadata['checkpoint_sha256'],fine_tuning=None,
        fields=len(g['starts']),output_shape=list(result_grid.shape),parent_conservation_max_scaled=error,
        parent_conservation_tolerance=5e-6,
        parent_conservation_scope='single_FOV_anchor',serialized_export=diagnostics,gap_fill=gap_report,
        elapsed_seconds=time.perf_counter()-begun,query_fine_labels_used_for_fitting=False)
    write(output/'report.json',report);return report
