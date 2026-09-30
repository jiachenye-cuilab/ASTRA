"""Portable Space Ranger count/position conversion and registered image cache reuse."""
import csv
from pathlib import Path
import os
import shutil
import h5py
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from astra.data.image_readers import registered_transform
from astra.inference.section import read, write


IMAGE_ARRAYS = {'density_float32': ((128,128,3),np.float32), 'area_uint16': ((128,128),np.uint16),
                'field_valid_bool': ((128,128),np.bool_), 'features_float32': ((1024,14,14),np.float32),
                'pathology_valid_bool': ((14,14),np.bool_)}


def copy_image_cache(cache, output, starts, metadata, transform):
    """Only reuse explicitly registered raw features for these exact field identities."""
    cache=Path(cache);info=read(cache/'images.json')
    if (info.get('preprocessing_mode')!='raw' or info.get('uni_checkpoint_sha256')!=metadata['uni_checkpoint_sha256']
            or info.get('density_method')!='native_RGB_OD_HE_bilinear_midpoint_8x8'
            or not np.array_equal(info.get('spot_to_image'),transform)
            or not np.array_equal(np.load(cache/'starts_yx.npy',allow_pickle=False),starts)):
        raise ValueError('image cache must match raw preprocessing, UNI identity, registration and exact FOV order')
    for name,(shape,dtype) in IMAGE_ARRAYS.items():
        values=np.load(cache/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        if values.shape!=(len(starts),*shape) or values.dtype!=dtype:
            raise ValueError(f'image cache shape/dtype differs: {name}')
        for first in range(0,len(starts),4):
            if values.dtype.kind=='f' and (not np.isfinite(values[first:first+4]).all() or np.any(values[first:first+4]<0) and name=='density_float32'):
                raise ValueError(f'invalid image cache values: {name}')
        shutil.copyfile(cache/(name+'.npy'),Path(output)/(name+'.npy'))
    write(Path(output)/'images.json',info)
    return metadata['uni_checkpoint_sha256']


def prepare_visium(config_path, output):
    """Align filtered 10x H5 barcodes to CSV positions; no guessed image scale."""
    path,output=Path(config_path).resolve(),Path(output)
    if output.exists(): raise FileExistsError('choose a new Visium conversion directory')
    c=read(path); transform=registered_transform(c)
    if ((c.get('diameter_um'),c.get('pitch_um'))!=(55,100) or type(c['protocol_id']) is not int or c['protocol_id'] not in (0,1)):
        raise ValueError('Space Ranger Visium adapter requires explicit 55um diameter and 100um pitch')
    def decode(v): return [x.decode() if isinstance(x,bytes) else str(x) for x in v]
    with h5py.File(path.parent/c['matrix_h5'],'r') as f:
        m=f['matrix'];shape=tuple(m['shape'][:])
        counts=sparse.csc_matrix((m['data'][:],m['indices'][:],m['indptr'][:]),shape=shape).T.tocsr()
        barcodes=decode(m['barcodes'][:]);genes=decode(m['features/id'][:])
        if 'feature_type' in m['features'] and set(decode(m['features/feature_type'][:]))!={'Gene Expression'}:
            raise ValueError('convert Gene Expression features only')
    if (counts.shape!=(len(barcodes),len(genes)) or counts.dtype.kind not in 'iu' or np.any(counts.data<0)
            or len(set(barcodes))!=len(barcodes) or len(set(genes))!=len(genes)):
        raise ValueError('10x matrix needs aligned unique barcodes/genes and raw nonnegative integer counts')
    with (path.parent/c['positions_csv']).open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    positions={r['barcode']:r for r in rows}
    if len(positions)!=len(rows) or not set(barcodes)<=set(positions):
        raise ValueError('positions must uniquely cover all matrix barcodes')
    selected=[];centers=[]
    for i,b in enumerate(barcodes):
        r=positions[b]
        if r['in_tissue'] not in ('0','1') or r.get('qc_pass','1') not in ('0','1'):
            raise ValueError('in_tissue/qc_pass must be 0 or 1')
        if r['in_tissue']=='0' or r.get('qc_pass','1')=='0': continue
        pixels=np.array([float(r['pxl_col_in_fullres']),float(r['pxl_row_in_fullres']),1.])
        grid=np.linalg.solve(transform,pixels);grid=grid[:2]/grid[2]
        centers.append((grid[::-1]*2+1).tolist());selected.append(i)
    if not selected: raise ValueError('no retained tissue/QC spots')
    centers=np.array(centers,float);shape=np.asarray(c['capture_shape_yx_2um'])
    if not np.isfinite(centers).all() or len(np.unique(centers,axis=0))!=len(centers):
        raise ValueError('registered spot centers must be finite and unique')
    if shape.shape!=(2,) or shape.dtype.kind not in 'iu' or np.any(shape<128):
        raise ValueError('explicit capture shape on the registered 2um physical grid is required')
    if np.any(centers<27.5) or np.any(centers+27.5>shape*2):
        raise ValueError('retained observations require complete 55um circles inside capture')
    if 'query_yx_8um' in c:
        from astra.inference.section_preparation import coordinates
        queries=coordinates(path.parent/c['query_yx_8um'],integer=True)
        support='user supplied fixed capture/tissue/support mask'
    else:
        lo=np.maximum(0,np.floor((centers.min(0)-100)/8).astype(int))
        hi=np.minimum(shape//4,np.ceil((centers.max(0)+100)/8).astype(int))
        if np.prod(hi-lo,dtype='i8')>64*2**20: raise ValueError('partition sections with >64M candidate bins')
        queries=np.indices(hi-lo).reshape(2,-1).T+lo
        distance,_=cKDTree(centers).query(queries*8+4)
        queries=queries[distance<=100]
        if 'tissue_mask_2um' in c:
            mask=np.load(path.parent/c['tissue_mask_2um'],mmap_mode='r',allow_pickle=False)
            if mask.shape!=tuple(shape) or mask.dtype!=np.bool_: raise ValueError('tissue mask shape/dtype differs')
            keep=[]
            for y,x in queries: keep.append(mask[y*4:y*4+4,x*4:x*4+4].sum()>=8)
            queries=queries[np.asarray(keep,bool)]
        support='<=100um from retained QC centers; optional >=half tissue subcells; not exact paper support unless supplied'
    if not len(queries) or np.any((queries+1)*4>shape): raise ValueError('empty/out-of-capture query support')
    output.mkdir(parents=True)
    sparse.save_npz(output/'counts.npz',counts[selected])
    write(output/'gene_ids.json',genes);write(output/'barcodes.json',[barcodes[i] for i in selected])
    np.save(output/'spot_yx_um.npy',centers);np.save(output/'query_yx_8um.npy',queries)
    section=dict(task='Spot55',sample_id=c['sample_id'],protocol_id=c['protocol_id'],
                 counts='counts.npz',gene_ids='gene_ids.json',spot_yx_um='spot_yx_um.npy',query_yx_8um='query_yx_8um.npy',
                 capture_shape_yx_2um=shape.tolist(),spot_to_image=transform.tolist(),core_um=160,stride_um=144)
    for key in ('tissue_image','uni_checkpoint','image_cache'):
        if key in c: section[key]=os.path.relpath((path.parent/c[key]).resolve(),output.resolve())
    write(output/'section.json',section)
    report=dict(status='complete',sample_id=c['sample_id'],spots=len(selected),queries=len(queries),
                geometry='field256/core160/stride144',query_support=support,
                registration_inferred=False,counts_normalized=False)
    write(output/'conversion.json',report);return report
