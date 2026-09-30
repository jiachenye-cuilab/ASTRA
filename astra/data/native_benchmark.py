"""Native 2um 10x feature-slice adapter; coarse inputs and evaluation truth stay separate."""
from pathlib import Path
import json
import h5py
import numpy as np
from scipy import sparse
from astra.training.native_counts import feature_table
from astra.data.assay_geometry import circle_bins, lattice, lattice_origin, interior_queries
from astra.inference.section import read, write
from astra.inference.section_preparation import coordinates
from astra.data.image_readers import registered_transform


def feature_metadata(path):
    with h5py.File(path, 'r') as f:
        m = json.loads(f.attrs['metadata_json'])
    if float(m['spot_pitch']) != 2:
        raise ValueError('native feature_slice must have 2um pitch; no resampling')
    shape = np.asarray([m['nrows'], m['ncols']])
    if shape.dtype.kind not in 'iu' or np.any(shape < 128):
        raise ValueError('capture must contain a 256um FOV')
    return m, shape.astype('i8')


def tissue_parents(path, shape):
    with h5py.File(path, 'r') as f:
        group = f['masks/square_016um']
        keep = np.asarray(group['data'][:], bool)
        yx = np.column_stack((group['row'][:], group['col'][:]))[keep].astype('i8')
    yx = yx[np.lexsort((yx[:,1], yx[:,0]))]
    if (not len(yx) or len(np.unique(yx, axis=0)) != len(yx) or np.any(yx < 0)
            or np.any((yx+1)*8 > shape)):
        raise ValueError('tissue mask must contain unique complete 16um parents')
    return yx


def native_values(handle, source, shape):
    key = f'feature_slices/{source}'
    if source < 0 or key not in handle:
        return np.empty((0,2), 'i8'), np.empty(0, 'i8')
    group = handle[key]
    yx = np.column_stack((group['row'][:], group['col'][:]))
    values = np.asarray(group['data'][:])
    if (yx.dtype.kind not in 'iu' or values.shape != (len(yx),) or values.dtype.kind not in 'iu' or np.any(values < 0)
            or np.any(values >= np.iinfo('i8').max) or np.any(yx < 0) or np.any(yx >= shape)):
        raise ValueError('native coordinates/counts must be valid raw integer UMIs')
    return yx.astype('i8'), values.astype('i8')


def aggregate_queries(path, sources, shape, queries, destination):
    """Read one gene at a time, sum the sixteen native cells in each 8um bin."""
    width = (int(shape[1])+3)//4
    keys = queries[:,0]*width+queries[:,1]
    order = np.argsort(keys); ordered = keys[order]
    with h5py.File(path, 'r') as handle:
        for g, source in enumerate(sources):
            yx, values = native_values(handle, source, shape)
            native_keys = (yx[:,0]//4)*width+yx[:,1]//4
            slots = np.searchsorted(ordered, native_keys)
            keep = slots < len(ordered)
            keep[keep] &= ordered[slots[keep]] == native_keys[keep]
            column = np.zeros(len(queries), 'i8')
            np.add.at(column, order[slots[keep]], values[keep])
            destination[:,g] = column
    destination.flush()


def prepare_benchmark(config_path, output, package):
    path, output, package = Path(config_path).resolve(), Path(output), Path(package)
    if output.exists():
        raise FileExistsError('choose a new benchmark preparation directory')
    c = read(path)
    if (c['task'] not in ('HD16', 'Spot55') or type(c['protocol_id']) is not int or c['protocol_id'] not in (0,1)
            or not isinstance(c['sample_id'],str) or not c['sample_id'].strip()):
        raise ValueError('benchmark supports HD16 or pseudo-Visium Spot55 and explicit assay identity')
    native = (path.parent/c['feature_slice']).resolve()
    m, shape = feature_metadata(native)
    genes = read(package/'model/input_gene_ids.json')
    table = feature_table(native, 'WT' if c['protocol_id'] == 1 else '3prime')
    available = np.array([g in table for g in genes])
    if not available.any():
        raise ValueError('native assay has no measured ASTRA genes')
    sources = [table[g][0] if g in table else -1 for g in genes]
    parents = tissue_parents(native, shape)
    transform = registered_transform(dict(spot_to_image=m['transform_matrices']['spot_colrow_to_microscope_colrow']))
    # Geometry-only image filtering matches the native benchmark; it reads no expression.
    from astra.data.image_readers import open_rgb_image
    with open_rgb_image(path.parent/c['tissue_image']) as image:
        image_size = np.array([image.width,image.height])
    corners = parents[:,None,::-1]*8+np.array([[-.5,-.5],[7.5,-.5],[-.5,7.5],[7.5,7.5]])
    pixels = np.einsum('ij,nkj->nki',transform,np.concatenate((corners,np.ones((*corners.shape[:2],1))),axis=-1))
    pixels = pixels[...,:2]/pixels[...,2:]
    parents = parents[((pixels>=-.5)&(pixels<image_size-.5)).all((1,2))]
    if not len(parents): raise ValueError('no tissue parents with complete registered-image support')
    if c['task'] == 'Spot55':
        origin = lattice_origin(c['sample_id'], c.get('lattice_seed', 20260907))
        centers, array = lattice(shape, origin)
        tissue_keys = set(map(tuple, parents))
        keep = np.array([tuple(v) in tissue_keys for v in (centers//16).astype(int)])
        centers = centers[keep]
        # Keep only complete 56um registered image squares, matching the published adapter.
        corners = centers[:,None,::-1]+np.array([[-28,-28],[-28,28],[28,-28],[28,28]])
        grid = corners/2-.5
        pixels = np.einsum('ij,nkj->nki', transform, np.concatenate((grid,np.ones((*grid.shape[:2],1))),axis=-1))
        pixels = pixels[...,:2]/pixels[...,2:]
        centers = centers[((pixels >= -.5) & (pixels < image_size-.5)).all((1,2))]
        flat, owner = circle_bins(centers, shape)
        default_queries, _ = interior_queries(centers)
        observed_coords = centers
        observed = np.zeros((len(centers),len(genes)), 'i8')
        with h5py.File(native, 'r') as f:
            for g, source in enumerate(sources):
                yx, values = native_values(f, source, shape)
                keys = yx[:,0]*shape[1]+yx[:,1]; slots = np.searchsorted(flat,keys)
                keep = slots < len(flat); keep[keep] &= flat[slots[keep]] == keys[keep]
                np.add.at(observed[:,g], owner[slots[keep]], values[keep])
    else:
        origin = None; observed_coords = parents
        default_queries = (parents[:,None]*2+np.array([(0,0),(0,1),(1,0),(1,1)])).reshape(-1,2)
        # Coarse aggregation only. No 8um labels are passed to ASTRA.
        parent_width = (int(shape[1])+7)//8
        keys = parents[:,0]*parent_width+parents[:,1]
        observed = np.zeros((len(parents),len(genes)), 'i8')
        with h5py.File(native, 'r') as f:
            for g, source in enumerate(sources):
                yx, values = native_values(f,source,shape)
                slots = np.searchsorted(keys,(yx[:,0]//8)*parent_width+yx[:,1]//8)
                keep = slots < len(keys)
                keep[keep] &= keys[slots[keep]] == ((yx[:,0]//8)*parent_width+yx[:,1]//8)[keep]
                np.add.at(observed[:,g],slots[keep],values[keep])
    queries = (coordinates(path.parent/c['evaluation_query_yx_8um'], integer=True)
               if 'evaluation_query_yx_8um' in c else default_queries)
    if not len(queries) or np.any((queries+1)*4 > shape):
        raise ValueError('evaluation queries must be nonempty complete native 8um bins')
    if c['task']=='HD16':
        from astra.inference.benchmark import query_lookup
        query_lookup(default_queries,queries)
    output.mkdir(parents=True)
    obs, ref = output/'observations', output/'reference'
    obs.mkdir(); ref.mkdir()
    sparse.save_npz(obs/'counts.npz', sparse.csr_matrix(observed[:,available]))
    write(obs/'gene_ids.json', np.asarray(genes)[available].tolist())
    coord_name = 'parent_yx_16um' if c['task']=='HD16' else 'spot_yx_um'
    np.save(obs/(coord_name+'.npy'),observed_coords)
    if c['task']=='Spot55':
        np.save(obs/'query_yx_8um.npy',queries)
    section = dict(task=c['task'],sample_id=c['sample_id'],protocol_id=c['protocol_id'],
        capture_shape_yx_2um=shape.tolist(),spot_to_image=transform.tolist(),counts='counts.npz',
        gene_ids='gene_ids.json',**{coord_name:coord_name+'.npy'},core_um=160,
        stride_um=160 if c['task']=='HD16' else 144)
    import os
    for key in ('tissue_image','uni_checkpoint','image_cache'):
        if key in c: section[key]=os.path.relpath((path.parent/c[key]).resolve(),obs.resolve())
    if c['task']=='Spot55': section['query_yx_8um']='query_yx_8um.npy'
    write(obs/'section.json',section)
    np.save(ref/'query_yx_8um.npy',queries); np.save(ref/'gene_available.npy',available)
    write(ref/'gene_ids.json',genes)
    target=np.lib.format.open_memmap(ref/'counts_8um.npy',mode='w+',dtype='i8',shape=(len(queries),len(genes)))
    aggregate_queries(native,sources,shape,queries,target)
    write(ref/'reference.json',dict(role='evaluation_only_measured_8um',sample_id=c['sample_id'],
        task=c['task'],counts_unit='raw_UMI_counts',queries=len(queries),genes=len(genes),
        available_genes=int(available.sum()),query_support=('explicit' if 'evaluation_query_yx_8um' in c else
            'complete_native_tissue_image_valid_16um_children' if c['task']=='HD16' else 'complete_8um_squares_inside_55um_circles'),
        lattice_origin_yx_um=origin,reference_used_for_model_fitting=False))
    return dict(status='complete',sample_id=c['sample_id'],task=c['task'],
        section_config=str(obs/'section.json'),evaluation_reference=str(ref),query_count=len(queries),
        reference_used_for_model_fitting=False)
