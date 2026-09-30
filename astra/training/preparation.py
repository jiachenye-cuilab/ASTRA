"""Portable preparation for the frozen twelve-section ASTRA training recipe."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from astra.training.cache import authorized_ids, load_panels, validate_resources
from astra.training.utils import ROOT, artifact_path, read_json


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f'existing configuration or metadata differs: {path}')
        return
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def save_array(path, value):
    if path.exists():
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), value)
    else:
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)


def smoke_config(config):
    result = deepcopy(config)
    result.update(engineering_smoke_dataset=True, output_root='outputs/training-smoke')
    result['training'].update(formal_training=False, fields_per_section=3,
        fields_per_epoch=3*len(authorized_ids(config,'train')), section_field_overrides={}, cpu_threads=2,
        validation_fields_per_section=4)
    result['cache'].update(train_root='outputs/training-smoke-cache/train',
        validation_root='outputs/training-smoke-cache/validation',
        uni_root='outputs/training-smoke-cache/uni', uni_summary='outputs/training-smoke-cache/uni/summary.json',
        validation_layout_root='outputs/training-smoke-cache/layouts',
        spatial_split_artifact='training/smoke/spatial_split.json')
    return result


def prepare_geometry(config):
    """Restore exact FOV ordering; the bounded smoke uses actual selected rows."""
    smoke = config.get('engineering_smoke_dataset', False)
    spatial = deepcopy(read_json(ROOT / 'training/spatial_split.json'))
    for rid in authorized_ids(config, 'train'):
        with np.load(ROOT / f'training/geometry/{rid}.npz', allow_pickle=False) as f:
            arrays = {key: f[key] for key in f.files}
        spec = spatial['sections'][rid]
        rows = np.arange(len(arrays['starts_yx']), dtype=np.int64)
        if smoke:
            rows = np.asarray(sorted(set(arrays['training_indices'][:3].tolist() +
                                        [x['field_index'] for x in spec['monitor_items']])), dtype=np.int64)
        old_slots = arrays['train_block_slot'][rows]
        block_rows, slots = np.unique(old_slots, return_inverse=True)
        blocks = arrays['block_starts_yx'][block_rows]
        starts = arrays['starts_yx'][rows]
        if not smoke:
            # Retain all blocks and their source IDs for the formal cache.
            block_rows = np.arange(len(arrays['block_starts_yx']))
            blocks, slots = arrays['block_starts_yx'], old_slots
        native = artifact_path(config['cache']['train_root']) / rid
        native.mkdir(parents=True, exist_ok=True)
        for name, value in dict(block_starts_yx=blocks, starts_yx=starts,
            train_block_slot=slots.astype(np.uint16), train_offset_yx=arrays['train_offset_yx'][rows],
            source_indices=rows, source_block_indices=block_rows).items():
            save_array(native / f'{name}.npy', value)
        if smoke:
            inverse = {int(original): index for index, original in enumerate(rows)}
            selected = np.asarray([i for i, original in enumerate(rows)
                                   if original in set(arrays['training_indices'][:3])], dtype=np.int64)
            spec.update(source_fields=len(rows), retained_fields=len(selected),
                        excluded_fields=len(rows) - len(selected),
                        training_indices=f'training/smoke/{rid}_training_indices.npy')
            for item in spec['monitor_items']:
                item['field_index'] = inverse[item['field_index']]
            index_path = artifact_path(spec['training_indices'])
            index_path.parent.mkdir(parents=True, exist_ok=True)
            save_array(index_path, selected)
    if smoke:
        write_new(artifact_path(config['cache']['spatial_split_artifact']), spatial)
    return spatial


def source(config, rid):
    import h5py
    from astra.data.image_readers import open_rgb_image
    registry = validate_resources(config)
    if rid not in authorized_ids(config, 'train'):
        raise PermissionError('native preparation is restricted to the training cohort')
    record = registry['datasets'][rid]
    feature = ROOT / record['feature_slice']
    image = ROOT / record['tissue_image']
    with h5py.File(feature, 'r') as handle:
        meta = json.loads(handle.attrs['metadata_json'])
    if float(meta['spot_pitch']) != 2:
        raise ValueError('feature_slice must use the original 2um grid')
    transform = np.asarray(meta['transform_matrices']['spot_colrow_to_microscope_colrow'])
    return feature, image, meta, transform, open_rgb_image


def prepare_native(config, rid):
    from astra.training.native_counts import stage_counts
    from astra.data.native_image import aggregate_density_region, full_valid_cells, quantize_valid_fraction
    root = artifact_path(config['cache']['train_root']) / rid
    panel = load_panels(config)
    if (root / 'section.json').exists():
        record = read_json(root / 'section.json')
        if record['status'] != 'complete' or record['input_gene_ids'] != list(panel.input_gene_ids):
            raise ValueError('existing native cache differs')
        return
    files = ('local_linear.npy', 'gene_index.npy', 'count_uint16.npy', 'block_indptr.npy',
             'density_float32.npy', 'valid_area_uint16.npy', 'full_valid_bool.npy')
    if any((root / name).exists() for name in files):
        raise FileExistsError(f'partial native cache must be recovered or removed explicitly: {root}')
    feature, image, meta, transform, reader_class = source(config, rid)
    blocks = np.load(root / 'block_starts_yx.npy', allow_pickle=False)
    starts = np.load(root / 'starts_yx.npy', allow_pickle=False)
    protocol = 0 if rid in config['three_prime']['auxiliary_train'] else 1
    nnz, available = stage_counts(root, dict(feature_slice=feature, protocol='3prime' if protocol == 0 else 'WT'),
                                  panel.input_gene_ids, blocks, (meta['nrows'], meta['ncols']))
    density = np.lib.format.open_memmap(root / 'density_float32.npy', mode='w+', dtype=np.float32,
                                       shape=(len(blocks), 256, 256, 3))
    area = np.lib.format.open_memmap(root / 'valid_area_uint16.npy', mode='w+', dtype=np.uint16,
                                    shape=(len(blocks), 256, 256))
    valid = np.lib.format.open_memmap(root / 'full_valid_bool.npy', mode='w+', dtype=bool,
                                     shape=(len(blocks), 256, 256))
    with reader_class(image) as reader:
        for index, (row, col) in enumerate(blocks):
            bounds = dict(row_start=int(row), row_stop=int(row) + 256,
                          column_start=int(col), column_stop=int(col) + 256)
            values, coverage = aggregate_density_region(reader, transform, **bounds, samples_per_cell_axis=8)
            density[index] = values
            area[index] = quantize_valid_fraction(coverage)
            valid[index] = full_valid_cells(transform, **bounds, image_width=reader.width, image_height=reader.height)
    density.flush(); area.flush(); valid.flush()
    write_new(root / 'section.json', dict(status='complete', resource_id=rid, section=rid, role='train',
        protocol_id=protocol, fields=len(starts), blocks=len(blocks), block_cells_2um=256, patch_cells_2um=128,
        input_gene_ids=list(panel.input_gene_ids), output_gene_ids=list(panel.output_gene_ids),
        gene_available=available, sparse_nonzero_records=int(nnz), image_quadrature_samples_per_2um_axis=8,
        engineering_smoke_dataset=config.get('engineering_smoke_dataset', False)))


def prepare_uni(config, rid, encoder, checkpoint_sha, device, batch_size):
    from astra.data.native_image import sample_registered_rgb
    native = artifact_path(config['cache']['train_root']) / rid
    starts = np.load(native / 'starts_yx.npy', allow_pickle=False)
    root = artifact_path(config['cache']['uni_root']) / rid
    if (root / 'section.json').exists():
        record = read_json(root / 'section.json')
        if record['status'] != 'complete' or record['uni']['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('existing UNI cache differs')
        np.testing.assert_array_equal(np.load(root / 'starts_yx.npy', allow_pickle=False), starts)
        return
    root.mkdir(parents=True, exist_ok=False)
    _, image, _, transform, reader_class = source(config, rid)
    features = np.lib.format.open_memmap(root / 'features_float32.npy', mode='w+', dtype=np.float32,
                                        shape=(len(starts), 1024, 14, 14))
    valid = np.lib.format.open_memmap(root / 'valid_bool.npy', mode='w+', dtype=bool,
                                     shape=(len(starts), 14, 14))
    with reader_class(image) as reader, torch.no_grad():
        for begin in range(0, len(starts), batch_size):
            rows = [sample_registered_rgb(reader, transform, start) for start in starts[begin:begin + batch_size]]
            rgb = np.stack([x[0] for x in rows]).transpose(0, 3, 1, 2)
            support = torch.from_numpy(np.stack([x[1] for x in rows]))[:, None].float().to(device)
            mask = torch.nn.functional.avg_pool2d(support, 16, 16)[:, 0] == 1
            encoded = encoder(torch.from_numpy(rgb).to(device), extent_um=256.)
            if encoded.dtype != torch.float32 or not bool(mask.flatten(1).any(1).all()):
                raise ValueError('UNI precision or image validity differs')
            features[begin:begin + len(rows)] = encoded.cpu().numpy()
            valid[begin:begin + len(rows)] = mask.cpu().numpy()
    features.flush(); valid.flush()
    save_array(root / 'starts_yx.npy', starts)
    write_new(root / 'section.json', dict(status='complete', resource_id=rid, role='train',
        fields=len(starts), uni=dict(checkpoint_sha256=checkpoint_sha)))


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument('--workspace', type=Path, help='user task created by configure-training')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--stage', choices=('check', 'geometry', 'native', 'uni', 'all'), default='check')
    parser.add_argument('--section', help='one training resource ID; default: all twelve sections')
    parser.add_argument('--smoke', action='store_true', help='create a bounded, formal-training-disabled data task')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args(argv)
    if args.workspace:
        from astra.training.utils import use_workspace
        use_workspace(args.workspace)
    args.config = args.config or ROOT / 'training/config.json'
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    config = read_json(args.config)
    if args.smoke:
        config = smoke_config(config)
        write_new(ROOT / 'training/smoke/config.json', config)
    registry = validate_resources(config)
    load_panels(config)
    ids = authorized_ids(config, 'train')
    selected = [args.section] if args.section else ids
    if not set(selected) <= set(ids):
        parser.error('--section must name a training resource; test data cannot be prepared here')
    if args.stage == 'check':
        missing = [registry['datasets'][rid][key] for rid in selected for key in ('feature_slice', 'tissue_image')
                   if not (ROOT / registry['datasets'][rid][key]).is_file()]
        uni = registry['model_weights'][config['cache']['uni_resource_id']]
        if not (ROOT / uni['weight_file']).is_file():
            missing.append(uni['weight_file'])
        print(json.dumps(dict(status='missing_inputs' if missing else 'inputs_present', missing=missing,
            training_started=False, test_expression_read=False), indent=2))
        raise SystemExit(1 if missing else 0)
    prepare_geometry(config)
    torch.set_num_threads(2)
    if args.stage in ('native', 'all'):
        for rid in selected:
            prepare_native(config, rid)
            print(json.dumps(dict(stage='native', section=rid, status='complete')), flush=True)
    if args.stage in ('uni', 'all'):
        from astra.model.pathology import FrozenUNI
        uni = registry['model_weights'][config['cache']['uni_resource_id']]
        path = ROOT / uni['weight_file']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        expected = read_json(ROOT / 'model/metadata.json')['uni_checkpoint_sha256']
        if digest != expected or digest != uni['sha256']:
            raise ValueError('UNI checkpoint differs from the published training identity')
        device = torch.device(args.device)
        if device.type == 'cuda':
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError('expose exactly one authorized GPU with CUDA_VISIBLE_DEVICES')
            torch.cuda.set_per_process_memory_fraction(config['training']['cuda_memory_fraction'], device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        encoder = FrozenUNI(path, config_path=ROOT / uni['config_file']).to(device).eval()
        for rid in selected:
            prepare_uni(config, rid, encoder, digest, device, args.batch_size)
            print(json.dumps(dict(stage='uni', section=rid, status='complete')), flush=True)
        complete = all((artifact_path(config['cache']['uni_root']) / rid / 'section.json').is_file() for rid in ids)
        summary = artifact_path(config['cache']['uni_summary'])
        summary.write_text(json.dumps(dict(status='complete' if complete else 'partial',
            dtype='float32', checkpoint_sha256=digest), indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
