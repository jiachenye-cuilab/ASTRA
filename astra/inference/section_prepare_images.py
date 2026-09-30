from contextlib import nullcontext
import numpy as np
import torch
from torch.nn import functional as F
from astra.data.native_image import SourceDensityCache, full_valid_cells, quantize_valid_fraction, sample_registered_rgb
from astra.model.pathology import FrozenUNI
from astra.inference.section_image import aggregate_density_region

def write(path, data):
    import json
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')

def image_source(raw):
    from astra.data.image_readers import open_rgb_image, registered_transform
    return (dict(nrows=raw['capture_shape_yx_2um'][0], ncols=raw['capture_shape_yx_2um'][1]),
            registered_transform(raw), open_rgb_image)

@torch.no_grad()
def prepare_images(cfg, source, raw, destination, starts, timing, *, config_directory):
    """Registered HE/Reinhard and raw FP32 UNI, rebuilt for either observation task."""
    n = len(starts)
    with timing.phase('HE_preparation'):
        metadata, transform, reader_class = image_source(raw)
        padded = cfg.get('pad_capture', False)
        shape = np.asarray([metadata['nrows'], metadata['ncols']])
        statistics_starts = starts[((starts >= 0) & (starts + 128 <= shape)).all(1)] if padded else starts
        settings = source['image_preprocessing']
        if settings['mode'] != 'raw':
            raise ValueError('ASTRA requires raw H&E')
        normalize = None
        arrays = {name: np.lib.format.open_memmap(destination / (name + '.npy'), mode='w+', dtype=dtype,
                    shape=(n, *shape)) for name, shape, dtype in (
            ('density_float32', (128, 128, 3), np.float32), ('area_uint16', (128, 128), np.uint16),
            ('field_valid_bool', (128, 128), bool), ('rgb_float32', (3, 224, 224), np.float32),
            ('rgb_valid_bool', (224, 224), bool))}
        cache_bytes = int(cfg.get('he_density_cache_mib', 512) * 2**20)
        integration_rows = cfg.get('he_integration_rows', 16)
        if cache_bytes < 0:
            raise ValueError('he_density_cache_mib must be nonnegative; zero disables block reuse')
        with reader_class(raw['tissue_image']) as reader, (
                SourceDensityCache(reader, rgb_transform=normalize, max_bytes=cache_bytes)
                if cache_bytes else nullcontext()) as cache:
            density_options = dict(density_cache=cache) if cache is not None else dict(rgb_transform=normalize)
            for i, (y, x) in enumerate(starts):
                local_transform, origin = transform, starts[i]
                if padded:
                    local_transform = transform @ np.asarray([[1., 0., x], [0., 1., y], [0., 0., 1.]])
                    origin = np.asarray([0, 0])
                bounds = dict(row_start=int(origin[0]), row_stop=int(origin[0])+128,
                              column_start=int(origin[1]), column_stop=int(origin[1])+128)
                valid = full_valid_cells(local_transform, **bounds, image_width=reader.width, image_height=reader.height)
                if cfg.get('require_full_FOV_HE', True) and not valid.all():
                    raise ValueError('FOV lacks complete registered H&E coverage')
                density, area = aggregate_density_region(reader, local_transform, **bounds,
                    samples_per_cell_axis=8, integration_rows=integration_rows, **density_options)
                rgb, rgb_valid = sample_registered_rgb(reader, local_transform, origin)
                if padded:
                    yy, xx = y + np.arange(128)[:, None], x + np.arange(128)[None, :]
                    capture = (yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])
                    valid &= capture
                    density[~capture], area[~capture] = 0, 0
                    offsets = -.5 + (np.arange(224) + .5) * 128 / 224
                    rgb_capture = ((y + offsets[:, None] >= -.5) & (y + offsets[:, None] < shape[0] - .5)
                        & (x + offsets[None, :] >= -.5) & (x + offsets[None, :] < shape[1] - .5))
                    rgb[~rgb_capture], rgb_valid[~rgb_capture] = 1, False
                arrays['density_float32'][i] = density
                arrays['area_uint16'][i] = quantize_valid_fraction(area)
                arrays['field_valid_bool'][i] = valid
                arrays['rgb_float32'][i] = rgb.transpose(2, 0, 1)
                arrays['rgb_valid_bool'][i] = rgb_valid
                if (i+1) % 100 == 0:
                    print(f'HE: {i+1}/{n}', flush=True)
            write(destination / 'HE_cache.json', dict(scope='fresh_in_memory_per_method',
                reinhard_diagnostics=False, raw_UNI_unchanged=True, preprocessing_mode=settings['mode'],
                integration_rows=integration_rows,
                density_cache=cache.statistics() if cache is not None else None))
        for array in arrays.values():
            array.flush()
    with timing.phase('UNI_features', cuda=True):
        uni_sha = source['uni_checkpoint_sha256']
        uni = FrozenUNI(source['uni_checkpoint'], config_path=source['uni_config']).to(cfg['device']).eval()
        features = np.lib.format.open_memmap(destination / 'features_float32.npy', mode='w+',
                                             dtype=np.float32, shape=(n, 1024, 14, 14))
        valid = np.lib.format.open_memmap(destination / 'pathology_valid_bool.npy', mode='w+',
                                          dtype=bool, shape=(n, 14, 14))
        # Retain raw FP32 UNI features, with fixed batch and precision settings.
        batch_size = cfg['uni_batch_size']
        for begin in range(0, n, batch_size):
            rgb = torch.from_numpy(np.array(arrays['rgb_float32'][begin:begin+batch_size], copy=True)).to(cfg['device'])
            support = torch.from_numpy(np.array(arrays['rgb_valid_bool'][begin:begin+batch_size], copy=True)).to(cfg['device'])
            features[begin:begin+len(rgb)] = uni(rgb.contiguous(), extent_um=256.0).cpu().numpy()
            valid[begin:begin+len(rgb)] = (F.avg_pool2d(support[:, None].float(), 16, 16)[:, 0] == 1).cpu().numpy()
        features.flush()
        valid.flush()
        del uni, rgb, support
        if torch.device(cfg['device']).type == 'cuda':
            torch.cuda.empty_cache()
    return uni_sha
