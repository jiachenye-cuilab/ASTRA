"""Reaggregate serialized grids; report rasterization error without changing counts."""
from pathlib import Path
import json

import numpy as np

ATOL = 1e-5
RTOL = 1e-5


def count_errors(predicted, observed):
    predicted, observed = np.asarray(predicted, np.float64), np.asarray(observed, np.float64)
    if predicted.shape != observed.shape or not np.isfinite(predicted).all():
        raise ValueError("reaggregation shape or finite values differ")
    error = np.abs(predicted - observed)
    nonzero = np.abs(observed) > 0
    return dict(compared_values=int(error.size), max_absolute=float(error.max(initial=0)),
                max_scaled=float((error / (1 + np.abs(observed))).max(initial=0)),
                max_relative_nonzero=float((error[nonzero] / np.abs(observed[nonzero])).max(initial=0)),
                within_tolerance=bool(np.all(error <= ATOL + RTOL * np.abs(observed))) if error.size else None,
                atol=ATOL, rtol=RTOL)


def _merge(reports):
    reports = list(reports)
    return dict(compared_values=sum(r['compared_values'] for r in reports),
                **{key: max((r[key] for r in reports), default=0.) for key in
                   ('max_absolute', 'max_scaled', 'max_relative_nonzero')},
                within_tolerance=all(r['within_tolerance'] for r in reports) if reports else None, atol=ATOL, rtol=RTOL)


def fov_grid_diagnostics(path, data):
    """Uniform-density observation operator on the input's valid 2um support."""
    with np.load(path, allow_pickle=False) as archive:
        grid = archive['pred_count_8um'].astype(np.float64)
        available = archive['gene_available']
        if not np.array_equal(archive['gene_ids'], data['gene_ids']) or not np.array_equal(available, data['gene_available']):
            raise ValueError('serialized FOV panel differs from its input')
    owner, valid = data['owner_map'], data['field_valid']
    rows, cols = valid.shape
    if grid.shape != (rows // 4, cols // 4, len(available)):
        raise ValueError('serialized FOV grid shape differs')
    if not np.isfinite(grid).all() or np.any(grid < 0):
        raise ValueError('serialized counts must be finite and nonnegative')
    area = valid.reshape(rows // 4, 4, cols // 4, 4).sum((1, 3))
    parents = np.flatnonzero(data['parent_valid'])
    predicted = []
    for parent in parents:
        overlap = ((owner == parent) & valid).reshape(rows // 4, 4, cols // 4, 4).sum((1, 3))
        weight = np.divide(overlap, area, out=np.zeros_like(area, dtype=float), where=area > 0)
        predicted.append(np.einsum('yx,yxg->g', weight, grid[..., available]))
    observed = data['parent_counts'][np.ix_(parents, available)]
    errors = count_errors(np.asarray(predicted).reshape(observed.shape), observed)
    # Grid sums retain exact parent identity only if no output cell mixes owners/gaps.
    homogeneous = True
    for y, x in np.argwhere(area > 0):
        support = valid[4*y:4*y+4, 4*x:4*x+4]
        if len(np.unique(owner[4*y:4*y+4, 4*x:4*x+4][support])) > 1:
            homogeneous = False
            break
    if homogeneous and errors['compared_values'] and not errors['within_tolerance']:
        raise ValueError('serialized homogeneous FOV grid does not conserve observed counts')
    return dict(status='evaluated' if errors['compared_values'] else 'not_evaluated',
                operator='uniform_density_on_valid_2um_raster', exact_representation=homogeneous,
                support='input owner_map & field_valid; measured genes and valid parents only', **errors)


def spot_grid_weights(center, diameter_um=55.):
    """Areas of center-rasterized 2um circle pixels in complete 8um cells / 64um²."""
    center = np.asarray(center, np.float64)
    radius = diameter_um / 2
    low = np.ceil((center - radius - 1) / 2).astype(np.int64)
    high = np.floor((center + radius - 1) / 2).astype(np.int64)
    yy, xx = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1), indexing='ij')
    keep = ((2*yy+1-center[0])**2 + (2*xx+1-center[1])**2) <= radius**2
    cells, sizes = np.unique(np.column_stack((yy[keep], xx[keep])) // 4, axis=0, return_counts=True)
    return cells, sizes.astype(np.float64) / 16


def section_grid_diagnostics(output, fields):
    """Read final NPY files; compare full matching supports, never partial/full totals."""
    output = Path(output)
    prediction = np.load(output / 'prediction.npy', mmap_mode='r', allow_pickle=False)
    available = np.load(output / 'gene_available.npy', allow_pickle=False)
    if not np.array_equal(available, fields.available):
        raise ValueError('serialized section gene availability differs')
    if json.loads((output / 'gene_ids.json').read_text(encoding='utf-8')) != fields.metadata['output_gene_ids']:
        raise ValueError('serialized section gene order differs')
    reports = []
    if fields.task == 'HD16':
        assignment = np.load(fields.root / 'assignment.npy', allow_pickle=False)
        local = np.load(fields.root / 'local_parent.npy', allow_pickle=False)
        coordinates = np.load(output / 'parent_yx_16um.npy', allow_pickle=False)
        if not np.array_equal(coordinates, np.load(fields.root / 'parent_yx_16um.npy', allow_pickle=False)):
            raise ValueError('serialized HD16 coordinates differ')
        if prediction.shape != (len(coordinates), 4, len(available)):
            raise ValueError('serialized HD16 shape differs')
        for start in range(0, len(prediction), 128):
            stop = start + 128
            values = prediction[start:stop]
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('serialized counts must be finite and nonnegative')
            truth = fields.counts[assignment[start:stop] * 256 + local[start:stop]].toarray()
            reports.append(count_errors(values.sum(1, dtype=np.float64)[:, available], truth[:, available]))
        errors = _merge(reports)
        if errors['compared_values'] and not errors['within_tolerance']:
            raise ValueError('serialized HD16 grid does not conserve complete parent counts')
        return dict(status='evaluated' if errors['compared_values'] else 'not_evaluated',
                    operator='sum_four_complete_8um_children', exact_representation=True, **errors)

    # Old external caches may omit the original section observations. Do not invent them.
    observations = fields.root / 'observations'
    if not (observations / 'spot_yx_um.npy').is_file():
        return dict(status='not_evaluated', reason='original Spot55 observations missing from prepared cache',
                    exact_representation=False)
    centers = np.load(observations / 'spot_yx_um.npy', allow_pickle=False)
    truth = np.load(observations / 'parent_counts.npy', mmap_mode='r', allow_pickle=False)
    queries = np.load(output / 'query_yx_8um.npy', allow_pickle=False)
    if (not np.array_equal(queries, np.load(fields.root / 'query_yx_8um.npy', allow_pickle=False))
            or not np.array_equal(available, np.load(observations / 'gene_available.npy', allow_pickle=False))):
        raise ValueError('serialized Spot55 coordinates or observation panel differ')
    if prediction.shape != (len(queries), len(available)) or truth.shape != (len(centers), int(available.sum())):
        raise ValueError('serialized Spot55 shape or measured count panel differs')
    for start in range(0, len(prediction), 128):
        values = prediction[start:start+128]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('serialized counts must be finite and nonnegative')
    # Sorted integer keys avoid a per-query Python dictionary for large sections.
    width = int(queries[:, 1].max(initial=0)) + 1
    keys = queries[:, 0] * width + queries[:, 1]
    order = np.argsort(keys)
    keys = keys[order]
    complete, partial, empty, coverages, areas = 0, 0, 0, [], []
    for index, center in enumerate(centers):
        cells, weights = spot_grid_weights(center)
        areas.append(float(weights.sum()*64))
        target = cells[:, 0] * width + cells[:, 1]
        found = np.searchsorted(keys, target)
        inside = (cells >= 0).all(1) & (cells[:, 1] < width) & (found < len(keys))
        matched = np.zeros(len(cells), bool)
        matched[inside] = keys[found[inside]] == target[inside]
        coverage = float(weights[matched].sum() / weights.sum()) if weights.size else 0.
        coverages.append(coverage)
        if not matched.any():
            empty += 1
        elif not matched.all():
            partial += 1
        else:
            complete += 1
            values = prediction[order[found]][:, available].astype(np.float64)
            reports.append(count_errors(weights @ values, truth[index]))
    errors = _merge(reports)
    return dict(status='evaluated' if complete else 'not_evaluated',
                reason=None if complete else 'no full discrete spot support in exported queries',
                operator='uniform_density_8um_grid_over_center_rasterized_55um_circles',
                exact_representation=False, complete_spots=complete, partial_spots_excluded=partial,
                empty_spots_excluded=empty, min_support_fraction=min(coverages, default=0.),
                max_support_fraction=max(coverages, default=0.),
                nominal_circle_area_um2=float(np.pi*27.5**2),
                raster_area_um2_min=min(areas, default=0.), raster_area_um2_max=max(areas, default=0.), **errors)
