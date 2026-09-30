"""Published circular observation geometry, independent of model and expression."""
import math
import numpy as np
from astra.training.utils import stable_seed


def lattice_origin(sample_id, seed=20260907):
    u, v = np.random.default_rng(stable_seed(seed, sample_id, 'v030_visium_lattice')).random(2)
    return float(v * 50 * math.sqrt(3)), float(u * 100 + v * 50)


def lattice(shape, origin):
    centers, array = [], []
    height, width = np.asarray(shape) * 2
    ay, ax = origin
    for row in range(-1, int(np.ceil(height / (50 * np.sqrt(3)))) + 1):
        y, offset = ay + row * 50 * np.sqrt(3), ax + row * 50
        for col in range(int(np.floor(-offset / 100)) - 1, int(np.ceil((width - offset) / 100)) + 1):
            x = offset + col * 100
            if 28 <= y <= height - 28 and 28 <= x <= width - 28:
                centers.append((y, x)); array.append((row, col * 2 + row))
    return np.asarray(centers, float).reshape(-1, 2), np.asarray(array, np.int64).reshape(-1, 2)


def circle_bins(centers, shape, diameter_um=55):
    centers, shape = np.asarray(centers, float), np.asarray(shape, np.int64)
    radius = diameter_um / 2
    if (centers.ndim != 2 or centers.shape[1] != 2 or not len(centers)
            or not np.isfinite(centers).all() or np.any(centers < radius)
            or np.any(centers + radius > shape * 2)):
        raise ValueError('observations need finite complete capture circles')
    flats, owners = [], []
    for i, center in enumerate(centers):
        low = np.maximum(0, np.floor((center-radius)/2).astype(int))
        high = np.minimum(shape, np.ceil((center+radius)/2).astype(int))
        yy, xx = np.meshgrid(np.arange(low[0], high[0]), np.arange(low[1], high[1]), indexing='ij')
        keep = (2*yy+1-center[0])**2 + (2*xx+1-center[1])**2 <= radius**2
        flats.append(yy[keep]*shape[1]+xx[keep]); owners.append(np.full(keep.sum(), i, np.int64))
    flat, owner = np.concatenate(flats), np.concatenate(owners)
    order = np.argsort(flat)
    if np.any(np.diff(flat[order]) == 0):
        raise ValueError('overlapping capture circles would double-count observations')
    return flat[order], owner[order]


def interior_queries(centers, diameter_um=55):
    """Canonical 8um squares wholly inside a continuous capture circle."""
    queries, owners = [], []
    radius = diameter_um/2
    for i, center in enumerate(np.asarray(centers, float)):
        low = np.maximum(0, np.floor((center-radius)/8).astype(int))
        high = np.ceil((center+radius)/8).astype(int)
        yy, xx = np.meshgrid(np.arange(low[0], high[0]), np.arange(low[1], high[1]), indexing='ij')
        cells = np.column_stack((yy.ravel(), xx.ravel()))
        keep = ((np.abs(cells*8+4-center)+4)**2).sum(1) <= radius**2
        queries.append(cells[keep]); owners.append(np.full(keep.sum(), i, 'i8'))
    if not queries:
        raise ValueError('no complete capture circles for evaluation')
    return np.concatenate(queries), np.concatenate(owners)


def st100_owner(start, center_yx):
    axis = (np.arange(128)+.5)*2
    local = np.asarray(center_yx)-np.asarray(start)*2
    owner = np.where((axis[:, None]-local[0])**2 + (axis[None, :]-local[1])**2 <= 2500, 0, -1).astype('i2')
    area = (owner == 0).reshape(32, 4, 32, 4).sum((1, 3))
    if not area.sum() or area.sum() != area[6:26, 6:26].sum():
        raise ValueError('complete 100um anchor must lie inside the central 160um core')
    return owner


def st100_geometry(centers):
    """Same nearest-8um centers, +/-24um grid9 and mean coverage as TNBC inputs."""
    centers = np.asarray(centers, float)
    if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError('provide finite [N,2] centers in physical y,x microns')
    base = ((np.floor(centers/8+.5)*8-128)/2).astype('i4')
    shifts = np.array([(y, x) for y in (-24, 0, 24) for x in (-24, 0, 24)], 'i4')//2
    starts = (base[:, None]+shifts[None]).reshape(-1, 2)
    anchors = np.repeat(np.arange(len(base)), 9)
    lower, upper = (starts//4+6).min(0), (starts//4+26).max(0)
    shape = upper-lower
    if np.prod(shape, dtype=np.int64) > 64*2**20:
        raise ValueError('ST100 grid bounding box exceeds 64M cells; partition this section')
    delta = np.zeros(shape+1, 'i4'); p = starts//4+6-lower
    for dy, dx, sign in [(0,0,1),(20,0,-1),(0,20,-1),(20,20,1)]:
        np.add.at(delta, (p[:,0]+dy, p[:,1]+dx), sign)
    weight = delta.cumsum(0).cumsum(1)[:-1, :-1]
    yx = np.argwhere(weight > 0)+lower
    index = np.full(shape, -1, 'i4'); index[weight > 0] = np.arange(len(yx))
    parent, areas = np.full(len(yx), -1, 'i4'), np.zeros(len(yx), 'u1')
    for j, start in enumerate(base):
        y, x = start//4+6-lower
        idx = index[y:y+20, x:x+20].ravel()
        area = (st100_owner(start, centers[j]) == 0).reshape(32,4,32,4).sum((1,3))[6:26,6:26].ravel()
        use = area > 0; target, values = idx[use], area[use]
        empty = parent[target] == -1
        parent[target[empty]], areas[target[empty]] = j, values[empty]
        parent[target[~empty]], areas[target[~empty]] = -2, 0
    return dict(starts=starts, anchors=anchors, query_yx=yx.astype('i4'),
                coverage=weight[weight > 0].astype('i4'), parent=parent, areas=areas, lower=lower, index=index)


def st100_field_indices(geometry, start):
    y, x = start//4+6-geometry['lower']
    return geometry['index'][y:y+20, x:x+20].ravel()
