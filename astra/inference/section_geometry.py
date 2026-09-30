import numpy as np
import torch
from scipy import sparse
from astra.fine_tuning.geometry import OwnerMapSample, masks_to_owner


def complete_spot_owner(centers_yx_um, *, fov_origin_yx_um, diameter_um=55.0, device="cpu"):
    """Bin centers discretize circles only after continuous full-support filtering.

    Global physical bin edges are 2*i um; centers are 2*i+1 um. Counts are
    aggregated by the existing collator over each retained complete support.
    """
    centers = np.asarray(centers_yx_um, dtype=np.float64).reshape(-1, 2)
    origin = np.asarray(fov_origin_yx_um, dtype=np.float64)
    if origin.shape != (2,) or not np.isfinite(centers).all() or not np.isfinite(origin).all() or not 0 < diameter_um <= 256:
        raise ValueError("invalid registered spot geometry")
    local = centers - origin
    radius = diameter_um / 2
    keep = np.all((local >= radius) & (local <= 256 - radius), axis=1)
    retained = np.flatnonzero(keep)
    if not len(retained):
        return OwnerMapSample(torch.full((128, 128), -1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.bool, device=device), dict(task="Spot55", retained_indices=[]))
    c = torch.tensor(local[keep], dtype=torch.float64, device=device)
    axis = (torch.arange(128, dtype=torch.float64, device=device) + 0.5) * 2
    masks = ((axis[None, :, None] - c[:, 0, None, None]).square()
             + (axis[None, None, :] - c[:, 1, None, None]).square()) <= radius**2
    owner, valid = masks_to_owner(masks, check_connectivity=True)
    return OwnerMapSample(owner, valid, dict(task="Spot55", retained_indices=retained.tolist(),
        centers_yx_um=centers[keep].tolist(), diameter_um=diameter_um,
        boundary_policy="complete_spots_only", count_scope="full_retained_spot"))


def tile_plan(parents, shape, *, core_um=None):
    """Use fixed supervised cores when requested; retain legacy nearest-center ownership otherwise."""
    parents = np.asarray(parents, dtype=np.int64)
    if core_um is not None:
        if not isinstance(core_um, (int, np.integer)) or not 0 < core_um <= 256 or core_um % 32:
            raise ValueError("HD16 core must align complete 16um parents on both sides of a 256um FOV")
        # Pad context outside capture so border parents also stay inside the trained core.
        indices = np.flatnonzero(((parents >= 0) & ((parents + 1) * 8 <= np.asarray(shape))).all(1))
        core_parents = core_um // 16
        starts = (parents[indices] // core_parents) * core_parents * 8 - (256 - core_um) // 4
        origins, assignment = np.unique(starts, axis=0, return_inverse=True)
        local = (parents[indices] * 8 - origins[assignment]) // 8
        return origins, indices, assignment, local[:, 0] * 16 + local[:, 1]
    axes = []
    for length in shape:
        last = (int(length) - 128) // 8 * 8
        if last < 0:
            raise ValueError("capture must contain at least one full 256um FOV")
        axes.append(np.unique(np.r_[np.arange(0, last + 1, 96), last]))
    starts = np.column_stack([axis[np.abs(parents[:, dim, None] * 8 + 4 - (axis + 64)).argmin(1)]
                              for dim, axis in enumerate(axes)])
    keep = ((parents * 8 >= starts) & (parents * 8 + 8 <= starts + 128)).all(1)
    indices = np.flatnonzero(keep)
    origins, assignment = np.unique(starts[keep], axis=0, return_inverse=True)
    local = (parents[keep] * 8 - origins[assignment]) // 8
    return origins, indices, assignment, local[:, 0] * 16 + local[:, 1]


def core_tiles(cells, shape_yx_2um, *, core_um=192, margin_um=32):
    """Every query belongs to one fixed core, independently of parent geometry."""
    if (core_um,margin_um) not in ((192,32),(144,56)):
        raise ValueError("expected 192/32 or 144/56 core/context in a 256um input field")
    core_cells = core_um//8
    core_start = (np.asarray(cells)//core_cells)*core_cells
    last = (np.asarray(shape_yx_2um)-128)//4*4
    if np.any(last < 0):
        raise ValueError("capture must cover one complete input FOV")
    desired = core_start*4-margin_um//2
    if core_um == 192:
        desired = np.clip(desired, 0, last)
    origins, assignment = np.unique(desired, axis=0, return_inverse=True)
    local = cells-origins[assignment]//4
    if np.any(local < 0) or np.any(local >= 32):
        raise ValueError("query lies outside its core input field")
    if core_um == 144 and not ((local >= 7) & (local < 25)).all():
        raise ValueError("padded boundary queries must remain in the central 144um output")
    return origins, assignment


def prediction_tiles(cells, shape_yx_2um, geometry):
    """Return every core contribution; overlapping predictions have equal weight."""
    cells = np.asarray(cells)
    core, stride = geometry["core_um"] // 8, geometry["stride_um"] // 8
    if geometry["core_um"] == 192:
        origins, assignment = core_tiles(cells, shape_yx_2um)
        return origins, np.arange(len(cells)), assignment, np.ones(len(cells))
    if (core, stride, geometry["context_margin_um"]) not in ((18, 18, 56), (18, 16, 56), (20, 18, 48)):
        raise ValueError("unsupported circular-observation core/stride/context combination")
    if (cells.ndim != 2 or cells.shape[1] != 2 or cells.dtype.kind not in "iu"
            or np.any(cells < 0) or np.any((cells + 1) * 4 > np.asarray(shape_yx_2um))):
        raise ValueError("queries must be complete native 8um capture cells")
    starts, indices = [], []
    for dy in range((core + stride - 1) // stride):
        for dx in range((core + stride - 1) // stride):
            start = (cells // stride - (dy, dx)) * stride
            keep = ((start >= 0) & (cells >= start) & (cells < start + core)).all(1)
            starts.append(start[keep] * 4 - geometry["context_margin_um"] // 2)
            indices.append(np.flatnonzero(keep))
    query = np.concatenate(indices)
    origins, assignment = np.unique(np.concatenate(starts), axis=0, return_inverse=True)
    multiplicity = np.bincount(query, minlength=len(cells))
    if not (multiplicity > 0).all():
        raise ValueError("core grid leaves uncovered queries")
    local = cells[query] - origins[assignment] // 4
    margin = geometry["context_margin_um"] // 8
    if not ((local >= margin) & (local < margin + core)).all():
        raise ValueError("contribution lies outside the supervised central core")
    return origins, query, assignment, 1.0 / multiplicity[query]


def coarse_observations(centers, counts, available, origin, field, *, return_indices=False):
    sample = complete_spot_owner(centers, fov_origin_yx_um=np.asarray(origin)*2)
    retained = np.asarray(sample.parameters["retained_indices"], dtype=int)
    owner = sample.owner_map.numpy().copy()
    valid = np.asarray([np.all(field[owner == i]) for i in range(len(retained))], dtype=bool)
    owner[np.isin(owner, np.flatnonzero(~valid))] = -1
    # The model accepts padding slots but requires at least one parent slot.
    coarse = np.zeros((max(1, len(retained)), len(available)), dtype=np.float64)
    if len(retained):
        coarse[np.ix_(valid, available)] = counts[retained[valid]]
    else:
        valid = np.zeros(1, dtype=bool)
    return (coarse, valid, owner, retained) if return_indices else (coarse, valid, owner)


def spot_counts(case, destination, starts, *, neighbor_distance=110.):
    """Pack complete measured circles only; gaps and partial circles carry no labels."""
    centers = np.load(case / 'spot_yx_um.npy')
    counts = np.load(case / 'parent_counts.npy', mmap_mode='r')
    available = np.load(case / 'gene_available.npy')
    fields = np.load(destination / 'field_valid_bool.npy', mmap_mode='r')
    owners, valid_rows, matrices, pairs = [], [], [], []
    for i, origin in enumerate(starts):
        coarse, valid, owner, retained = coarse_observations(centers, counts, available, origin, fields[i], return_indices=True)
        matrices.append(sparse.csr_matrix(coarse.astype(np.int64)))
        owners.append(owner.astype(np.int16)); valid_rows.append(valid)
        xy = centers[retained]
        distance = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        eligible = (distance > 0) & (distance <= neighbor_distance)
        if len(retained): eligible &= valid[:, None] & valid[None, :]
        pairs.append(np.argwhere(np.triu(eligible, k=1)).tolist())
    slots = max(len(v) for v in valid_rows)
    valid = np.zeros((len(starts), slots), bool)
    padded = []
    for i, matrix in enumerate(matrices):
        valid[i, :len(valid_rows[i])] = valid_rows[i]
        padded.append(sparse.vstack((matrix, sparse.csr_matrix((slots - matrix.shape[0], len(available)), dtype=np.int64))))
    matrix = sparse.vstack(padded, format='csr'); matrix.eliminate_zeros()
    for name, value in dict(counts55_indptr=matrix.indptr.astype(np.int64), counts55_indices=matrix.indices.astype(np.int32),
            counts55_data=matrix.data.astype(np.int64), spot_owner_int16=np.stack(owners), parent_valid_bool=valid,
            starts_yx=starts).items():
        np.save(destination / (name + '.npy'), value)
    return available, slots, pairs
