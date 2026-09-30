import h5py
import numpy as np

def feature_table(path, protocol):
    def decode(values):
        return [v.decode() if isinstance(v, bytes) else str(v) for v in values]
    with h5py.File(path, "r") as handle:
        ids, names, types = (decode(handle["features/" + key][:]) for key in ("id", "name", "feature_type"))
        available = np.ones(len(ids), dtype=bool)
        if protocol == "WT":
            targets = handle["features"].get("target_sets")
            if targets is None or len(targets) != 1:
                raise ValueError("WT must contain one probe target set")
            available[:] = False
            available[np.asarray(targets[next(iter(targets))][:], dtype=np.int64)] = True
        elif protocol != "3prime":
            raise ValueError("unknown assay")
    result = {gene: (i, names[i]) for i, gene in enumerate(ids) if types[i] == "Gene Expression" and available[i]}
    if len(result) != sum(t == "Gene Expression" and ok for t, ok in zip(types, available)):
        raise ValueError("duplicate measurable gene identities")
    return result

def _point_slots(group, lookup, block_columns):
    rows = np.asarray(group["row"][:], dtype=np.int64)
    columns = np.asarray(group["col"][:], dtype=np.int64)
    return rows, columns, np.flatnonzero(lookup[(rows // 256) * block_columns + columns // 256] >= 0)

def stage_counts(root, record, gene_ids, blocks, shape):
    """v027 sparse-block layout; feature availability now follows actual protocol."""
    table = feature_table(record["feature_slice"], record["protocol"])
    sources = [table[gene][0] if gene in table else -1 for gene in gene_ids]
    block_columns = (shape[1] + 255) // 256
    lookup = np.full(((shape[0] + 255) // 256) * block_columns, -1, dtype=np.int32)
    lookup[(blocks[:, 0] // 256) * block_columns + blocks[:, 1] // 256] = np.arange(len(blocks))
    with h5py.File(record["feature_slice"], "r") as handle:
        per_block = np.zeros(len(blocks), dtype=np.int64)
        for source in sources:
            key = f"feature_slices/{source}"
            if source < 0 or key not in handle:
                continue
            rows, cols, chosen = _point_slots(handle[key], lookup, block_columns)
            slots = lookup[(rows[chosen] // 256) * block_columns + cols[chosen] // 256]
            per_block += np.bincount(slots, minlength=len(blocks))
        indptr = np.r_[0, np.cumsum(per_block)]
        arrays = [np.lib.format.open_memmap(root / filename, mode="w+", dtype=np.uint16,
                  shape=(int(indptr[-1]),)) for filename in ("local_linear.npy", "gene_index.npy", "count_uint16.npy")]
        cursor = indptr[:-1].copy()
        for gene, source in enumerate(sources):
            key = f"feature_slices/{source}"
            if source < 0 or key not in handle:
                continue
            group = handle[key]
            rows, cols, chosen = _point_slots(group, lookup, block_columns)
            if not len(chosen):
                continue
            slots = lookup[(rows[chosen] // 256) * block_columns + cols[chosen] // 256]
            order = np.argsort(slots, kind="stable")
            slots, chosen = slots[order], chosen[order]
            values = np.asarray(group["data"][:])[chosen]
            if (not np.isfinite(values).all() or np.any(values < 0) or
                    np.any(values != np.floor(values)) or values.max() > 65535):
                raise ValueError("native counts cannot be represented losslessly as uint16")
            local = (rows[chosen] % 256) * 256 + cols[chosen] % 256
            boundaries = np.flatnonzero(np.r_[True, slots[1:] != slots[:-1], True])
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                slot = slots[left]
                destination = slice(cursor[slot], cursor[slot] + right - left)
                arrays[0][destination] = local[left:right]
                arrays[1][destination] = gene
                arrays[2][destination] = values[left:right]
                cursor[slot] += right - left
        if not np.array_equal(cursor, indptr[1:]):
            raise ValueError("sparse count fill differs from its planned size")
        for array in arrays:
            array.flush()
    np.save(root / "block_indptr.npy", indptr, allow_pickle=False)
    return int(indptr[-1]), [source >= 0 for source in sources]
