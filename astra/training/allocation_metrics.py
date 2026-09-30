"""Exact HD16 panel-total/composition NLL decomposition and residual metrics."""
import torch


def _flat_pcc(x, y):
    x, y = x.reshape(-1).double(), y.reshape(-1).double()
    if x.numel() < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    xx, yy = x.square().sum(), y.square().sum()
    if float(xx) <= 1e-20 or float(yy) <= 1e-20:
        return None
    return float(((x * y).sum() / (xx * yy).sqrt()).clamp(-1, 1))


def moments(prediction, truth):
    pp, tt = prediction.square().sum(), truth.square().sum()
    return dict(pcc=_flat_pcc(prediction, truth),
                rms_ratio=float((pp / tt).sqrt()) if tt > 0 else None)


def decompose(prediction, truth, counts):
    """Reuse v032 round005 allocation_diagnostic definitions, without old-version imports."""
    p, y, c = prediction.double(), truth.double(), counts.double()
    for value in (p, y):
        if not bool(torch.isfinite(value).all() & (value >= 0).all()):
            raise ValueError('invalid scored counts')
        torch.testing.assert_close(value.sum(1), c, atol=1e-8, rtol=1e-9)
    baseline = c[:, None] / 4
    pt, yt, ct = p.sum(-1), y.sum(-1), c.sum(-1)
    bt = ct[:, None] / 4
    umi = y.sum()
    if not umi > 0:
        raise ValueError('empty scored FOV')
    log = lambda x: x.clamp_min(1e-12).log()
    full = (baseline - p + y * (log(p) - log(baseline))).sum() / umi
    total = (bt - pt + yt * (log(pt) - log(bt))).sum() / umi
    parent_composition = c / ct[:, None].clamp_min(1)
    pred_composition = p / pt[:, :, None].clamp_min(1e-12)
    composition = (y * (log(pred_composition) - log(parent_composition[:, None]))).sum() / umi
    shared = parent_composition[:, None] * pt[:, :, None]
    shared_gain = (baseline - shared + y * (log(shared) - log(baseline))).sum() / umi
    torch.testing.assert_close(full, total + composition, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(shared_gain, total, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(shared.sum(1), c, atol=1e-8, rtol=1e-9)
    result = dict(full_gain=float(full), total_gain=float(total), composition_gain=float(composition),
                  target_umis=float(umi), complete_parents=len(c),
                  decomposition_error=float((full-total-composition).abs()))
    for name, values in dict(gene=(p-baseline, y-baseline), total=(pt-bt, yt-bt),
            composition=(p-shared, y-parent_composition[:, None]*yt[:, :, None])).items():
        result.update({f'{name}_residual_{key}': value for key, value in moments(*values).items()})
    return result
