from copy import deepcopy
import hashlib
import json
import numpy as np
import torch

def stable_seed(*parts):
    value = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


def cpu_state(model):
    return {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else deepcopy(v)
            for k, v in model.state_dict().items()}


def epoch_items(spec, epoch, *, smoke=False):
    support = spec.get("adaptation_pool", spec["groups"]["support"])
    if smoke:
        if "adaptation_pool" in spec:
            rng = np.random.default_rng(stable_seed(spec["seed"], spec["name"], "smoke_pool"))
            support = [support[i] for i in rng.permutation(len(support))[:16]]
        else:
            support = support[:16]
    count = 16 if smoke else len(spec["groups"]["support"])
    if len(support) < count:
        raise ValueError("adaptation pool is smaller than the fixed per-epoch exposure budget")
    rng = np.random.default_rng(stable_seed(spec["seed"], spec["name"], epoch))
    return [support[i] for i in rng.permutation(len(support))[:count]]


def pair_choices(fields, selected, spec, epoch, *, selection_view=None):
    result=[]
    for item in selected:
        index=item['field_index'];pairs=fields.metadata['neighbor_pairs'][index]
        if selection_view is None:
            rng=np.random.default_rng(stable_seed(spec['seed'],spec['case'],epoch,index,'spot_pair_training'))
            choice=int(rng.integers(len(pairs)))
        else:
            rng=np.random.default_rng(stable_seed(spec['seed'],spec['case'],index,'spot_pair_selection'))
            choice=int(rng.permutation(len(pairs))[selection_view])
        result.append(pairs[choice])
    return result
