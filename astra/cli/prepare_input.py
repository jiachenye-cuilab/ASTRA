"""Prepare one registered FOV for ASTRA, or check input preprocessing offline."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from astra.runtime import ROOT
from astra.data.preparation import prepare_arrays, self_test


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="observations.npz with counts, geometry and image arrays")
    parser.add_argument("--output", type=Path, help="new prepared input.npz; existing outputs are never overwritten")
    parser.add_argument("--gene-ids", type=Path, default=ROOT / "model/input_gene_ids.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.self_test:
        self_test()
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test is used")
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists():
        raise FileExistsError(args.output)
    panel = json.loads(args.gene_ids.read_text(encoding="utf-8"))
    panel = panel["gene_ids"] if isinstance(panel, dict) else panel
    with np.load(args.input, allow_pickle=False) as source:
        prepared = prepare_arrays(source, panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **prepared)
    print(json.dumps(dict(status="prepared", output=str(args.output),
                          **json.loads(prepared["preparation_metadata_json"].item()))))
