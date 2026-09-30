"""Prepare native HD16/pseudo-Visium coarse arrays and separate evaluation truth."""
import argparse
import json
from pathlib import Path
from astra.assets import ROOT
from astra.data.native_benchmark import prepare_benchmark


def main(argv=None, *, prog=None):
    p=argparse.ArgumentParser(prog=prog,description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    print(json.dumps(prepare_benchmark(a.config,a.output,ROOT),indent=2))
