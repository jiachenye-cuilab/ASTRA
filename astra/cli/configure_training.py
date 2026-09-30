"""Configure a separate task for training ASTRA from user Visium HD feature slices."""
import argparse
import json
from pathlib import Path
from astra.training.user_data import configure


def main(argv=None, *, prog=None):
    p=argparse.ArgumentParser(prog=prog,description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    print(json.dumps(configure(a.manifest,a.output),indent=2))
