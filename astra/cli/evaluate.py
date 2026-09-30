"""Evaluate saved prediction arrays against measured fine-grid reference; no fitting."""
import argparse
import json
from pathlib import Path
from astra.inference.benchmark import evaluate


def main(argv=None, *, prog=None):
    p=argparse.ArgumentParser(prog=prog,description=__doc__)
    p.add_argument('--prediction',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--position-mask',type=Path,help='fixed cross-method common support in reference coordinate order')
    a=p.parse_args(argv)
    print(json.dumps(evaluate(a.prediction,a.reference,a.output,position_mask=a.position_mask),indent=2))
