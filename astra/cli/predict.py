"""Reconstruct one FOV with frozen UNI features or local UNI1 weights."""
import argparse
from pathlib import Path

from .common import positive_gib
from astra.inference.predict import predict


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uni-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cuda:0", "cpu"))
    parser.add_argument("--gpu-memory-gib", type=positive_gib, default=6.0,
                        help="CUDA allocator budget in GiB (default: 6; also capped at 75%% of device memory)")
    parser.add_argument("--reference", type=Path)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--fine-tuned", type=Path, help="optional target-specific fine-tuned checkpoint")
    checkpoints.add_argument("--trained-checkpoint", type=Path, help="complete best.pt or last.pt produced by python -m astra train")
    parser.add_argument("--sample-id", help="target sample identity recorded by python -m astra fine-tune")
    parser.add_argument("--task", choices=("HD16", "Spot55"))
    args = parser.parse_args(argv)
    if bool(args.fine_tuned) != bool(args.sample_id and args.task):
        parser.error("--fine-tuned requires both --sample-id and --task")
    if not args.fine_tuned and (args.sample_id or args.task):
        parser.error("--sample-id and --task require --fine-tuned")
    predict(args)
