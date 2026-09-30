"""Whole-section ASTRA FP32 inference with batching, prefetch and core stitching."""
import argparse
import json
from pathlib import Path
import torch
from astra.runtime import ROOT, configure_cuda
from astra.cli.common import positive_gib
from astra.inference.section import predict_section


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True,help='prepared section cache from python -m astra prepare-section')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fine-tuned',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--pipeline',choices=('cpu','device'),default='device')
    parser.add_argument('--gpu-memory-gib',type=positive_gib,default=8.)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith('cuda'):
        configure_cuda(torch.device(args.device),args.gpu_memory_gib)
    report = predict_section(args.inputs,args.output,ROOT,device=args.device,batch_size=args.batch_size,
                             pipeline=args.pipeline,fine_tuned=args.fine_tuned)
    print(json.dumps(report,indent=2))
