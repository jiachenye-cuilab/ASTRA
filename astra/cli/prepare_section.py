"""Prepare whole-section measured HD16/Spot55 counts and registered raw H&E for ASTRA."""
import argparse
import json
from pathlib import Path
import torch
from astra.runtime import ROOT, configure_cuda
from astra.cli.common import positive_gib
from astra.inference.section_preparation import prepare


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--uni-batch-size',type=int,default=16)
    parser.add_argument('--gpu-memory-gib',type=positive_gib,default=8.)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith('cuda'):
        configure_cuda(torch.device(args.device),args.gpu_memory_gib)
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if config.get('task') == 'ST100':
        from astra.inference.st100 import prepare as prepare_st100
        report=prepare_st100(args.config,args.output,ROOT,device=args.device,uni_batch_size=args.uni_batch_size)
    else:
        report=prepare(args.config,args.output,ROOT,device=args.device,uni_batch_size=args.uni_batch_size)
    print(json.dumps(report,indent=2))
