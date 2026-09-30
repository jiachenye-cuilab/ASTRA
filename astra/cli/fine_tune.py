"""Fine-tune ASTRA on one target section's measured HD16 or Spot55 observations."""
import argparse
import json
from pathlib import Path

import torch

from astra.runtime import ROOT, configure_cuda
from astra.cli.common import positive_gib
from astra.fine_tuning.training import fit


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True,help='target sample, task and spatially separate support/selection FOVs')
    parser.add_argument('--output',type=Path,required=True,help='new checkpoint/report directory')
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--gpu-memory-gib',type=positive_gib,default=12.0)
    parser.add_argument('--smoke',action='store_true',help='one update on 16 FOVs; not a completed adaptation')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('output directory exists; choose a new output path')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        configure_cuda(device,args.gpu_memory_gib)
    print(json.dumps(fit(args.manifest,args.output,ROOT,device=device,smoke=args.smoke),indent=2))
