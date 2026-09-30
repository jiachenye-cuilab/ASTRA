"""Run the frozen checkpoint on prepared HD16/pseudo-Visium input, then score saved arrays."""
import argparse
import json
from pathlib import Path
import torch
from astra.runtime import ROOT, configure_cuda
from astra.cli.common import positive_gib
from astra.inference.section import predict_section, read
from astra.inference.benchmark import evaluate


def main(argv=None, *, prog=None):
    p=argparse.ArgumentParser(prog=prog,description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu');p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--gpu-memory-gib',type=positive_gib,default=12.)
    p.add_argument('--position-mask',type=Path)
    a=p.parse_args(argv)
    if a.output.exists(): raise FileExistsError('choose a new benchmark output directory')
    if read(a.inputs/'inputs.json').get('role') not in ('observed16_input','observed_spot55_input'):
        raise ValueError('benchmark requires HD16 or pseudo-Visium prepared coarse inputs')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.device.startswith('cuda'): configure_cuda(torch.device(a.device),a.gpu_memory_gib)
    a.output.mkdir(parents=True)
    report=predict_section(a.inputs,a.output/'prediction',ROOT,device=a.device,batch_size=a.batch_size)
    metrics=evaluate(a.output/'prediction',a.reference,a.output/'metrics.json',position_mask=a.position_mask)
    print(json.dumps(dict(prediction=report,evaluation=metrics),indent=2))
