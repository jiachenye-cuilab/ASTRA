"""Cache frozen UNI features for portable, coarse-only ASTRA fine-tuning inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from astra.runtime import ROOT, configure_cuda
from astra.data.inputs import load_input, semantic_input
from astra.cli.common import positive_gib
from astra.fine_tuning.inputs import KEYS


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--uni-checkpoint',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--gpu-memory-gib',type=positive_gib,default=6.)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.suffix.lower() != '.npz':
        raise ValueError('output must be a new .npz file')
    genes = json.loads((ROOT/'model/input_gene_ids.json').read_text(encoding='utf-8'))
    metadata = json.loads((ROOT/'model/metadata.json').read_text(encoding='utf-8'))
    data = load_input(args.input,genes,keys=KEYS+('rgb','rgb_valid'))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        configure_cuda(device,args.gpu_memory_gib)
    with torch.no_grad():
        features,valid,source = semantic_input(data,device,metadata,args.uni_checkpoint)
    output = {k:v for k,v in data.items() if k in KEYS}
    output.update(pathology_features=features[0].cpu().numpy(),pathology_valid=valid[0].cpu().numpy(),
                  uni_checkpoint_sha256=np.asarray(metadata['uni_checkpoint_sha256']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream,**output)
    print(json.dumps(dict(status='prepared',semantic_source=source,output=str(args.output))))
