"""Convert filtered Space Ranger H5 plus calibrated CSV positions for frozen Visium inference."""
import argparse
import json
from pathlib import Path
from astra.data.section_inputs import prepare_visium


def main(argv=None, *, prog=None):
    p=argparse.ArgumentParser(prog=prog,description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    print(json.dumps(prepare_visium(a.config,a.output),indent=2))
