import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import astra
from astra.assets import ROOT
from astra.cli import COMMANDS
from astra.data.inputs import load_input
from astra.inference.section_preparation import counts_in_panel
from astra.model.model import Direct8Model
from fixtures import synthetic_package


def cli(arguments,cwd):
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
    env['PYTHONPATH'] = str(Path(astra.__file__).resolve().parents[1])
    command = [sys.executable,'-B','-m','astra']
    return subprocess.run(command+arguments,cwd=cwd,env=env,text=True,capture_output=True,timeout=90)


class SmokeTests(unittest.TestCase):
    def test_package_and_cli_help(self):
        self.assertEqual(ROOT,Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as d:
            for arguments in [['--help'],['--version']] + [[name,'--help'] for name in COMMANDS]:
                with self.subTest(arguments=arguments):
                    result = cli(arguments,d)
                    self.assertEqual(result.returncode,0,result.stderr)

    def test_offline_preparation_and_training_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            for arguments,status in [(['prepare-input','--self-test'],'passed'),(['train','--check-config'],'valid')]:
                result = cli(arguments,d)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(json.loads(result.stdout)['status'],status)

    def test_missing_input_error(self):
        with tempfile.TemporaryDirectory() as d:
            result = cli(['predict','--input','missing.npz','--output','out.npz'],d)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('FileNotFoundError',result.stderr)
            self.assertIn('missing.npz',result.stderr)

    def test_panel_reordering_zero_columns_and_unknown_genes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            np.save(root/'counts.npy',np.array([[11,0,5],[19,0,7]],np.int64))
            (root/'genes.json').write_text(json.dumps(['B','A','outside']))
            counts,available = counts_in_panel(root/'counts.npy',root/'genes.json',['A','B','C'])
            np.testing.assert_array_equal(counts.toarray(),[[0,11,0],[0,19,0]])
            np.testing.assert_array_equal(available,[True,True,False])
            (root/'genes.json').write_text(json.dumps(['B','B','outside']))
            with self.assertRaisesRegex(ValueError,'unique'):
                counts_in_panel(root/'counts.npy',root/'genes.json',['A','B'])

    def test_synthetic_checkpoint_rejects_reordered_gene_identity(self):
        with tempfile.TemporaryDirectory() as d:
            package,genes,_ = synthetic_package(Path(d)/'package')
            config = json.loads((package/'model/config.json').read_text())['kwargs']
            state = torch.load(package/'model/checkpoint.pt',weights_only=True,map_location='cpu')
            model = Direct8Model(**config)
            model.load_state_dict(state,strict=True)
            config['input_gene_ids'] = genes[::-1]
            with self.assertRaises((ValueError,RuntimeError)):
                Direct8Model(**config).load_state_dict(state,strict=True)


@unittest.skipUnless(os.environ.get('ASTRA_RUN_BUNDLED') == '1','enable bundled pretrained replay with ASTRA_RUN_BUNDLED=1')
class BundledTests(unittest.TestCase):
    def test_bundled_checkpoint_and_two_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            for task in ('hd16','spot55'):
                output = Path(d)/(task+'.npz')
                result = cli(['predict','--input',str(ROOT/'examples'/f'{task}_input.npz'),
                              '--output',str(output),'--reference',str(ROOT/'examples'/f'{task}_reference.npz')],d)
                self.assertEqual(result.returncode,0,result.stderr)
                report = json.loads(result.stdout)
                self.assertTrue(report['reference']['passed'])
                self.assertEqual(report['prediction_shape'],[32,32,2000])
                with np.load(output,allow_pickle=False) as archive:
                    self.assertEqual(int(archive['gene_available'].sum()),640)
                genes = json.loads((ROOT/'model/input_gene_ids.json').read_text())
                data = load_input(ROOT/'examples'/f'{task}_input.npz',genes)
                data['gene_ids'] = data['gene_ids'][::-1]
                broken = Path(d)/(task+'_reordered.npz'); np.savez(broken,**data)
                with self.assertRaisesRegex(ValueError,'gene_ids must match'):
                    load_input(broken,genes)


if __name__ == '__main__':
    unittest.main()
