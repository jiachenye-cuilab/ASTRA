from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import astra
from astra.assets import ROOT
from astra.cli import COMMANDS
from astra.data.inputs import load_input
from astra.data.preparation import DENSITY_METHOD, RGB_SAMPLING, prepare_arrays
from astra.inference.section_preparation import counts_in_panel
from astra.model.model import Direct8Model
from astra.training.checkpoint import load_checkpoint, save_checkpoint
from astra.training.generator import GeneratorConfig, OwnerMapSample
from astra.training.model_factory import model_class, model_family
from astra.training.parent_sampling import random_owner_with_resampling
from astra.training.train import configuration_version, resume_config_matches
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


class TrainingNamingTests(unittest.TestCase):
    def test_public_model_name_preserves_checkpoint_compatibility(self):
        kwargs = json.loads((ROOT/'model/config.json').read_text())['kwargs']
        kwargs.update(input_gene_ids=['synthetic_A','synthetic_B'],
                      output_gene_ids=['synthetic_A','synthetic_B'])
        torch.manual_seed(930)
        model = model_class('ASTRA')(**kwargs)
        self.assertIs(model_class(model_family(model)),type(model))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'checkpoint.pt'
            save_checkpoint(path,model,step=3)
            restored,payload = load_checkpoint(path,expected_panels=model.panels.as_dict())
            self.assertEqual(payload['step'],3)
            for key,value in model.state_dict().items():
                other = restored.state_dict()[key]
                if isinstance(value,torch.Tensor):
                    self.assertTrue(torch.equal(value,other),key)
                else:
                    self.assertEqual(value,other)

    def test_resume_accepts_renamed_descriptions_but_rejects_setting_changes(self):
        current = json.loads((ROOT/'training/config.json').read_text())
        previous = deepcopy(current)
        previous['model_family'] = configuration_version(ROOT/'training/config.json')
        for key in ('initialization','panel','formal_training'):
            previous['boundaries'][key] = 'original descriptive label'
        self.assertTrue(resume_config_matches(previous,current))
        for section,key,value in [('training','learning_rate',0.01),
                                  ('training','formal_training',False),
                                  ('boundaries','test_expression_read',True)]:
            with self.subTest(section=section,key=key):
                changed = deepcopy(current)
                changed[section][key] = value
                self.assertFalse(resume_config_matches(previous,changed))
        changed = deepcopy(current)
        changed['panel_artifact'] = 'training/different_panel.json'
        self.assertFalse(resume_config_matches(previous,changed))

    def test_named_generator_failure_retains_deterministic_retry(self):
        generator = GeneratorConfig()
        sample = OwnerMapSample(torch.zeros((1,1),dtype=torch.long),torch.ones(1,dtype=torch.bool),{})
        failure = RuntimeError(f'ASTRA generator failed after {generator.maximum_attempts} attempts for seed 123')
        with patch('astra.training.parent_sampling.generate_random_owner',side_effect=[failure,sample]) as generate:
            result = random_owner_with_resampling(123,generator,0)
            self.assertIs(result,sample)
            self.assertEqual(generate.call_count,2)
            self.assertEqual(generate.call_args_list[1].args[0],5735505987267529465)
            self.assertEqual(result.parameters['seed_resample_index'],1)
        with patch('astra.training.parent_sampling.generate_random_owner',side_effect=RuntimeError('unrelated failure')) as generate:
            with self.assertRaisesRegex(RuntimeError,'unrelated failure'):
                random_owner_with_resampling(123,generator,0)
            self.assertEqual(generate.call_count,1)


class SingleFieldCountTests(unittest.TestCase):
    def setUp(self):
        self.genes = json.loads((ROOT/'model/input_gene_ids.json').read_text())
        self.source = dict(
            parent_counts=np.array([[7.,3.],[0.,0.]]), gene_ids=np.asarray(self.genes[:2][::-1]),
            owner_map=np.zeros((128,128),np.int64), parent_valid=np.array([True,False]),
            field_valid=np.ones((128,128),bool), protocol_id=np.array(1,np.int64),
            cell_um=np.array(2.), physical_extent_um=np.array([256.,256.]),
            density_2um=np.ones((128,128,3),np.float32), valid_area_2um=np.full((128,128),65535,np.uint16),
            density_method=np.array(DENSITY_METHOD), rgb=np.ones((3,224,224),np.float32),
            rgb_valid=np.ones((224,224),bool), rgb_sampling=np.array(RGB_SAMPLING))
        self.prepared = prepare_arrays(self.source,self.genes)

    def test_fractional_counts_rejected_during_preparation_and_loading(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'input.npz'
            for value in (.25,np.nextafter(7.,8.)):
                with self.subTest(value=value):
                    source = dict(self.source,parent_counts=self.source['parent_counts'].copy())
                    source['parent_counts'][0,0] = value
                    with self.assertRaisesRegex(ValueError,'raw integer UMI counts'):
                        prepare_arrays(source,self.genes)
                    prepared = dict(self.prepared,parent_counts=self.prepared['parent_counts'].astype(np.float64))
                    prepared['parent_counts'][0,0] = value
                    np.savez(path,**prepared)
                    with self.assertRaisesRegex(ValueError,'raw integer UMI counts'):
                        load_input(path,self.genes)

    def test_integer_values_normalize_integer_and_float_storage_to_int64(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'input.npz'
            for dtype in (np.int64,np.uint64,np.float32,np.float64):
                with self.subTest(dtype=dtype):
                    source = dict(self.source,parent_counts=self.source['parent_counts'].astype(dtype))
                    prepared = prepare_arrays(source,self.genes)
                    self.assertEqual(prepared['parent_counts'].dtype,np.dtype(np.int64))
                    np.testing.assert_array_equal(prepared['parent_counts'],self.prepared['parent_counts'])
                    prepared['parent_counts'] = prepared['parent_counts'].astype(dtype)
                    np.savez(path,**prepared)
                    loaded = load_input(path,self.genes)
                    self.assertEqual(loaded['parent_counts'].dtype,np.dtype(np.int64))
                    np.testing.assert_array_equal(loaded['parent_counts'][:,:2],[[3,7],[0,0]])
                    np.testing.assert_array_equal(loaded['gene_available'][:3],[True,True,False])

    def test_large_integer_counts_preserved_without_float_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'input.npz'
            for dtype in (np.int64,np.uint64):
                with self.subTest(dtype=dtype):
                    source = dict(self.source,parent_counts=self.source['parent_counts'].astype(dtype))
                    source['parent_counts'][0,0] = 2**53+1
                    prepared = prepare_arrays(source,self.genes)
                    self.assertEqual(int(prepared['parent_counts'][0,1]),2**53+1)
                    np.savez(path,**prepared)
                    loaded = load_input(path,self.genes)
                    self.assertEqual(int(loaded['parent_counts'][0,1]),2**53+1)

    def test_out_of_range_counts_rejected_before_int64_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'input.npz'
            for dtype,value in ((np.int64,np.iinfo(np.int64).max),(np.uint64,2**63),
                                (np.float32,2**63),(np.float64,2**63)):
                with self.subTest(dtype=dtype):
                    source = dict(self.source,parent_counts=self.source['parent_counts'].astype(dtype))
                    source['parent_counts'][0,0] = value
                    with self.assertRaisesRegex(ValueError,'int64 count range'):
                        prepare_arrays(source,self.genes)
                    prepared = dict(self.prepared,parent_counts=self.prepared['parent_counts'].astype(dtype))
                    prepared['parent_counts'][0,0] = value
                    np.savez(path,**prepared)
                    with self.assertRaisesRegex(ValueError,'int64 count range'):
                        load_input(path,self.genes)

    def test_preparation_masks_invalid_parents_and_unavailable_genes(self):
        source = dict(self.source,parent_counts=np.array([[7.,.25],[.5,np.nan]]),
                      gene_available=np.array([True,False]))
        prepared = prepare_arrays(source,self.genes)
        expected = np.zeros_like(prepared['parent_counts']);expected[0,1] = 7
        np.testing.assert_array_equal(prepared['parent_counts'],expected)
        np.testing.assert_array_equal(prepared['gene_available'][:3],[False,True,False])

    def test_cli_rejects_invalid_counts_before_writing_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for command,data in (('prepare-input',self.source),('predict',self.prepared)):
                for value in (.25,-.25,np.nan,np.inf):
                    with self.subTest(command=command,value=value):
                        broken = dict(data,parent_counts=data['parent_counts'].astype(np.float64))
                        broken['parent_counts'][0,0] = value
                        path = root/(command+'_input.npz');np.savez(path,**broken)
                        output = root/(command+'_output.npz')
                        result = cli([command,'--input',str(path),'--output',str(output)],root)
                        self.assertNotEqual(result.returncode,0)
                        self.assertIn('counts',result.stderr)
                        self.assertFalse(output.exists())


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
