"""Portable synthetic protocols; fixtures are not biological or paper benchmark evidence."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image
from scipy import sparse
import torch

from fixtures import synthetic_package, write_json
from astra.assets import ROOT
from astra.data.native_benchmark import prepare_benchmark
from astra.data.assay_geometry import st100_geometry, st100_owner, st100_field_indices, interior_queries
from astra.data.image_features import image_features_from_raw
from astra.inference.section_preparation import prepare as prepare_section
from astra.inference.section import predict_section
from astra.inference.section_geometry import tile_plan, prediction_tiles
from astra.model.model import Direct8Model
from astra.model.fp32 import enable_fp32
from astra.data.section_inputs import prepare_visium
from astra.inference.benchmark import GeneMetrics, evaluate, query_lookup
from astra.inference.st100 import prepare as prepare_st100, predict as predict_st100, fill_gaps
from astra.training.user_data import configure


def native_fixture(path, genes, shape=(768,768)):
    with h5py.File(path,'w') as f:
        f.attrs['metadata_json']=json.dumps(dict(spot_pitch=2,nrows=shape[0],ncols=shape[1],
            transform_matrices=dict(spot_colrow_to_microscope_colrow=[[1,0,0],[0,1,0],[0,0,1]])))
        f['features/id']=np.array(genes,dtype='S');f['features/name']=np.array(genes,dtype='S')
        f['features/feature_type']=np.array(['Gene Expression']*len(genes),dtype='S')
        f['features/target_sets/probes']=np.arange(len(genes),dtype='i4')
        yx=np.array([[0,0],[1,1],[4,4],[7,7],[8,8],[63,64],[64,64],[100,100],[128,128]],'i4')
        yx=yx[(yx<np.asarray(shape)).all(1)]
        for i in range(len(genes)):
            f[f'feature_slices/{i}/row']=yx[:,0];f[f'feature_slices/{i}/col']=yx[:,1]
            f[f'feature_slices/{i}/data']=np.arange(1,len(yx)+1,dtype='i4')*(i+1)
        yy,xx=np.indices((shape[0]//8,shape[1]//8))
        f['masks/square_016um/row']=yy.ravel();f['masks/square_016um/col']=xx.ravel()
        f['masks/square_016um/data']=np.ones(yy.size,'u1')
    return yx


def image_cache(root, starts, metadata, transform, capture_shape=None):
    root.mkdir()
    n=len(starts)
    arrays=dict(starts_yx=starts,density_float32=np.ones((n,128,128,3),'f4'),
        area_uint16=np.full((n,128,128),65535,'u2'),field_valid_bool=np.ones((n,128,128),bool),
        features_float32=np.zeros((n,1024,14,14),'f4'),pathology_valid_bool=np.ones((n,14,14),bool))
    yy,xx=np.indices((128,128))
    arrays['density_float32'][...,0]=1+yy/128+xx/256
    if capture_shape is not None:
        arrays['field_valid_bool']=np.stack([((yy+y>=0)&(yy+y<capture_shape[0])&(xx+x>=0)&(xx+x<capture_shape[1])) for y,x in starts])
    for name,value in arrays.items():np.save(root/(name+'.npy'),value)
    write_json(root/'images.json',dict(preprocessing_mode='raw',uni_checkpoint_sha256=metadata['uni_checkpoint_sha256'],
        spot_to_image=transform,density_method='native_RGB_OD_HE_bilinear_midpoint_8x8',
        fixture='synthetic density and zero features; NOT native UNI measurements'))


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_hd_coarse_reference_separation_and_panel_availability(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);package,genes,metadata=synthetic_package(p/'package')
            yx=native_fixture(p/'native.h5',genes[:1],shape=(128,128))
            Image.fromarray(np.full((130,130,3),180,'u1')).save(p/'image.png')
            c=dict(task='HD16',sample_id='fixture',protocol_id=1,feature_slice='native.h5',
                   tissue_image='image.png',uni_checkpoint='not_bundled.bin')
            write_json(p/'config.json',c);prepare_benchmark(p/'config.json',p/'bench',package)
            obs=p/'bench/observations';ref=p/'bench/reference'
            self.assertEqual(json.loads((obs/'gene_ids.json').read_text()),genes[:1])
            np.testing.assert_array_equal(np.load(ref/'gene_available.npy'),[True,False])
            truth=np.load(ref/'counts_8um.npy');queries=np.load(ref/'query_yx_8um.npy')
            lookup={tuple(v):i for i,v in enumerate(queries)}
            expected=np.zeros(len(queries),'i8')
            for point,value in zip(yx,np.arange(1,len(yx)+1)):expected[lookup[tuple(point//4)]]+=value
            np.testing.assert_array_equal(truth[:,0],expected)
            np.testing.assert_array_equal(truth[:,1],0)
            parents=np.load(obs/'parent_yx_16um.npy');coarse=sparse.load_npz(obs/'counts.npz').toarray()
            for i,parent in enumerate(parents):
                children=(parent[None]*2+np.array([(0,0),(0,1),(1,0),(1,1)]))
                self.assertEqual(coarse[i,0],sum(expected[lookup[tuple(q)]] for q in children))
            self.assertNotIn('reference',json.loads((obs/'section.json').read_text()))

    def test_pseudo_visium_circle_counts_are_native_center_sums(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);package,genes,_=synthetic_package(p/'package')
            yx=native_fixture(p/'native.h5',genes,shape=(256,256))
            Image.fromarray(np.full((260,260,3),180,'u1')).save(p/'image.png')
            c=dict(task='Spot55',sample_id='fixture',protocol_id=1,feature_slice='native.h5',
                   tissue_image='image.png',uni_checkpoint='not_bundled.bin')
            write_json(p/'config.json',c);prepare_benchmark(p/'config.json',p/'bench',package)
            centers=np.load(p/'bench/observations/spot_yx_um.npy')
            counts=sparse.load_npz(p/'bench/observations/counts.npz').toarray()
            for i,center in enumerate(centers):
                inside=((yx*2+1-center)**2).sum(1)<=27.5**2
                for g in range(2):self.assertEqual(counts[i,g],(np.arange(1,len(yx)+1)[inside]*(g+1)).sum())
            geometry=json.loads((p/'bench/observations/section.json').read_text())
            self.assertEqual((geometry['core_um'],geometry['stride_um']),(160,144))
            np.testing.assert_array_equal(np.load(p/'bench/reference/query_yx_8um.npy'),interior_queries(centers)[0])

    def test_native_benchmark_through_preparation_prediction_and_disk_scoring(self):
        for task in ('HD16','Spot55'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as td:
                p=Path(td);package,genes,metadata=synthetic_package(p/'package')
                native_fixture(p/'native.h5',genes,shape=(256,256))
                Image.fromarray(np.full((260,260,3),180,'u1')).save(p/'image.png')
                write_json(p/'c.json',dict(task=task,sample_id='fixture',protocol_id=1,
                    feature_slice='native.h5',tissue_image='image.png'))
                prepare_benchmark(p/'c.json',p/'bench',package)
                config_path=p/'bench/observations/section.json';config=json.loads(config_path.read_text())
                if task=='HD16':
                    starts=tile_plan(np.load(config_path.parent/'parent_yx_16um.npy'),[256,256],core_um=160)[0]
                else:
                    starts=prediction_tiles(np.load(config_path.parent/'query_yx_8um.npy'),[256,256],
                        dict(core_um=160,stride_um=144,context_margin_um=48))[0]
                image_cache(p/'images',starts,metadata,np.eye(3).tolist(),capture_shape=[256,256])
                config['image_cache']=str(p/'images');write_json(config_path,config)
                prepare_section(config_path,p/'inputs',package)
                report=predict_section(p/'inputs',p/'prediction',package,batch_size=4)
                metrics=evaluate(p/'prediction',p/'bench/reference',p/'metrics.json')
                self.assertTrue(metrics['prediction_read_from_disk'])
                self.assertFalse(metrics['reference_used_for_model_fitting']);self.assertFalse(metrics['fine_tuned'])
                self.assertEqual(report['checkpoint_sha256'],metadata['checkpoint_sha256'])
                self.assertEqual(metrics['gene_wise']['measured_genes'],2)

    def test_metrics_preserve_undefined_values_and_published_minmax_definition(self):
        p=np.array([[1.,4.,0.],[3.,2.,0.],[2.,1.,0.],[8.,7.,0.]])
        y=np.array([[2.,5.,0.],[4.,3.,0.],[1.,2.,0.],[9.,8.,0.]])
        metrics=GeneMetrics(3);metrics.add(p[:2],y[:2]);metrics.add(p[2:],y[2:])
        summary,rows=metrics.result(['A','B','zero'])
        for g in range(2):
            self.assertAlmostEqual(rows[g]['pcc'],np.corrcoef(p[:,g],y[:,g])[0,1],places=12)
            pn=(p[:,g]-p[:,g].min())/np.ptp(p[:,g]);yn=(y[:,g]-y[:,g].min())/np.ptp(y[:,g])
            self.assertAlmostEqual(rows[g]['nrmse'],np.sqrt(np.mean((pn-yn)**2)),places=12)
        self.assertIsNone(rows[2]['pcc']);self.assertEqual(summary['defined_pcc_genes'],2)

    def test_scoring_reads_files_and_rejects_mismatched_support_and_panel(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);pred=p/'prediction';ref=p/'reference';pred.mkdir();ref.mkdir()
            genes=['A','B'];queries=np.array([[0,0],[0,1],[1,0],[1,1]],'i8')
            truth=np.array([[0,2],[1,3],[2,1],[4,0]],'i8')
            for root in (pred,ref):write_json(root/'gene_ids.json',genes);np.save(root/'gene_available.npy',np.ones(2,bool))
            write_json(pred/'report.json',dict(sample_id='fixture',task='Spot55',checkpoint_sha256='fixture',fine_tuning=None))
            write_json(ref/'reference.json',dict(sample_id='fixture',task='Spot55',role='evaluation_only_measured_8um',reference_used_for_model_fitting=False))
            np.save(ref/'query_yx_8um.npy',queries);np.save(ref/'counts_8um.npy',truth)
            np.save(pred/'query_yx_8um.npy',queries[::-1]);np.save(pred/'prediction.npy',truth[::-1].astype('f4'))
            r=evaluate(pred,ref,p/'metrics.json');self.assertAlmostEqual(r['gene_wise']['pcc'],1.)
            self.assertAlmostEqual(r['position_wise']['bray_curtis'],0.)
            np.save(pred/'query_yx_8um.npy',queries[:3])
            with self.assertRaises(ValueError):evaluate(pred,ref,p/'bad.json')
            np.save(pred/'query_yx_8um.npy',queries[::-1]);write_json(pred/'gene_ids.json',genes[::-1])
            with self.assertRaisesRegex(ValueError,'gene order'):evaluate(pred,ref,p/'bad.json')

    def test_visium_barcode_alignment_and_explicit_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);matrix=sparse.csc_matrix(np.array([[2,3],[5,7]],'i8'))
            with h5py.File(p/'matrix.h5','w') as f:
                for name,value in [('data',matrix.data),('indices',matrix.indices),('indptr',matrix.indptr),('shape',matrix.shape)]:f['matrix/'+name]=value
                f['matrix/barcodes']=np.array(['b','a'],dtype='S');f['matrix/features/id']=np.array(['B','A'],dtype='S')
            with (p/'positions.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['barcode','in_tissue','pxl_col_in_fullres','pxl_row_in_fullres']);writer.writeheader()
                writer.writerows([dict(barcode='a',in_tissue=1,pxl_col_in_fullres=130,pxl_row_in_fullres=130),dict(barcode='b',in_tissue=1,pxl_col_in_fullres=80,pxl_row_in_fullres=80)])
            c=dict(sample_id='visium',protocol_id=0,diameter_um=55,pitch_um=100,matrix_h5='matrix.h5',positions_csv='positions.csv',
                   capture_shape_yx_2um=[256,256],spot_to_image=np.eye(3).tolist(),tissue_image='image.tif',uni_checkpoint='not_bundled.bin')
            write_json(p/'c.json',c);prepare_visium(p/'c.json',p/'out')
            self.assertEqual(json.loads((p/'out/barcodes.json').read_text()),['b','a'])
            np.testing.assert_array_equal(sparse.load_npz(p/'out/counts.npz').toarray(),matrix.T.toarray())
            np.testing.assert_array_equal(np.load(p/'out/spot_yx_um.npy'),[[161,161],[261,261]])
            c.pop('spot_to_image');write_json(p/'bad.json',c)
            with self.assertRaises(ValueError):prepare_visium(p/'bad.json',p/'bad')

    def test_st100_nine_positions_full_anchor_and_independent_core_coverage(self):
        centers=np.array([[256.,256.],[256.,406.]])
        g=st100_geometry(centers);self.assertEqual(len(g['starts']),18)
        expected={}
        for start,anchor in zip(g['starts'],g['anchors']):
            owner=st100_owner(start,centers[anchor]);self.assertEqual(int((owner==0).sum()),1976)
            indices=st100_field_indices(g,start)
            for q in g['query_yx'][indices]:expected[tuple(q)]=expected.get(tuple(q),0)+1
        np.testing.assert_array_equal(g['coverage'],[expected[tuple(q)] for q in g['query_yx']])
        np.testing.assert_array_equal(np.unique(g['starts'][:9]-g['starts'][4],axis=0),np.array([(y,x) for y in (-12,0,12) for x in (-12,0,12)]))

    def test_st100_end_to_end_frozen_cache_export_and_grid_residual(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);package,genes,metadata=synthetic_package(p/'package')
            centers=np.array([[256.,256.]]);g=st100_geometry(centers);transform=np.eye(3).tolist()
            sparse.save_npz(p/'counts.npz',sparse.csr_matrix([[19,37]],dtype='i8'))
            write_json(p/'genes.json',genes);np.save(p/'centers.npy',centers)
            image_cache(p/'images',g['starts'],metadata,transform)
            c=dict(task='ST100',sample_id='synthetic_ST100',protocol_id=0,diameter_um=100,pitch_um=150,
                counts='counts.npz',gene_ids='genes.json',spot_yx_um='centers.npy',image_cache='images',
                capture_shape_yx_2um=[256,256],spot_to_image=transform)
            write_json(p/'c.json',c);prepare_st100(p/'c.json',p/'inputs',package)
            r=predict_st100(p/'inputs',p/'out',package,batch_size=4)
            self.assertEqual(r['arithmetic_dtype'],'float32');self.assertEqual(r['fields'],9)
            self.assertLessEqual(r['parent_conservation_max_scaled'],5e-6)
            self.assertFalse(r['serialized_export']['exact_representation'])
            values=np.load(p/'out/prediction.npy');self.assertEqual(values.shape,(676,2))
            self.assertTrue(np.isfinite(values).all());self.assertEqual(values.dtype,np.float32)
            np.testing.assert_array_equal(np.load(p/'out/query_coverage.npy'),g['coverage'])
            # Independently accumulate all complete FOV cores in FP64, then compare the saved FP32 mean.
            model=enable_fp32(Direct8Model(**json.loads((package/'model/config.json').read_text())['kwargs']).eval())
            model.load_state_dict(torch.load(package/'model/checkpoint.pt',weights_only=True),strict=True)
            sums=np.zeros_like(values,dtype='f8');seen=np.zeros(len(values),'i4')
            with torch.no_grad():
                for f,start in enumerate(g['starts']):
                    density=torch.from_numpy(np.load(p/'inputs/density_float32.npy')[f:f+1])
                    area=torch.from_numpy(np.load(p/'inputs/area_uint16.npy')[f:f+1]);valid=torch.ones((1,128,128),dtype=torch.bool)
                    result=model(parent_counts=torch.tensor([[[19.,37.]]]),owner_map=torch.from_numpy(st100_owner(start,centers[0])[None]).long(),
                        parent_valid=torch.ones((1,1),dtype=torch.bool),field_valid=valid,
                        image_features_2um=image_features_from_raw(density,area,valid).permute(0,3,1,2).contiguous(),
                        gene_available=torch.ones((1,2),dtype=torch.bool),protocol_id=torch.zeros(1,dtype=torch.long),
                        pathology_features=torch.zeros((1,1024,14,14)),pathology_valid=torch.ones((1,14,14),dtype=torch.bool))
                    core=result['pred_count_8um'][0,6:26,6:26].numpy()
                    lookup={tuple(q):i for i,q in enumerate(g['query_yx'])}
                    for y in range(20):
                        for x in range(20):
                            index=lookup[tuple(start//4+[y+6,x+6])];sums[index]+=core[y,x];seen[index]+=1
            np.testing.assert_allclose(values,sums/seen[:,None],rtol=3e-6,atol=1e-5)

    def test_st100_gap_sources_empty_support_and_tissue_barrier(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);np.save(p/'query_yx_8um.npy',np.array([[2,2]],'i4'));np.save(p/'prediction.npy',np.array([[3,7]],'f4'))
            mask=np.zeros((12,12),bool);mask[:8,:8]=True;mask[:,5]=False
            result=fill_gaps(p,mask);source=np.load(p/'prediction_source.npy')
            self.assertGreater(result['interpolated_bins'],0);self.assertEqual(source[2,2],1)
            self.assertTrue((source[:,6:]==0).all())
            np.testing.assert_array_equal(np.load(p/'prediction_interpolated.npy'),np.tile([3,7],(result['interpolated_bins'],1)))
            fill_gaps(p,np.zeros_like(mask));self.assertEqual(np.load(p/'prediction_interpolated.npy').shape,(0,2))

    def test_user_training_workspace_spatial_holdout_and_original_panel_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);native_fixture(p/'native.h5',json.loads((ROOT/'model/input_gene_ids.json').read_text())[:1])
            Image.fromarray(np.full((768,768,3),180,'u1')).save(p/'image.png')
            m=dict(uni_checkpoint='missing_UNI.bin',formal_training=False,max_epochs=3,sections=[dict(sample_id='user_section',
                role='train',protocol_id=0,feature_slice='native.h5',tissue_image='image.png',
                blocks_yx_2um=[[0,0],[0,256],[256,0]],monitor_blocks_yx_2um=[[0,0],[0,256]])])
            write_json(p/'manifest.json',m);configure(p/'manifest.json',p/'task')
            split=json.loads((p/'task/training/spatial_split.json').read_text())['sections']['user_section']
            self.assertEqual(len(split['monitor_items']),8);self.assertEqual(split['retained_fields'],4)
            self.assertEqual((p/'task/training/panel.json').read_bytes(),(ROOT/'training/panel.json').read_bytes())
            result=subprocess.run([sys.executable,'-B','-m','astra','train','--workspace',str(p/'task'),'--check-config'],
                                  text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['input_genes'],2000)
            result=subprocess.run([sys.executable,'-B','-m','astra','prepare-training','--workspace',str(p/'task'),
                '--smoke','--stage','geometry'],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            smoke=json.loads((p/'task/training/smoke/config.json').read_text())
            self.assertEqual(smoke['training']['fields_per_epoch'],3)
            result=subprocess.run([sys.executable,'-B','-m','astra','train','--workspace',str(p/'task'),
                '--config',str(p/'task/training/smoke/config.json'),'--check-config'],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            m['sections'][0]['sample_id']='WT_Ovarian_FF_6p5mm';write_json(p/'bad.json',m)
            with self.assertRaises(PermissionError):configure(p/'bad.json',p/'bad_task')


@unittest.skipUnless(os.environ.get('ASTRA_RUN_BUNDLED')=='1','enable actual pretrained checkpoint compatibility with ASTRA_RUN_BUNDLED=1')
class BundledWorkflows(unittest.TestCase):
    def test_st100_bundled_2000_gene_checkpoint_synthetic_inputs(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);metadata=json.loads((ROOT/'model/metadata.json').read_text())
            genes=json.loads((ROOT/'model/input_gene_ids.json').read_text());centers=np.array([[256.,256.]])
            g=st100_geometry(centers);transform=np.eye(3).tolist()
            sparse.save_npz(p/'counts.npz',sparse.csr_matrix([[19,37]],dtype='i8'))
            write_json(p/'genes.json',genes[:2]);np.save(p/'centers.npy',centers)
            image_cache(p/'images',g['starts'],metadata,transform)
            write_json(p/'config.json',dict(task='ST100',sample_id='synthetic_ST100_checkpoint_contract',
                protocol_id=0,diameter_um=100,pitch_um=150,counts='counts.npz',gene_ids='genes.json',
                spot_yx_um='centers.npy',image_cache='images',capture_shape_yx_2um=[256,256],spot_to_image=transform))
            prepare_st100(p/'config.json',p/'inputs',ROOT)
            report=predict_st100(p/'inputs',p/'prediction',ROOT,batch_size=1)
            self.assertEqual(report['checkpoint_sha256'],metadata['checkpoint_sha256'])
            self.assertEqual(report['output_shape'],[676,2000])
            self.assertEqual(int(np.load(p/'prediction/gene_available.npy').sum()),2)
            self.assertLessEqual(report['parent_conservation_max_scaled'],5e-6)
            self.assertTrue(np.isfinite(np.load(p/'prediction/prediction.npy')).all())


if __name__=='__main__':unittest.main()
