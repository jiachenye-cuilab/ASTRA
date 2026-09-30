"""Build an explicit user Visium HD training task, preserving the existing optimizer/loss."""
from pathlib import Path
from copy import deepcopy
import os
import shutil
import re
import numpy as np
from astra.assets import ROOT as PACKAGE
from astra.inference.section import read, write
from astra.data.native_benchmark import feature_metadata


def configure(manifest_path,output,package=PACKAGE):
    path,output,package=Path(manifest_path).resolve(),Path(output).resolve(),Path(package)
    if output.exists(): raise FileExistsError('choose a new user training workspace')
    manifest=read(path);sections=manifest['sections']
    names=[s['sample_id'] for s in sections]
    if not names or len(set(names))!=len(names) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+',n) or n in ('.','..') for n in names):
        raise ValueError('unique safe sample IDs are required')
    original=read(package/'training/config.json')
    fixed_tests=set(original['wt'].get('test',[])+original['three_prime'].get('test',[]))
    if fixed_tests.intersection(names): raise PermissionError('frozen paper benchmark tests cannot enter user fitting')
    if not isinstance(manifest.get('formal_training',False),bool): raise ValueError('formal_training must be boolean')
    records={};geometry={};spatial={};wt=[];three_prime=[]
    for s in sections:
        if s.get('role')!='train' or s.get('protocol_id') not in (0,1) or isinstance(s['protocol_id'],bool):
            raise ValueError('declare each fitting section role=train and protocol_id=0 or 1')
        native=(path.parent/s['feature_slice']).resolve();image=(path.parent/s['tissue_image']).resolve()
        if not image.is_file(): raise FileNotFoundError(image)
        _,shape=feature_metadata(native)
        blocks=np.asarray(s['blocks_yx_2um']);monitor=np.asarray(s['monitor_blocks_yx_2um'])
        if (blocks.ndim!=2 or blocks.shape[1]!=2 or blocks.dtype.kind not in 'iu' or np.any(blocks<0)
                or np.any(blocks%256) or np.any(blocks+256>shape) or len(np.unique(blocks,axis=0))!=len(blocks)
                or monitor.shape!=(2,2) or monitor.dtype.kind not in 'iu' or len(np.unique(monitor,axis=0))!=2
                or not set(map(tuple,monitor))<=set(map(tuple,blocks)) or len(blocks)<=2 or len(blocks)>65535):
            raise ValueError('provide unique aligned complete 512um blocks, including two distinct spatial monitor blocks and training blocks')
        # Four nonoverlapping 256um fields in each 512um block. No expression-based selection.
        offsets=np.array([(0,0),(0,128),(128,0),(128,128)],'i4')
        starts=(blocks[:,None]+offsets).reshape(-1,2).astype('i4')
        excluded=np.zeros(len(starts),bool)
        for origin in monitor: excluded|=((starts<origin+256)&(starts+128>origin)).all(1)
        training=np.flatnonzero(~excluded).astype('i4');held=np.flatnonzero(excluded)
        rid=s['sample_id'];protocol=s['protocol_id']
        geometry[rid]=dict(block_starts_yx=blocks.astype('i4'),starts_yx=starts,
            train_block_slot=np.repeat(np.arange(len(blocks)),4).astype('u2'),
            train_offset_yx=np.tile(offsets,(len(blocks),1)).astype('u1'),training_indices=training)
        spatial[rid]=dict(protocol_id=protocol,source_fields=len(starts),retained_fields=len(training),excluded_fields=len(held),
            block_origins_yx_2um=monitor.tolist(),training_indices=f'training/geometry/{rid}_training_indices.npy',
            monitor_items=[dict(sample=rid,protocol_id=protocol,field_index=int(i),start_yx_2um=starts[i].tolist()) for i in held])
        records[rid]=dict(collection='visium_hd_wt' if protocol==1 else 'visium_hd_3prime',
                         feature_slice=os.path.relpath(native,output),tissue_image=os.path.relpath(image,output))
        (wt if protocol==1 else three_prime).append(rid)
    config=deepcopy(original)
    for key in ('dataset_split','split_id','fold','final_refit_preprocessing_reference'): config.pop(key,None)
    config.update(experiment_id='ASTRA/user-Visium-HD',panel_policy='reuse_published_panel',
                  wt=dict(train=wt,validation=[],test=[]),three_prime=dict(auxiliary_train=three_prime,validation=[],test=[]))
    settings=config['training'];settings.update(formal_training=manifest.get('formal_training',False),
        fields_per_section=manifest.get('fields_per_section',100),section_field_overrides={},cpu_threads=2)
    for rid in names:
        retained=spatial[rid]['retained_fields']
        if retained<settings['fields_per_section']:
            if 'fields_per_section' in manifest: raise ValueError('requested epoch quota exceeds distinct retained user FOVs')
            settings['section_field_overrides'][rid]=retained
    settings['fields_per_epoch']=sum(settings['section_field_overrides'].get(rid,settings['fields_per_section']) for rid in names)
    for key in ('seed','max_epochs','batch_size','learning_rate','gradient_accumulation_steps'):
        if key in manifest: settings[key]=manifest[key]
    settings['minimum_epochs']=min(settings['minimum_epochs'],settings['max_epochs'])
    config['final_refit'].update(scope='user_sections_from_scratch',epoch=settings['max_epochs'])
    config['selection']['aggregation']='equal_FOV_within_section_then_equal_user_sections'
    config['boundaries']=dict(initialization='scratch_no_pretrained_ASTRA_weights',test_expression_read=False,
        panel='reuse_published_panel_with_original_fit_provenance; not newly fitted on user data',
        training_coordinates='user_explicit_blocks_excluding_two_spatial_monitor_blocks_per_section',
        validation='eight_disjoint_monitor_FOVs_per_section',formal_training='explicit_user_manifest')
    uni=(path.parent/manifest['uni_checkpoint']).resolve()
    metadata=read(package/'model/metadata.json')
    registry=dict(datasets=records,model_weights=dict(UNI=dict(weight_file=os.path.relpath(uni,output),
                  config_file='model/uni_config.json',sha256=metadata['uni_checkpoint_sha256'])))
    output.mkdir(parents=True);(output/'training/geometry').mkdir(parents=True);(output/'model').mkdir()
    for name in ('metadata.json','uni_config.json'): shutil.copyfile(package/'model'/name,output/'model'/name)
    for name in ('panel.json',): shutil.copyfile(package/'training'/name,output/'training'/name)
    shutil.copyfile(package/'VERSION',output/'VERSION')
    for rid,arrays in geometry.items():
        np.savez(output/'training/geometry'/f'{rid}.npz',**arrays)
        np.save(output/'training/geometry'/f'{rid}_training_indices.npy',arrays['training_indices'])
    write(output/'training/spatial_split.json',dict(sections=spatial));write(output/'training/config.json',config)
    write(output/'resource.json',registry)
    report=dict(status='configured',sections=names,workspace=str(output),formal_training=settings['formal_training'],
        model_weights_copied=False,training_started=False,selection='original joint HD16/Spot55 spatial holdout',
        paper_replication=False,panel='published fixed panel, original fitting provenance retained')
    write(output/'configuration.json',report);return report
