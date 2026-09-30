"""Registered full-model target adaptation on portable, coarse-only FOV inputs."""
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from astra.model.model import Direct8Model
from astra.fine_tuning.checkpoint import load_fine_tuned, save_fine_tuned
from astra.fine_tuning.inputs import Fields, packets
from astra.fine_tuning.fp32 import data_objective, protection, selection_score, enable_fp32_training
from astra.fine_tuning.sampling import cpu_state, epoch_items


def fit(manifest_path, output, package, *, device='cpu', smoke=False):
    from astra.runtime import sha256
    started = time.monotonic()
    package, output = Path(package), Path(output)
    metadata = json.loads((package/'model/metadata.json').read_text(encoding='utf-8'))
    for name in ('checkpoint.pt','config.json','input_gene_ids.json','output_gene_ids.json'):
        if sha256(package/'model'/name) != metadata[Path(name).stem+'_sha256']:
            raise ValueError(f'base package identity mismatch: {name}')
    config = json.loads((package/'model/config.json').read_text(encoding='utf-8'))
    recipe = json.loads((package/'model/fine_tuning.json').read_text(encoding='utf-8'))
    training = recipe['training']
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if 'prepared_section' in manifest:
        from astra.inference.section import FineTuningFields
        fields = FineTuningFields(manifest_path,config['kwargs']['input_gene_ids'],metadata,device,smoke=smoke)
    else:
        fields = Fields(manifest_path,config['kwargs']['input_gene_ids'],metadata,device,smoke=smoke)
    source = dict(supervision_core_um=recipe['supervision_core_um'], training=dict(batch_pipeline='device'))
    spec = dict(name=fields.sample_id,case=fields.sample_id,
        settings=dict(task=fields.task,fine_preservation_weight=training['fine_preservation_weight']),
        groups=dict(support=fields.groups['support'][:training['fields_per_epoch']],selection=fields.groups['selection']),
        adaptation_pool=fields.groups['support'],
        spot_selection_pair_views=recipe['tasks']['Spot55']['spot_selection_pair_views'],**training)
    torch.manual_seed(spec['seed'])
    if fields.device.type == 'cuda':
        torch.cuda.manual_seed_all(spec['seed'])
    state = torch.load(package/'model/checkpoint.pt',map_location='cpu',weights_only=True)
    model = Direct8Model(**config['kwargs']).to(fields.device)
    model.load_state_dict(state,strict=True)
    model = enable_fp32_training(model)
    teacher = Direct8Model(**config['kwargs']).to(fields.device).eval().requires_grad_(False)
    teacher.load_state_dict(state,strict=True)
    teacher = enable_fp32_training(teacher)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameters,lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    initial = best = selection_score(model,fields,spec,source,smoke=smoke)
    if not math.isfinite(initial):
        raise ValueError('initial coarse selection loss is not finite')
    output.mkdir(parents=True,exist_ok=False)
    best_epoch,best_state,updates,exposures = 0,cpu_state(model),0,0
    epochs = 1 if smoke else spec['epochs']
    accumulation = spec['gradient_accumulation_steps']
    changed = False
    for epoch in range(1,epochs+1):
        items = epoch_items(spec,epoch,smoke=smoke)
        losses,penalties = [],[]
        with packets(fields,None,source,items,batch_size=spec['batch_size'],role='target_input',task=fields.task) as batches:
            for i,(selected,packet,available,protocol,semantic) in enumerate(batches):
                first,last = i%accumulation==0,(i+1)%accumulation==0
                lr = spec['learning_rate']*min(1.,(updates+1)/spec['warmup_updates'])
                if first:
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                model.train()
                prediction,loss = data_objective(model,fields,selected,packet,available,protocol,semantic,spec,source,epoch)
                if not torch.isfinite(loss) or not loss.requires_grad:
                    raise ValueError('coarse loss must be finite and connected to student weights')
                (loss/accumulation).backward()
                losses.extend([float(loss.detach())]*len(selected))
                del prediction,loss
                penalty = protection(model,teacher,fields,packet,available,protocol,semantic,spec,source)
                if not torch.isfinite(penalty) or not penalty.requires_grad:
                    raise ValueError('preservation loss must be finite and connected to student weights')
                (spec['fine_preservation_weight']*penalty/accumulation).backward()
                penalties.extend([float(penalty.detach())]*len(selected))
                del penalty
                if last:
                    torch.nn.utils.clip_grad_norm_(parameters,spec['max_gradient_norm'],error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                exposures += len(selected)
        row = dict(epoch=epoch,optimizer_updates=updates,field_exposures=exposures,
            training_nll_per_umi=float(np.mean(losses)),preservation_loss=float(np.mean(penalties)),learning_rate=lr)
        if epoch%spec['selection_interval']==0 or epoch==epochs:
            value = selection_score(model,fields,spec,source,smoke=smoke)
            if not math.isfinite(value):
                raise ValueError('coarse selection loss is not finite')
            row['selection_nll_per_umi'] = value
            if value < best:
                best,best_epoch,best_state = value,epoch,cpu_state(model)
        with (output/'epochs.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(dict(epoch=epoch,optimizer_updates=updates,best_epoch=best_epoch)),flush=True)
    expected = (1,16) if smoke else (spec['optimizer_updates'],spec['field_exposures'])
    if (updates,exposures) != expected:
        raise ValueError('optimizer/exposure budget differs from registered recipe')
    changed = any(not torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items() if isinstance(v,torch.Tensor))
    if not changed:
        raise ValueError('fine-tuning made no parameter update')
    if any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
        raise ValueError('source teacher must remain frozen')
    if not all(torch.equal(v.cpu(),state[k]) for k,v in teacher.state_dict().items() if isinstance(v,torch.Tensor)):
        raise ValueError('source teacher weights changed')
    model.load_state_dict(best_state,strict=True)
    model.eval()
    scope = dict(sample_id=fields.sample_id,task=fields.task,image_preprocessing='raw',arithmetic_dtype='float32',
        query_fine_labels_used_for_fitting=False,smoke=smoke,best_epoch=best_epoch,
        total_epochs_run=epochs,optimizer_updates=updates,field_exposures=exposures,
        fine_preservation_weight=spec['fine_preservation_weight'],recipe=recipe)
    if hasattr(fields,'geometry'):
        scope.update(geometry=fields.geometry,boundary_x_2um=fields.boundary_x_2um,shared_train_inference_cache=True)
    checkpoint = output/'checkpoint.pt'
    save_fine_tuned(checkpoint,model,metadata,scope)
    del model,teacher,optimizer,parameters,best_state,state
    if fields.device.type == 'cuda':
        torch.cuda.empty_cache()
    restored,_ = load_fine_tuned(checkpoint,config['kwargs'],metadata,
        sample_id=fields.sample_id,task=fields.task,device=fields.device)
    replay = selection_score(restored,fields,spec,source,smoke=smoke)
    if abs(replay-best) > 1e-6:
        raise ValueError('selected fine-tuned checkpoint failed coarse-selection replay')
    report = dict(status='complete',model_name='ASTRA fine-tuned',sample_id=fields.sample_id,task=fields.task,arithmetic_dtype='float32',
        smoke=smoke,best_epoch=best_epoch,total_epochs_run=epochs,optimizer_updates=updates,field_exposures=exposures,
        initial_selection_nll=initial,selected_selection_nll=replay,checkpoint_reload_max_score_difference=abs(replay-best),
        base_checkpoint_sha256=metadata['checkpoint_sha256'],checkpoint_sha256=sha256(checkpoint),
        update_changed_weights=changed,teacher_weights_unchanged=True,query_fine_labels_used_for_fitting=False,
        elapsed_seconds=time.monotonic()-started,device=str(fields.device))
    if fields.device.type == 'cuda':
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated(fields.device)/2**20
    if hasattr(fields,'geometry'):
        report.update(geometry=fields.geometry,boundary_x_2um=fields.boundary_x_2um,shared_train_inference_cache=True)
    with (output/'report.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return report
