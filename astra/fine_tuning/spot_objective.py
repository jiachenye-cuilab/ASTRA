import torch
from astra.fine_tuning.geometry import core_supervision
from astra.fine_tuning.inputs import SpotPacket

def model_inputs(packet, available, protocol, semantic):
    if not isinstance(packet,SpotPacket):
        raise TypeError('spot adaptation requires a label-free SpotPacket')
    return dict(parent_counts=packet.parent_counts,owner_map=packet.owner_map,parent_valid=packet.parent_valid,
        image_features_2um=packet.image_features_2um,field_valid=packet.field_valid,
        gene_available=available,protocol_id=protocol,pathology_features=semantic[0],pathology_valid=semantic[1])


def merge_inputs(packet, available, protocol, semantic, pairs):
    inputs=model_inputs(packet,available,protocol,semantic)
    counts,owner=packet.parent_counts.detach(),packet.owner_map
    b,p,g=counts.shape
    pairs=torch.as_tensor(pairs,dtype=torch.long,device=owner.device)
    if pairs.shape!=(b,2) or bool((pairs[:,0]==pairs[:,1]).any()):
        raise ValueError('each FOV must merge two different observed parents')
    selected=torch.zeros_like(packet.parent_valid).scatter_(1,pairs,True)
    if not bool((~selected|packet.parent_valid).all()):
        raise ValueError('cannot merge a missing parent')
    mapping=torch.arange(p,device=owner.device)[None].expand(b,-1).clone()
    mapping.scatter_(1,pairs[:,1:2],pairs[:,0:1])
    merged=torch.zeros_like(counts).scatter_add_(1,mapping[...,None].expand(-1,-1,g),counts)
    merged_valid=packet.parent_valid.clone().scatter_(1,pairs[:,1:2],False)
    merged_owner=mapping.gather(1,owner.clamp_min(0).flatten(1)).reshape_as(owner)
    merged_owner=torch.where(owner>=0,merged_owner,-1)
    inputs.update(parent_counts=merged,owner_map=merged_owner,parent_valid=merged_valid)
    return inputs,selected


def original_spot_prediction(result, original_owner, parent_slots):
    """Integrate actual owner/8um intersection masses; never fill circles or gaps."""
    layout=result['segment_layout']
    inverse=layout.cell_to_segment.flatten()[layout.valid_cell_indices]
    original=original_owner.flatten()[layout.valid_cell_indices]
    low=torch.full((layout.segments,),parent_slots,dtype=torch.long,device=original.device)
    high=torch.full_like(low,-2)
    low.scatter_reduce_(0,inverse,original,reduce='amin',include_self=True)
    high.scatter_reduce_(0,inverse,original,reduce='amax',include_self=True)
    if not torch.equal(low,high):
        raise ValueError('an output segment crosses original spots; exact integration requires a refined layout')
    keep=high>=0
    mass=result['coarse_segment_mass']
    prediction=mass.new_zeros((len(original_owner)*parent_slots,mass.shape[-1]))
    prediction.index_add_(0,layout.segment_batch[keep]*parent_slots+high[keep],mass[keep])
    return prediction.reshape(len(original_owner),parent_slots,-1)


def objective(model, packet, available, protocol, semantic, pairs):
    inputs,selected=merge_inputs(packet,available,protocol,semantic,pairs)
    prediction=model(**inputs)
    restored=original_spot_prediction(prediction,packet.owner_map,packet.parent_counts.shape[1])
    truth=model.select_output_genes(packet.parent_counts.detach()).double()
    keep=selected[...,None]&model.select_output_genes(available)[:,None]
    y=torch.where(keep,truth,0)
    p=torch.where(keep,restored,0)
    terms=p-y*p.clamp_min(1e-12).log()
    loss=(terms.sum((1,2))/y.sum((1,2)).clamp_min(1)).mean()
    return prediction,loss


def preservation_objective(model, teacher, packet, available, protocol, semantic, core_um):
    inputs=model_inputs(packet,available,protocol,semantic)
    with torch.no_grad(): reference=teacher(**inputs)['pred_count_8um'].double()
    prediction=model(**inputs)['pred_count_8um'].double()
    mask=core_supervision(batch_size=len(prediction),device=prediction.device,core_um=core_um)&packet.field_valid
    active=mask.reshape(len(mask),32,4,32,4).all((2,4))
    keep=active[...,None]&model.select_output_genes(available)[:,None,None]
    q=torch.where(keep,reference,0);p=torch.where(keep,prediction,0)
    terms=q*(q.clamp_min(1e-12).log()-p.clamp_min(1e-12).log())+p-q
    return (terms.sum((1,2,3))/q.sum((1,2,3)).clamp_min(1)).mean()
