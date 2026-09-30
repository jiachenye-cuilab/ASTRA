import numpy as np
import torch
from astra.fine_tuning.geometry import core_supervision
from astra.fine_tuning.hd_objective import objective as hd_objective, preservation_objective as hd_preservation, target_from_packet
from astra.fine_tuning.spot_objective import objective as spot_objective, preservation_objective as spot_preservation
from astra.fine_tuning.sampling import pair_choices
from astra.fine_tuning.inputs import packets as target_packets, packets as spot_packets

def data_objective(model, fields, selected, packet, available, protocol, semantic, spec, source, epoch, selection_view=None):
    if spec['settings']['task']=='HD16':
        target=target_from_packet(packet,available,protocol,semantic)
        mask=core_supervision(batch_size=len(selected),device=fields.device,core_um=source['supervision_core_um'])
        return hd_objective(model,target,supervision=mask)
    pairs=pair_choices(fields,selected,spec,epoch,selection_view=selection_view)
    return spot_objective(model,packet,available,protocol,semantic,pairs)


def protection(model, teacher, fields, packet, available, protocol, semantic, spec, source):
    if spec['settings']['task']=='HD16':
        target=target_from_packet(packet,available,protocol,semantic)
        mask=core_supervision(batch_size=len(packet.parent_counts),device=fields.device,core_um=source['supervision_core_um'])
        return hd_preservation(model,teacher,target,supervision=mask)
    return spot_preservation(model,teacher,packet,available,protocol,semantic,source['supervision_core_um'])


def selection_score(model, fields, spec, source, *, smoke):
    task=spec['settings']['task'];packets=target_packets if task=='HD16' else spot_packets
    items=spec['groups']['selection'][:1] if smoke else spec['groups']['selection']
    views=1 if task=='HD16' else spec['spot_selection_pair_views']
    model.eval();values=[]
    with torch.no_grad(), packets(fields,None,source,items,batch_size=4,role='target_input',task=task) as batches:
        for selected,packet,available,protocol,semantic in batches:
            for view in range(views):
                prediction,loss=data_objective(model,fields,selected,packet,available,protocol,semantic,spec,source,0,selection_view=view)
                values.extend([float(loss)]*len(selected));del prediction,loss
    return float(np.mean(values))
