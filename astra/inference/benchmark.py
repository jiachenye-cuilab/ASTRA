"""Evaluate serialized predictions against measured, separately stored 8um counts."""
from pathlib import Path
import numpy as np
from astra.inference.section import read, write


class GeneMetrics:
    """Published gene-wise PCC and independent min-max nRMSE, bounded by one block."""
    def __init__(self, genes):
        self.n=0; self.sums=np.zeros((6,genes),float)
        self.minimum=np.full((2,genes),np.inf); self.maximum=np.full((2,genes),-np.inf)

    def add(self, prediction, target):
        p,y=np.asarray(prediction,float),np.asarray(target,float)
        if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all() or np.any(p<0) or np.any(y<0):
            raise ValueError('score aligned finite nonnegative prediction/reference counts')
        if not len(p): return
        self.n+=len(p)
        self.sums+=np.array([p.sum(0),y.sum(0),(p*p).sum(0),(y*y).sum(0),(p*y).sum(0),abs(p-y).sum(0)])
        self.minimum=np.minimum(self.minimum,[p.min(0),y.min(0)])
        self.maximum=np.maximum(self.maximum,[p.max(0),y.max(0)])

    def result(self, genes):
        if not self.n: raise ValueError('empty evaluation support')
        mp,my,p2,y2,py,mae=self.sums/self.n
        vp,vy=np.maximum(p2-mp*mp,0),np.maximum(y2-my*my,0)
        cov=py-mp*my;span=self.maximum-self.minimum
        valid=(span>0).all(0)&(vp>0)&(vy>0)
        pcc=np.divide(cov,np.sqrt(vp*vy),out=np.zeros_like(cov),where=valid)
        a,b=np.divide(1.,span,out=np.zeros_like(span),where=span>0)
        difference=(mp-self.minimum[0])*a-(my-self.minimum[1])*b
        nrmse=np.sqrt(np.maximum(vp*a*a+vy*b*b-2*cov*a*b+difference*difference,0))
        rows=[dict(gene=g,pcc=float(pcc[j]) if valid[j] else None,nrmse=float(nrmse[j]),mae=float(mae[j])) for j,g in enumerate(genes)]
        return dict(positions=self.n,measured_genes=len(genes),defined_pcc_genes=int(valid.sum()),
                    pcc=float(pcc[valid].mean()) if valid.any() else None,nrmse=float(nrmse.mean())),rows


def query_lookup(queries, requested):
    q,r=np.asarray(queries),np.asarray(requested)
    if (q.ndim!=2 or q.shape[1]!=2 or r.ndim!=2 or r.shape[1]!=2 or q.dtype.kind not in 'iu'
            or r.dtype.kind not in 'iu' or np.any(q<0) or np.any(r<0) or len(np.unique(q,axis=0))!=len(q)):
        raise ValueError('query coordinates must be unique [N,2] y,x indices')
    dtype=np.dtype([('y','i8'),('x','i8')])
    left=np.ascontiguousarray(q,dtype='i8').view(dtype).ravel()
    right=np.ascontiguousarray(r,dtype='i8').view(dtype).ravel()
    order=np.argsort(left);slots=np.searchsorted(left[order],right)
    if np.any(slots>=len(left)) or not np.array_equal(left[order[slots]],right):
        raise ValueError('prediction does not cover every specified evaluation query; no zero filling')
    return order[slots]


def evaluate(prediction, reference, output, *, position_mask=None, block_size=512):
    prediction,reference,output=Path(prediction),Path(reference),Path(output)
    if output.exists(): raise FileExistsError('choose a new metrics file')
    if block_size<1: raise ValueError('block_size must be positive')
    info=read(reference/'reference.json'); report=read(prediction/'report.json')
    if info.get('role')!='evaluation_only_measured_8um' or info.get('reference_used_for_model_fitting') is not False:
        raise ValueError('benchmark needs measured reference with an evaluation-only role')
    if info['sample_id']!=report['sample_id'] or info['task']!=report['task']:
        raise ValueError('reference sample/task differs from prediction')
    genes=read(reference/'gene_ids.json')
    if genes!=read(prediction/'gene_ids.json'): raise ValueError('prediction/reference gene order differs')
    available=np.load(reference/'gene_available.npy',allow_pickle=False)
    if (available.shape!=(len(genes),) or available.dtype!=np.bool_
            or not np.array_equal(available,np.load(prediction/'gene_available.npy',allow_pickle=False)) or not available.any()):
        raise ValueError('score the same nonempty measured gene panel; missing genes are not truth')
    values=np.load(prediction/'prediction.npy',mmap_mode='r',allow_pickle=False)
    if report['task']=='HD16':
        parents=np.load(prediction/'parent_yx_16um.npy',allow_pickle=False)
        queries=(parents[:,None]*2+np.array([(0,0),(0,1),(1,0),(1,1)])).reshape(-1,2)
        values=values.reshape(-1,len(genes))
    else: queries=np.load(prediction/'query_yx_8um.npy',allow_pickle=False)
    requested=np.load(reference/'query_yx_8um.npy',allow_pickle=False)
    if (values.shape!=(len(queries),len(genes)) or len(np.unique(requested,axis=0))!=len(requested)):
        raise ValueError('prediction shape or evaluation query identities differ')
    lookup=query_lookup(queries,requested)
    target=np.load(reference/'counts_8um.npy',mmap_mode='r',allow_pickle=False)
    if target.shape!=(len(requested),len(genes)) or target.dtype.kind not in 'iu':
        raise ValueError('reference must contain aligned raw integer measured counts')
    masks={}
    if position_mask:
        masks['common']=np.load(position_mask,allow_pickle=False)
        if masks['common'].shape!=(len(requested),) or masks['common'].dtype!=np.bool_:
            raise ValueError('common position mask must follow reference coordinates')
    metrics=GeneMetrics(int(available.sum()));pcc_sum=bc_sum=0.;pcc_n=bc_n=0
    for start in range(0,len(requested),block_size):
        stop=min(start+block_size,len(requested))
        p=np.asarray(values[lookup[start:stop]][:,available],float)
        y=np.asarray(target[start:stop][:,available],float);metrics.add(p,y)
        pc,yc=p-p.mean(1,keepdims=True),y-y.mean(1,keepdims=True)
        denominator=np.sqrt((pc*pc).sum(1)*(yc*yc).sum(1))
        support=y.sum(1)>0
        if 'common' in masks: support &= masks['common'][start:stop]
        good=support&(denominator>0)
        pcc_sum+=float(((pc[good]*yc[good]).sum(1)/denominator[good]).sum());pcc_n+=int(good.sum())
        pt=p.sum(1,keepdims=True);yt=y.sum(1,keepdims=True)
        pn=np.divide(p,pt,out=np.zeros_like(p),where=pt>0)
        yn=np.divide(y,yt,out=np.zeros_like(y),where=yt>0)
        d=abs(pn-yn).sum(1)/2;d[pt[:,0]==0]=1
        bc_sum+=float(d[support].sum());bc_n+=int(support.sum())
    summary,rows=metrics.result(np.asarray(genes)[available].tolist())
    result=dict(status='complete',sample_id=info['sample_id'],task=info['task'],gene_wise=summary,per_gene=rows,
        position_wise=dict(pcc=pcc_sum/pcc_n if pcc_n else None,pcc_positions=pcc_n,
                           bray_curtis=bc_sum/bc_n if bc_n else None,bray_curtis_positions=bc_n,
                           support='explicit common mask' if position_mask else 'this ASTRA run; not cross-method common support'),
        prediction_read_from_disk=True,reference_used_for_model_fitting=False,
        checkpoint_sha256=report['checkpoint_sha256'],fine_tuned=bool(report.get('fine_tuning')),
        scope='ASTRA evaluation; external methods and paper-wide shared masks are supplied separately')
    output.parent.mkdir(parents=True,exist_ok=True);write(output,result)
    return {k:v for k,v in result.items() if k!='per_gene'}
