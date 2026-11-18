import numpy as np
def evaluate(ids,nodes,labels,p,choice):
    row=np.arange(len(choice))
    pred=ids[row,choice]
    classes=np.union1d(pred,labels)
    yi=np.searchsorted(classes,labels); pi=np.searchsorted(classes,pred)
    ref=np.bincount(yi,minlength=len(classes)); assigned=np.bincount(pi,minlength=len(classes))
    hits=np.bincount(yi[yi==pi],minlength=len(classes))
    precision=hits/np.maximum(assigned,1); recall=hits/np.maximum(ref,1)
    road=nodes[row,choice]
    continuous=(pred[:-1]==pred[1:])|(road[:-1,1]==road[1:,0])
    positive=ids==labels[:,None]
    return dict(top1=float((pred==labels).mean()),macro_f1=float((2*hits/(ref+assigned)).mean()),
        precision=float(precision.mean()),recall=float(recall.mean()),
        log_loss=float(-np.where(positive,np.log(p),np.log1p(-p)).mean()),
        continuity=float(continuous.mean()) if len(continuous) else 1.,
        miss_rate=float((~positive.any(1)).mean()))

def sequence_similarity(pred,route):
    a=pred[np.r_[True,pred[1:]!=pred[:-1]]]
    b=route[np.r_[True,route[1:]!=route[:-1]]]
    # Longest common subsequence using bit-parallel integer updates.
    masks={}
    for i,v in enumerate(b): masks[int(v)]=masks.get(int(v),0)|(1<<i)
    state=0
    for v in a:
        matches=masks.get(int(v),0)|state
        state=matches & ~(matches-((state<<1)|1))
    return 2*state.bit_count()/(len(a)+len(b))
