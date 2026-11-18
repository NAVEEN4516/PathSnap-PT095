"""Log-space Viterbi over directed road candidates; restart at sampling breaks."""
import numpy as np
CONFIG={"adjacent_cost":0.65,"jump_cost":4.0,"gap_s":10.0}
def decode(ids,nodes,times,probs,config=CONFIG):
    n,k=ids.shape
    # Relative log-likelihood of one chosen positive among other negatives.
    emission=np.log(probs)-np.log1p(-probs)
    back=np.zeros((n,k),dtype=int)
    ends=[]
    score=emission[0].copy()
    for i in range(1,n):
        dt=times[i]-times[i-1]
        if dt<=0 or dt>config["gap_s"]:
            ends.append((i-1,int(score.argmax())))
            score=emission[i].copy()
            continue
        same=ids[i-1,:,None]==ids[i,None,:]
        adjacent=nodes[i-1,:,1,None]==nodes[i,None,:,0]
        cost=np.where(same,0.,np.where(adjacent,config["adjacent_cost"],config["jump_cost"]))
        # Normalise each previous state's distribution over the current roads.
        log_transition = -cost - np.log(np.exp(-cost).sum(axis=1, keepdims=True))
        paths=score[:,None]+log_transition
        back[i]=paths.argmax(0)
        score=paths.max(0)+emission[i]
        score-=score.max()
    ends.append((n-1,int(score.argmax())))
    chosen=np.zeros(n,dtype=int)
    begin=0
    for end,state in ends:
        for i in range(end,begin-1,-1):
            chosen[i]=state
            state=back[i,state]
        begin=end+1
    return chosen
