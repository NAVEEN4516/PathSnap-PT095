"""Five binary candidate classifiers, implemented in NumPy. Models export to JSON."""
import numpy as np

DEFAULTS = {"Logistic":0.01,"GaussianNB":0.05,"HistogramNB":12,"LDA":0.1,"CART":5}
SPACE = {"Logistic":np.geomspace(0.0001,0.3,31),"GaussianNB":np.geomspace(0.001,0.8,31),
         "HistogramNB":np.arange(4,36),"LDA":np.geomspace(0.001,1,31),"CART":np.arange(2,13)}

def sigmoid(a):
    return 1/(1+np.exp(-np.clip(a,-35,35)))

def grow(x,y,depth):
    n=len(y)
    leaf={"p":float((y.sum()+1)/(n+2))}
    if depth==0 or n<50 or y.min()==y.max(): return leaf
    best=None
    for j,col in enumerate(x.T):
        edges=np.unique(np.quantile(col,[.1,.25,.5,.75,.9]))
        bins=np.searchsorted(edges,col,side="left")
        count=np.bincount(bins,minlength=len(edges)+1).cumsum()[:-1]
        pos=np.bincount(bins,weights=y,minlength=len(edges)+1).cumsum()[:-1]
        valid=(count>=20)&(n-count>=20)
        score=np.where(valid,pos**2/np.maximum(count,1)+(y.sum()-pos)**2/np.maximum(n-count,1),-np.inf)
        k=score.argmax()
        if valid[k] and (best is None or score[k]>best[0]):
            best=(score[k],j,float(edges[k]))
    if best is None or best[0]<=float(y.sum())**2/n+1e-8:return leaf
    _,j,cut=best
    left=x[:,j]<=cut
    return {"j":j,"cut":cut,"left":grow(x[left],y[left],depth-1),
            "right":grow(x[~left],y[~left],depth-1)}

def fit(name,param,x,y):
    x=np.asarray(x,dtype=float)
    mu=x.mean(0); sd=np.maximum(x.std(0),1e-5)
    x=(x-mu)/sd
    prior=np.clip(y.mean(),1e-6,1-1e-6)
    m=dict(name=name,param=float(param),mu=mu.tolist(),sd=sd.tolist(),
           intercept=float(np.log(prior/(1-prior))))
    if name=="Logistic":
        design=np.column_stack([np.ones(len(x)),x])
        w=np.zeros(design.shape[1])
        reg=np.r_[0,np.full(len(w)-1,param)]
        for _ in range(18):
            p=sigmoid(design@w)
            h=design.T@(design*(p*(1-p))[:,None])/len(x)+np.diag(reg+1e-8)
            delta=np.linalg.solve(h,(design.T@(p-y))/len(x)+reg*w)
            w-=delta
            if abs(delta).max()<1e-5:break
        m.update(w=w[1:].tolist(),intercept=float(w[0]))
    elif name=="CART":
        m["tree"]=grow(x,y,int(param))
    elif name=="HistogramNB":
        bins=[]; odds=[]
        for col in x.T:
            edge=np.unique(np.quantile(col,np.linspace(0,1,int(param)+1)[1:-1]))
            b=np.searchsorted(edge,col)
            counts=[np.bincount(b[y==c],minlength=len(edge)+1)+1 for c in [0,1]]
            bins.append(edge.tolist())
            odds.append(np.log((counts[1]/counts[1].sum())/(counts[0]/counts[0].sum())).tolist())
        m.update(edges=bins,odds=odds)
    else:
        parts=[x[y==c] for c in [0,1]]
        a,b=[group.mean(0) for group in parts]
        if name=="GaussianNB":
            m.update(means=[a.tolist(),b.tolist()],variances=[(v.var(0)+param).tolist() for v in parts])
        elif name=="LDA":
            cov=sum((v-u).T@(v-u) for v,u in zip(parts,[a,b]))/len(y)
            w=np.linalg.solve(cov+param*np.eye(len(a)),b-a)
            m.update(w=w.tolist(),intercept=float(m["intercept"]-(a+b)@w/2))
        else: raise ValueError(name)
    return m

def predict(m,x):
    shape=x.shape[:-1]
    x=(x.reshape(-1,x.shape[-1])-m["mu"])/m["sd"]
    if m["name"]=="CART":
        p=np.empty(len(x))
        def walk(node,idx):
            if "p" in node:
                p[idx]=node["p"];return
            left=x[idx,node["j"]]<=node["cut"]
            walk(node["left"],idx[left]);walk(node["right"],idx[~left])
        walk(m["tree"],np.arange(len(x)))
    else:
        score=np.full(len(x),m["intercept"])
        if "w" in m:score+=x@np.asarray(m["w"])
        elif "edges" in m:
            for col,edge,odds in zip(x.T,m["edges"],m["odds"]):
                score+=np.asarray(odds)[np.searchsorted(edge,col)]
        else:
            for c,sign in [(0,-1),(1,1)]:
                var=np.asarray(m["variances"][c])
                score+=sign*(-0.5*(np.log(var)+(x-m["means"][c])**2/var).sum(1))
        p=sigmoid(score)
    return np.clip(p,1e-9,1-1e-9).reshape(shape)
