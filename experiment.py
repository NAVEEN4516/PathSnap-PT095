"""Reproducible group CV, held-out evaluation and Gaussian-process Bayesian search."""
import csv,json,math,time
from pathlib import Path
import numpy as np
from prepare import ROOT
from models import DEFAULTS,SPACE,fit,predict
from metrics import evaluate,sequence_similarity
from decoder import decode,CONFIG
from geometry import FEATURES

SEED=95
CAP=300
OUT=ROOT/"results"
def write_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def collect(files):
    trips={}
    for f in files:
        with np.load(f) as z:trips[f.stem]={k:z[k] for k in z.files}
    return trips

def training(trips,keys):
    xs=[]; ys=[]
    for key in keys:
        t=trips[key]; rng=np.random.default_rng(SEED+int(key))
        obs=rng.choice(len(t["ids"]),min(CAP,len(t["ids"])),replace=False)
        x=t["x"][obs]
        xs.append(x.reshape(-1,x.shape[-1]))
        ys.append((t["ids"][obs]==t["labels"][obs,None]).ravel())
    return np.concatenate(xs),np.concatenate(ys)

def stats(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("trip","method")}

def test(model,trips,keys,with_decoder=False):
    rows=[]
    for key in keys:
        t=trips[key]; start=time.perf_counter()
        p=predict(model,t["x"])
        choices=p.argmax(1)
        methods=[("classifier",choices,time.perf_counter()-start)]
        if with_decoder:
            tick=time.perf_counter()
            path=decode(t["ids"],t["nodes"],t["time"],p)
            methods.append(("viterbi",path,methods[0][2]+time.perf_counter()-tick))
        for method,choice,seconds in methods:
            values=evaluate(t["ids"],t["nodes"],t["labels"],p,choice)
            pred=t["ids"][np.arange(len(choice)),choice]
            values["sequence_f1"]=sequence_similarity(pred,t["route"])
            rows.append(dict(trip=key,method=method,**values,ms_1000=seconds*1e6/len(p)))
    return rows

def suggest(space,used,values):
    # Fixed-length-scale RBF GP on the ordered search grid; maximise expected improvement.
    grid=np.linspace(0,1,len(space))
    seen=grid[used]
    kernel=lambda a,b:np.exp(-0.5*((a[:,None]-b[None,:])/.22)**2)
    cross=kernel(seen,grid)
    L=np.linalg.cholesky(kernel(seen,seen)+1e-5*np.eye(len(seen)))
    proj=np.linalg.solve(L,cross)
    y=np.asarray(values); z=(y-y.mean())/max(y.std(),1e-5)
    mean=proj.T@np.linalg.solve(L,z)
    sigma=np.sqrt(np.maximum(1-(proj**2).sum(0),1e-12))
    delta=mean-z.max()-.01
    q=delta/sigma
    cdf=np.array([.5*(1+math.erf(i/math.sqrt(2))) for i in q])
    ei=delta*cdf+sigma*np.exp(-q*q/2)/math.sqrt(2*math.pi)
    ei[used]=-np.inf
    return int(ei.argmax())

def main():
    OUT.mkdir(exist_ok=True)
    train=collect(sorted((ROOT/"cache/train").glob("*.npz")))
    student=collect(sorted((ROOT/"cache/student_test").glob("*.npz")))
    if len(train)!=80 or len(student)!=10:raise ValueError("Run prepare.py on both supplied splits first")
    keys=list(train)
    np.random.default_rng(SEED).shuffle(keys)
    folds=[keys[i::3] for i in range(3)]
    (OUT/"folds.json").write_text(json.dumps(folds,indent=2))
    batches=[training(train,[k for k in keys if k not in val]) for val in folds]
    full=training(train,keys)
    def cv(name,param):
        scores=[]
        for (x,y),val in zip(batches,folds):
            scores.append(stats(test(fit(name,param,x,y),train,val))["macro_f1"])
        return scores
    comparison=[]; trials=[]; detail=[]; cvlog=[]
    def record(name,param,stage,scores):
        tick=time.perf_counter()
        m=fit(name,param,*full)
        elapsed=time.perf_counter()-tick
        artifact_dir=ROOT/"artifacts"
        artifact_dir.mkdir(exist_ok=True)
        (artifact_dir/(name+"_"+stage+".json")).write_text(json.dumps(m,indent=2))
        seen=stats(test(m,train,keys))
        rows=test(m,student,list(student))
        detail.extend(dict(model=name,stage=stage,**r) for r in rows)
        s=stats(rows)
        comparison.append(dict(model=name,stage=stage,param=float(param),cv_f1=np.mean(scores),
            cv_sd=np.std(scores),train_f1=seen["macro_f1"],fit_seconds=elapsed,**s))
        cvlog.extend(dict(model=name,stage=stage,fold=i+1,macro_f1=v) for i,v in enumerate(scores))
        write_csv(OUT/"comparison.csv",comparison)
        return m
    for name,param in DEFAULTS.items():
        print("Comparing",name,flush=True)
        record(name,param,"baseline",cv(name,param))
    best=max(comparison,key=lambda r:r["cv_f1"])
    name=best["model"]; space=SPACE[name]
    # Include the baseline in selection even when search does not improve on it.
    initial=[0,len(space)//2,len(space)-1]
    used=[]; values=[]
    for trial in range(8):
        idx=initial[trial] if trial<3 else suggest(space,used,values)
        tick=time.perf_counter()
        scores=cv(name,space[idx])
        values.append(float(np.mean(scores)));used.append(idx)
        trials.append(dict(trial=trial+1,model=name,param=float(space[idx]),cv_f1=values[-1],
                 fold1=scores[0],fold2=scores[1],fold3=scores[2],seconds=time.perf_counter()-tick))
        write_csv(OUT/"bayesian_trials.csv",trials)
        print("Bayesian trial",trial+1,values[-1],flush=True)
    trial=max(trials,key=lambda r:r["cv_f1"])
    param=trial["param"]
    scores=[trial["fold1"],trial["fold2"],trial["fold3"]]
    tuned=record(name,param,"search_best",scores)
    chosen=tuned if np.mean(scores)>=best["cv_f1"] else fit(name,DEFAULTS[name],*full)
    dest=ROOT/"artifacts";dest.mkdir(exist_ok=True)
    (dest/"model.json").write_text(json.dumps(dict(model=chosen,decoder=CONFIG,features=FEATURES,seed=SEED),indent=2))
    write_csv(OUT/"per_model_trip.csv",detail)
    write_csv(OUT/"cross_validation.csv",cvlog)
    write_csv(OUT/"decoder_by_trip.csv",test(chosen,student,list(student),True))
    print("Selected",name,chosen["param"],flush=True)
if __name__=="__main__":main()
