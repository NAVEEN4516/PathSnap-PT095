"""Prepare candidate features and an auditable trip inventory."""
import argparse,csv,os
from pathlib import Path
import numpy as np
from geometry import load_trip,candidates,reference
ROOT=Path(__file__).resolve().parent
DEFAULT_DATA=Path(os.environ.get("PATHSNAP_DATA",ROOT.parent/"Student folder"/"Student folder"))

def connectivity(trip):
    parent=list(range(len(trip.nodes)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for a,b in trip.arcs.tolist():
        parent[find(a)]=find(b)
    return len({find(i) for i in range(len(parent))}),len(parent)-len(np.unique(trip.arcs))

def main(data):
    rows=[]
    for split in ("train","student_test"):
        trips=sorted(p for p in (data/split).iterdir() if p.is_dir())
        if not trips: raise ValueError("Empty partition")
        out=ROOT/"cache"/split
        out.mkdir(parents=True,exist_ok=True)
        for folder in trips:
            t=load_trip(folder)
            ids,x=candidates(t)
            route,labels=reference(folder,t)
            dt=np.diff(t.track[:,2])
            components,isolated=connectivity(t)
            np.savez(out/(t.name+".npz"),x=x,ids=ids,labels=labels,nodes=t.arcs[ids],
                     time=t.track[:,2],route=route)
            rows.append(dict(split=split,trip=t.name,observations=len(labels),arc_count=len(t.arcs),
                node_count=len(t.nodes),route_entries=len(route),invalid_gps=t.missing,
                nonincreasing=int((dt<=0).sum()),one_second=int((dt==1).sum()),
                gaps_over_10=int((dt>10).sum()),max_gap=float(dt.max()) if len(dt) else 0,
                lon_min=t.track[:,0].min(),lon_max=t.track[:,0].max(),lat_min=t.track[:,1].min(),
                lat_max=t.track[:,1].max(),weak_components=components,isolated_nodes=isolated,
                candidate_misses=int((~(ids==labels[:,None]).any(1)).sum()),
                input_bytes=sum(p.stat().st_size for p in folder.glob(t.name+".*"))))
            print(split,t.name,len(labels),flush=True)
    results=ROOT/"results"
    results.mkdir(exist_ok=True)
    with (results/"inventory.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,rows[0]); w.writeheader(); w.writerows(rows)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,default=DEFAULT_DATA)
    main(p.parse_args().data)
