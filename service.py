"""Backend operations shared by the API and the reproducibility checks."""
import json,os,re,time
from pathlib import Path
import numpy as np
from prepare import ROOT,DEFAULT_DATA
from geometry import candidates,load_trip,reference
from models import predict
from decoder import decode
from metrics import evaluate,sequence_similarity

def trip_dirs():
    partition=Path(os.environ.get("PATHSNAP_TRIPS",DEFAULT_DATA/"student_test"))
    if not partition.is_dir():
        raise FileNotFoundError("Set PATHSNAP_TRIPS to the directory containing trip folders")
    return {p.name:p for p in sorted(partition.iterdir()) if p.is_dir() and re.fullmatch(r"[0-9]{8}",p.name)}

def reconstruct(folder,artifact):
    trip=load_trip(folder)
    ids,x=candidates(trip)
    p=predict(artifact["model"],x)
    nodes=trip.arcs[ids]
    choice=decode(ids,nodes,trip.track[:,2],p,artifact["decoder"])
    return trip,ids,nodes,p,choice

def match(trip_id):
    folders=trip_dirs()
    if trip_id not in folders:raise KeyError("Unknown trip ID")
    folder=folders[trip_id]
    started=time.perf_counter()
    artifact=json.loads((ROOT/"artifacts/model.json").read_text())
    trip,ids,nodes,p,choice=reconstruct(folder,artifact)
    pred=ids[np.arange(len(choice)),choice]
    unique=pred[np.r_[True,pred[1:]!=pred[:-1]]]
    elapsed=time.perf_counter()-started
    response=dict(trip=trip_id,observations=len(pred),model=artifact["model"]["name"],
         inference_seconds=round(elapsed,3),arc_ids=pred.tolist(),metrics=None,
         segments=trip.nodes[trip.arcs[unique]].tolist(),gps=trip.track[:,:2].tolist())
    # Label files are opened only after the prediction is final.
    private=Path(os.environ.get("PATHSNAP_LABELS",folder.parent))/trip_id
    if (private/(trip_id+".route")).is_file():
        route,labels=reference(private,trip)
        m=evaluate(ids,nodes,labels,p,choice)
        response["metrics"]={k:m[k] for k in ("top1","macro_f1","continuity")}
        response["metrics"]["sequence_f1"]=sequence_similarity(pred,route)
    print(f"matched trip={trip_id} points={len(pred)} seconds={elapsed:.3f}",flush=True)
    return response
