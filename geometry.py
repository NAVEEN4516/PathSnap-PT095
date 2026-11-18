"""Trip-local geometry. No reference route is read on the inference path."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np

FEATURES = ["log_distance_m", "heading_agreement", "log_length_m", "projection_fraction",
            "log_speed_mps", "relative_distance_m", "distance_x_agreement"]
K = 6
CELL = 80.0

@dataclass
class Trip:
    name: str
    nodes: np.ndarray
    arcs: np.ndarray
    track: np.ndarray
    xy: np.ndarray
    gps: np.ndarray
    missing: int

def load_trip(folder):
    folder = Path(folder)
    def read(ext, dtype):
        return np.loadtxt(folder / (folder.name + ext), dtype=dtype, ndmin=2)
    nodes, arcs, track = read(".nodes", float), read(".arcs", int), read(".track", float)
    if nodes.shape[1] != 2 or arcs.shape[1] != 2 or track.shape[1] != 3:
        raise ValueError("Unexpected trip column count")
    if not len(arcs) or arcs.min() < 0 or arcs.max() >= len(nodes) or not np.isfinite(nodes).all():
        raise ValueError("Invalid road graph")
    valid = np.isfinite(track).all(1) & (abs(track[:,0]) <= 180) & (abs(track[:,1]) <= 90)
    missing = int((~valid).sum())
    track = track[valid]
    if not len(track):
        raise ValueError("Trip has no valid observations")
    origin = track[0,:2]
    scale = np.pi / 180 * 6371008.8 * np.array([np.cos(np.deg2rad(track[:,1].mean())), 1.])
    return Trip(folder.name, nodes, arcs, track, (nodes-origin)*scale,
                (track[:,:2]-origin)*scale, missing)

def segments(trip):
    start = trip.xy[trip.arcs[:,0]]
    return start, trip.xy[trip.arcs[:,1]] - start

def project(points, starts, vectors):
    norm = np.maximum((vectors*vectors).sum(-1), 1e-10)
    offsets = points - starts
    fraction = np.clip((offsets*vectors).sum(-1)/norm, 0, 1)
    distance = np.linalg.norm(offsets - fraction[...,None]*vectors, axis=-1)
    return distance, fraction

def candidates(trip):
    start, vec = segments(trip)
    length = np.linalg.norm(vec,axis=1)
    # Index the entire segment, including interiors of long roads.
    pieces = np.maximum(1, np.ceil(length/(CELL/2)).astype(int))
    ids = np.repeat(np.arange(len(vec)), pieces+1)
    first = np.repeat(np.cumsum(pieces+1)-(pieces+1), pieces+1)
    sample = start[ids] + vec[ids]*((np.arange(len(ids))-first)/pieces[ids])[:,None]
    def cell_keys(xy):
        return np.floor(xy/CELL).astype(int)
    cell = cell_keys(sample)
    order = np.lexsort((ids,cell[:,1],cell[:,0]))
    grid = {}
    boundaries = np.flatnonzero(np.any(np.diff(cell[order],axis=0),axis=1))+1
    for group in np.split(order,boundaries):
        grid[tuple(cell[group[0]])] = np.unique(ids[group])
    obs_cell = cell_keys(trip.gps)
    order = np.lexsort((obs_cell[:,1],obs_cell[:,0]))
    boundaries = np.flatnonzero(np.any(np.diff(obs_cell[order],axis=0),axis=1))+1
    k = min(K,len(vec))
    chosen = np.empty((len(order),k),int)
    for group in np.split(order,boundaries):
        cx,cy = obs_cell[group[0]]
        nearby = [grid[cx+i,cy+j] for i in range(-2,3) for j in range(-2,3) if (cx+i,cy+j) in grid]
        pool = np.unique(np.concatenate(nearby)) if nearby else np.array([],int)
        if len(pool)<k:
            pool = np.arange(len(vec))
        for chunk in np.array_split(group,max(1,int(np.ceil(len(group)*len(pool)/300000)))):
            dist,_=project(trip.gps[chunk,None,:],start[pool],vec[pool])
            chosen[chunk]=pool[np.argsort(dist,axis=1,kind="stable")[:,:k]]
    dist, frac = project(trip.gps[:,None,:],start[chosen],vec[chosen])
    i = np.arange(len(trip.gps))
    left, right = np.maximum(0,i-2), np.minimum(len(i)-1,i+2)
    motion = trip.gps[right]-trip.gps[left]
    moved = np.linalg.norm(motion,axis=1)
    dt = trip.track[right,2]-trip.track[left,2]
    speed = np.divide(moved,dt,out=np.zeros_like(moved),where=dt>0)
    agreement = (vec[chosen]*motion[:,None,:]).sum(2)/np.maximum(length[chosen]*moved[:,None],1e-8)
    agreement[(moved<2)|(dt<=0)|(dt>20)] = 0
    ld = np.log1p(dist)
    x = np.stack([ld,agreement,np.log1p(length[chosen]),frac,
                  np.broadcast_to(np.log1p(speed[:,None]),dist.shape),
                  np.log1p(dist-dist.min(1,keepdims=True)),ld*agreement],axis=-1)
    return chosen,x.astype("float32")

def reference(folder,trip):
    """Align GPS to route occurrences with monotone dynamic programming.
    Only training and scoring call this function. Occurrences can repeat."""
    route=np.loadtxt(Path(folder)/(trip.name+".route"),dtype=int,ndmin=1)
    if not len(route) or route.min()<0 or route.max()>=len(trip.arcs):
        raise ValueError("Invalid reference arc index")
    starts,vec=segments(trip)
    starts,vec=starts[route],vec[route]
    n=len(route)
    pred=np.empty((len(trip.track),n),dtype=np.int32)
    cost=np.zeros(n)
    positions=np.arange(n)
    for begin in range(0,len(trip.track),128):
        distances,_=project(trip.gps[begin:begin+128,None,:],starts,vec)
        for j,d in enumerate(distances):
            prefix=np.minimum.accumulate(cost)
            # Ties choose the earliest equally good predecessor.
            lower=np.r_[True,prefix[1:]<prefix[:-1]]
            pred[begin+j]=np.maximum.accumulate(np.where(lower,positions,0))
            cost=prefix+np.minimum(d,100.)
    idx=int(np.argmin(cost))
    labels=np.empty(len(trip.track),int)
    for t in range(len(labels)-1,-1,-1):
        labels[t]=route[idx]
        idx=pred[t,idx]
    return route,labels
