"""Offline analysis of a development trip; reference layers never enter the live UI."""
import argparse,json
from pathlib import Path
import folium
import numpy as np
from prepare import DEFAULT_DATA,ROOT
from service import reconstruct
from geometry import reference

def main(folder,dest):
    artifact=json.loads((ROOT/"artifacts/model.json").read_text())
    t,ids,nodes,p,choice=reconstruct(folder,artifact)
    pred=ids[np.arange(len(choice)),choice]
    route,_=reference(folder,t)
    gps=t.track[:,:2][:,::-1]
    m=folium.Map(location=gps.mean(0).tolist(),tiles="OpenStreetMap",control_scale=True,prefer_canvas=True)
    for name,color,width,arcs in [("Candidate roads","#86968e",2,np.unique(ids)),
             ("Reference route","#bb4e22",7,np.unique(route)),("Predicted route","#126c59",3,np.unique(pred))]:
        group=folium.FeatureGroup(name=name).add_to(m)
        lines=t.nodes[t.arcs[arcs]][:,:,::-1].tolist()
        folium.PolyLine(lines,color=color,weight=width,opacity=.8,tooltip=name).add_to(group)
    obs=folium.FeatureGroup(name="GPS observations").add_to(m)
    for pt in gps:
        folium.CircleMarker(pt.tolist(),radius=1,color="#3359a6",weight=1,fill=True).add_to(obs)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([gps.min(0).tolist(),gps.max(0).tolist()])
    dest.parent.mkdir(parents=True,exist_ok=True)
    m.save(str(dest))
if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--trip",type=Path,default=DEFAULT_DATA/"student_test"/"00000052")
    arg.add_argument("--out",type=Path,default=ROOT/"results/development_map.html")
    args=arg.parse_args();main(args.trip,args.out)
