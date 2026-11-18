"""Run from the assembled repository. Uses every supplied student-test trip."""
import json,tempfile,shutil
from pathlib import Path
import numpy as np
from prepare import ROOT,DEFAULT_DATA
from service import match,reconstruct,trip_dirs

def main():
    artifact=json.loads((ROOT/"artifacts/model.json").read_text())
    rows=[]
    for name,folder in trip_dirs().items():
        result=match(name)
        track=np.loadtxt(folder/(name+".track"),ndmin=2)
        assert result["observations"]==len(track)
        assert len(result["arc_ids"])==len(track)
        assert result["metrics"] is not None
        assert "labels" not in result and "reference" not in result and "route" not in result
        rows.append({k:result[k] for k in ("trip","observations","inference_seconds","metrics")})
    folder=DEFAULT_DATA/"student_test"/"00000052"
    original=reconstruct(folder,artifact)
    with tempfile.TemporaryDirectory() as temp:
        clean=Path(temp)/folder.name
        clean.mkdir()
        for extension in (".arcs",".nodes",".track"):
            shutil.copy2(folder/(folder.name+extension),clean/(folder.name+extension))
        without_labels=reconstruct(clean,artifact)
        assert np.array_equal(original[1],without_labels[1])
        assert np.array_equal(original[4],without_labels[4])
    payload={"trip_checks":rows,"inference_without_route_passed":True,
             "fastapi_http":"pending dependency installation",
             "folium_map":"pending dependency installation","two_laptop":"pending physical devices"}
    (ROOT/"results/verification.json").write_text(json.dumps(payload,indent=2))
    print("Passed",len(rows),"trip checks and inference without reference file")
if __name__=="__main__":
    main()
