"""After dependency installation, verify the real FastAPI HTTP interface."""
import json
from fastapi.testclient import TestClient
from api import app
from prepare import ROOT
with TestClient(app) as client:
    response=client.get("/trips")
    assert response.status_code==200
    trips=response.json()["trips"]
    assert len(trips)==10
    assert client.post("/match/invalid").status_code==404
    rows=[]
    for trip in trips:
        response=client.post("/match/"+trip)
        assert response.status_code==200
        result=response.json()
        assert len(result["arc_ids"])==result["observations"]
        assert "reference" not in result and "labels" not in result
        rows.append({"trip":trip,"status":200,"observations":result["observations"]})
    (ROOT/"results/http_verification.json").write_text(json.dumps(rows,indent=2))
print("Passed API checks for",len(trips),"trips")
