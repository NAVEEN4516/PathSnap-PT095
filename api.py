from fastapi import FastAPI,HTTPException
from service import trip_dirs,match

app=FastAPI(title="PathSnap",version="1.0")

@app.get("/trips")
def trips():
    try:
        return {"trips":list(trip_dirs())}
    except FileNotFoundError as exc:
        raise HTTPException(503,str(exc)) from exc

@app.post("/match/{trip_id}")
def match_trip(trip_id:str):
    try:
        return match(trip_id)
    except KeyError as exc:
        raise HTTPException(404,"Unknown trip ID") from exc
    except (FileNotFoundError,ValueError) as exc:
        raise HTTPException(422,"Trip data or inference artifacts are unavailable or invalid") from exc
