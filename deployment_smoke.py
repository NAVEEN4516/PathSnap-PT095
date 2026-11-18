"""Exercise both web processes and Folium using a small synthetic road graph."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

from visualize_map import main as make_map

def get(url, method="GET"):
    with urlopen(Request(url, method=method), timeout=10) as response:
        return response.read()

def ready(url, process):
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("Server stopped before becoming ready")
        try:
            get(url)
            return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(url)

def main():
    with tempfile.TemporaryDirectory() as scratch:
        partition = Path(scratch)
        trip = partition / "00000001"
        trip.mkdir()
        for suffix, data in {
            "nodes": "8 49\n8.001 49\n8.002 49\n",
            "arcs": "0 1\n1 2\n1 0\n2 1\n",
            "track": "8.0001 49 1\n8.0009 49 2\n8.0019 49 3\n",
            "route": "0\n1\n",
        }.items():
            (trip / f"{trip.name}.{suffix}").write_text(data)
        env = dict(os.environ, PATHSNAP_TRIPS=str(partition), PATHSNAP_API="http://127.0.0.1:8000")
        processes = []
        try:
            for args, url in [
                (["-m", "uvicorn", "api:app"], "http://127.0.0.1:8000/trips"),
                (["ui/server.py"], "http://127.0.0.1:8501/"),
            ]:
                process = subprocess.Popen([sys.executable, *args], env=env)
                processes.append(process)
                ready(url, process)
            base = "http://127.0.0.1:8501"
            assert b"Route explorer" in get(base)
            assert json.loads(get(base + "/trips"))["trips"] == [trip.name]
            result = json.loads(get(base + "/match/" + trip.name, "POST"))
            assert len(result["arc_ids"]) == 3 and result["metrics"] is not None
            assert not {"labels", "reference", "route"}.intersection(result)
            output = partition / "map.html"
            make_map(trip, output)
            html = output.read_text(encoding="utf-8")
            assert all(layer in html for layer in ["Candidate roads", "GPS observations", "Reference route", "Predicted route"])
            print("Live HTTP, UI proxy, server-side scoring and Folium HTML passed.")
        finally:
            for process in reversed(processes):
                process.terminate()
                process.wait(timeout=10)

if __name__ == "__main__":
    main()
