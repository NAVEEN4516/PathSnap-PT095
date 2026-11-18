"""Candidate confusion counts and development-trip size diagnostics."""
import csv
import json
import numpy as np
from PIL import Image, ImageDraw
from experiment import collect, write_csv
from models import predict
from prepare import ROOT

def main():
    results = ROOT / "results"
    trips = collect(sorted((ROOT / "cache/student_test").glob("*.npz")))
    if not trips:
        raise ValueError("Run prepare.py first")
    rows = []
    models = sorted((ROOT / "artifacts").glob("*_*.json"))
    image = Image.new("RGB", (780, 100 + 75 * len(models)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Student-test candidate confusion counts", fill="black", font_size=24)
    draw.text((20, 62), "Each observation selects one candidate. All ten trips pooled.", fill="#35564a", font_size=16)
    for index, file in enumerate(models):
        m = json.loads(file.read_text())
        counts = np.zeros((2, 2), dtype=int)
        for t in trips.values():
            truth = t["ids"] == t["labels"][:, None]
            selected = np.zeros_like(truth)
            selected[np.arange(len(truth)), predict(m, t["x"]).argmax(1)] = True
            counts += np.bincount((2 * truth + selected).ravel(), minlength=4).reshape(2, 2)
        tn, fp, fn, tp = map(int, counts.ravel())
        rows.append(dict(model=file.stem, TN=tn, FP=fp, FN=fn, TP=tp))
        y = 100 + index * 75
        draw.text((20, y), file.stem, fill="black", font_size=18)
        for j, (name, value) in enumerate(zip(("TN", "FP", "FN", "TP"), (tn, fp, fn, tp))):
            draw.text((28 + j * 176, y + 28), f"{name}   {value:,}", fill="#185640", font_size=17)
    write_csv(results / "candidate_confusion.csv", rows)
    image.save(results / "plots/candidate_confusion.png")
    with (results / "inventory.csv").open() as f:
        inventory = list(csv.DictReader(f))
    image = Image.new("RGB", (1000, 430), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), "GPS observations per trip, sorted within each split", fill="black", font_size=23)
    largest = max(int(r["observations"]) for r in inventory)
    x = 72
    for split, color in [("train", "#207965"), ("student_test", "#d37841")]:
        for value in sorted(int(r["observations"]) for r in inventory if r["split"] == split):
            draw.rectangle((x, 350 - value * 265 / largest, x + 7, 350), fill=color)
            x += 10
    draw.text((72, 370), "80 training trips", fill="#207965", font_size=16)
    draw.text((800, 370), "10 student-test trips", fill="#a3481a", font_size=16)
    for value in (0, 5000, 10000, 15000):
        draw.text((4, 343 - value * 265 / largest), f"{value:,}", fill="black", font_size=13)
    image.save(results / "plots/trip_lengths.png")

if __name__ == "__main__":
    main()
