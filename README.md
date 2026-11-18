# PathSnap PT-095

Distributed map matching for the supplied PathSnap hackathon dataset.

## Repository assembly

The complete application is published on main. The four member branches are available for follow-up work.

| Member | Branch | Responsibility |
|---|---|---|
| 1 | member-1-data | Data loading, candidate generation, reference alignment, inventory |
| 2 | member-2-models | Five models, trip cross-validation, Bayesian search, trained artifacts |
| 3 | member-3-decoder | Viterbi, metrics, inference service and regression checks |
| 4 | member-4-deployment | FastAPI, browser UI, maps, charts and deployment |

## Installation

Use Python 3.12. From the assembled repository run:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Extract the supplied dataset to a sibling directory named `Student folder/Student folder`, containing `train` and `student_test`. Alternatively supply `--data` to prepare.py and set PATHSNAP_TRIPS for inference.

## Reproduce experiments

```powershell
.venv/Scripts/python prepare.py
.venv/Scripts/python experiment.py
.venv/Scripts/python plots.py
.venv/Scripts/python visualize_map.py
.venv/Scripts/python verify.py
```

Training uses up to 300 randomly selected observations from each training trip, including all candidates of those observations. Three folds keep trips together. Validation and student-test scoring use every observation. Seed is 95. The original run completed all 80 training and 10 student-testing trips.

Five independently coded NumPy classifiers are logistic regression, Gaussian naive Bayes, histogram naive Bayes, regularised LDA and a CART-style classification tree. CART tests five quantile thresholds per feature. Eight Gaussian-process Bayesian trials tune the best family. This is a compact CPU baseline. Full configurations and limitations are in docs/METHODOLOGY.md.

## Laptop 1

Set PATHSNAP_TRIPS to the directory containing the ten chosen trip folders.

```powershell
$env:PATHSNAP_TRIPS="C:/path/to/student_test"
.venv/Scripts/python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

For final evaluation, point PATHSNAP_TRIPS at the organiser-provided evaluator trips. PATHSNAP_LABELS may point to a separate directory with matching trip subfolders holding reference files. Predictions finish before reference files are opened for scoring. The UI receives only predicted geometry, GPS observations and aggregate scores.

## Laptop 2

Copy the ui directory to the second laptop and run:

```powershell
$env:PATHSNAP_API="http://LAPTOP_1_IP:8000"
python ui/server.py
```

Open http://127.0.0.1:8501 on Laptop 2. Both laptops must be able to reach port 8000 on Laptop 1. The UI server proxies requests to Laptop 1. Observe Uvicorn logs on Laptop 1 during the demonstration.

## API

GET /trips lists available trip IDs. POST /match/{trip_id} reconstructs one trip. FastAPI documents the endpoints at /docs. Missing labels produce a prediction with null metrics. Unknown IDs return 404.

## Verification status

Model experiments and reference-free inference are locally executable using NumPy. Check results/verification.json for the completed checks. GitHub Actions passed the live FastAPI/Uvicorn request, UI proxy, server-side scoring and Folium HTML checks on a synthetic trip. [Verification run](https://github.com/NAVEEN4516/PathSnap-PT095/actions/runs/44248380610). Physical two-laptop evaluation remains pending. The ten reserved evaluator trips were not supplied.

The UI provides an offline geographic plot. visualize_map.py generates the required interactive Folium development map including GPS, candidate roads, reference and prediction. Reference layers are restricted to development analysis.

## Contribution and disclosure

The implementation and initial experiments were prepared with ChatGPT assistance. Team members should run, inspect and understand their assigned components, record their own changes and disclose assistance according to event policy. The PDF technical report is Deliverable 2 and will be prepared after deployment verification.

Run python deployment_smoke.py after installation to test the real HTTP API, UI proxy and Folium HTML on a synthetic trip. The same check is configured in GitHub Actions. See results/original_search_summary.csv for all eight measured optimisation scores.
