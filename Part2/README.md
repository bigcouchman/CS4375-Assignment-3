# Tweets Clustering using K-means (Jaccard Distance)

This repository is an implementation of tweet clustering for the UCI **Health News in Twitter** dataset.

- Similarity metric: Jaccard distance over unordered tweet word sets.
- Clustering algorithm: K-means-style iterative clustering with medoid centroid update (tweet with minimum total distance in each cluster).
- Required preprocessing: remove id/timestamp, remove `@mentions`, remove URL, normalize hashtags (`#word -> word`), lowercase all words.
- Experiment output: supports 5+ values of K and reports SSE and cluster sizes.
- Run tracking: every run prints important metrics in terminal and writes history artifacts to disk.

## Project Structure

```
Part2/
├─ run_experiment.py
├─ scripts/
│  └─ run_assignment.ps1
├─ requirements.txt
├─ README.md
├─ src/
│  └─ tweet_clustering/
│     ├─ __init__.py
│     ├─ types.py
│     ├─ preprocessing.py
│     ├─ jaccard_kmeans.py
│     └─ experiment.py
└─ tests/
   ├─ test_experiment_outputs.py
   ├─ test_preprocessing.py
   └─ test_kmeans.py
```

## Quick Start (Windows PowerShell)

From Project Root: 

```powershell
cd Part2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\scripts\run_assignment.ps1 -InputFile healthdataset/Health-Tweets/bbchealth.txt
```

Use `python` or `py -3` in the commands above.
If script execution is blocked, run:

```
powershell -ExecutionPolicy Bypass -File .\scripts\run_assignment.ps1 -InputFile healthdataset/Health-Tweets/bbchealth.txt
```

## 1. Setup

### Prerequisites

- Python 3.9+
- No third-party libraries required

### Create and activate virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Download Dataset

1. Download from UCI:
   - https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter
2. Use the zip file:
   - https://archive.ics.uci.edu/static/public/438/health+news+in+twitter.zip
3. Unzip it.
4. Choose one file from `Health-Tweets/` (for example `usnewshealth.txt` or `bbchealth.txt`).

## 3. Assignment Preprocessing Rules

For each line in the selected file (`tweet_id|timestamp|tweet_text`):

1. Remove tweet id and timestamp.
2. Remove any word starting with `@`.
3. Remove hashtag symbol `#`
4. Remove URLs.
5. Convert all words to lowercase.

The tweet is then represented as an unordered set of words for Jaccard distance.

## 4. Run Clustering Experiment (5+ K values)

Run from the Part2 folder:

```powershell
python run_experiment.py --input-file healthdataset/Health-Tweets/bbchealth.txt --k-values 5 10 15 20 25 --max-iter 40 --seed 42 --n-init 1 --init-strategy random
```

Alternative helper script (runs tests first, then experiment):

```powershell
.\scripts\run_assignment.ps1 -InputFile healthdataset/Health-Tweets/bbchealth.txt
```

### Optional flags

- `--output-csv results/kmeans_results.csv` : where to save the final table.
- `--keep-empty` : keep tweets that become empty after preprocessing (default is drop them).
- `--n-init 1` : number of restart trials per K.
- `--init-strategy random` : strategy (`random`, `kmedoids++`, or `hybrid`).
   - `hybrid` tries both `random` and `kmedoids++` for each restart and keeps the best SSE.

Baseline defaults are:

- `--n-init 1`
- `--init-strategy random`
- The CLI enforces at least 5 distinct K values per assignment requirement.

## 5. Output Format

Console output shows:

- Per-K table with: `K`, `SSE`, `SSE/tweet`, `iterations`, min/max cluster sizes, empty cluster count, and full cluster sizes.
- Table with: `Value of K`, `SSE`, `Size of each cluster`.
- Important run metrics summary with runtime, best SSE, best SSE/tweet, balance ratio, and iteration stats.

## 5.1 Files Produced After Every Run

Each run automatically writes all required metrics:

- `results/required_results.csv`
   - Table columns: Value of K, SSE, Size of each cluster.
- `results/kmeans_results.csv`
   - Latest run metrics table.
- `results/runs/run_<run_id>_assignment.csv`
   - Per-run table snapshot.
- `results/runs/run_<run_id>.csv`
   - Immutable snapshot of that specific run.
- `results/run_history.csv`
   - Appended history across all runs (one row per K per run).
- `results/latest_run_metrics.json`
   - Structured metrics + summary.

The metrics files include:

- `k`
- `sse`
- `sse_per_tweet`
- `iterations`
- `init_strategy`
- `init_trials`
- `best_init_seed`
- `best_init_strategy`
- `mean_trial_sse`
- `std_trial_sse`
- `min_cluster_size`
- `max_cluster_size`
- `empty_clusters`
- `imbalance_ratio`
- `cluster_sizes`

## 6. Testing

Run unit tests:

```powershell
python -m unittest discover -s tests -v
```

Tests cover:

- Jaccard distance correctness
- preprocessing rule compliance
- K-means clustering behavior on small controlled examples
- metrics/history file generation and summary logic

## 7. Reproducibility

- Use `--seed` for deterministic initialization.
- Because clustering is seed-sensitive, `--n-init` runs multiple restarts per K and keeps the best solution automatically.
- Internally, pairwise Jaccard distances are cached across trials to improve runtime for restart-heavy runs.

## 10. Quick Start (Copy/Paste)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_experiment.py --input-file healthdataset/Health-Tweets/bbchealth.txt --k-values 5 10 15 20 25 --max-iter 40 --seed 42 --n-init 1 --init-strategy random
python -m unittest discover -s tests -v
```
