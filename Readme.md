# dbx-mlops-taxi (Project 1)
Foundational MLOps on Databricks using the NYC Taxi dataset.

## What’s inside
- **Delta tables** (Bronze/Silver/Gold)
- **MLflow** experiment tracking + Model Registry
- **Batch scoring** job
- **Basic tests** with pytest

## Quick start (Databricks)
1. Create (or use) a cluster and a Unity Catalog catalog/schema (or `hive_metastore` if UC is not enabled).
   - Catalog: `mlops_demo`
   - Schema: `taxi`
2. Import this repo into **Repos** or upload the files.
3. Open and run the notebooks in order:
   1) `notebooks/01_ingest_bronze.py`
   2) `notebooks/02_silver_clean.py`
   3) `notebooks/03_gold_features.py`
   4) `notebooks/10_train.py`
   5) `notebooks/20_batch_score.py`
4. (Optional) Configure a Databricks **Job** using `resources/job_template.json` as a guide.

## Requirements
See `requirements.txt`. Install into your cluster or use `%pip install -r /Workspace/Repos/<you>/dbx-mlops-taxi/requirements.txt` in a notebook.

## Tables created
- `mlops_demo.taxi.bronze_trips`
- `mlops_demo.taxi.silver_trips`
- `mlops_demo.taxi.gold_training`
- `mlops_demo.taxi.batch_predictions`

## Model Registry
- Model name: `taxi_fare_rf`
