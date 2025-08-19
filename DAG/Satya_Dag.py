from __future__ import annotations
import json
import logging
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator, get_current_context
from airflow.operators.empty import EmptyOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.operators.email import EmailOperator  # optional bonus if SMTP is configured


PROJECT_ID = "londonfinal"
BQ_LOCATION = "EU"  
GCS_BUCKET = "londonfinal-ml-artifacts"
ENABLE_EMAIL = False
EMAIL_TO = "satyareddy8782@gmail.com"


BQ_DATASET = "ml_pipeline_demo"   
BQ_TABLE = "cycle_hire_features"
GCS_PREFIX = "composer_ml_bike"


SQL_EXTRACT = """
CREATE OR REPLACE TABLE `{{ params.project_id }}.{{ params.bq_dataset }}.{{ params.bq_table }}` AS
SELECT
  rental_id,
  start_date,
  duration AS duration_sec,
  duration / 60.0 AS trip_duration_min,
  EXTRACT(HOUR FROM start_date)        AS hour_of_day,
  EXTRACT(DAYOFWEEK FROM start_date)   AS day_of_week,
  EXTRACT(MONTH FROM start_date)       AS month,
  IF(EXTRACT(DAYOFWEEK FROM start_date) IN (1,7), 1, 0) AS is_weekend
FROM `bigquery-public-data.london_bicycles.cycle_hire`
WHERE duration BETWEEN 60 AND 7200;
"""


def train_model():
    # Get runtime context
    context = get_current_context()
    ds_nodash = context["ds_nodash"]

    # ⬇ Lazy imports so the DAG file parses even if packages finish installing later
    import joblib
    import pandas as pd  # noqa: F401  # used implicitly via sklearn inputs
    from google.cloud import bigquery, storage
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    
    bq = bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)

   
    query = f"""
      SELECT
        trip_duration_min,
        hour_of_day,
        day_of_week,
        month,
        is_weekend
      FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
      LIMIT 200000
    """
    df = bq.query(query).result().to_dataframe(create_bqstorage_client=True)

    y = df["trip_duration_min"].astype(float)
    X = df[["hour_of_day", "day_of_week", "month", "is_weekend"]].astype(float)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(Xtr, ytr)

    preds = model.predict(Xte)
    mae = float(mean_absolute_error(yte, preds))
    rmse = float(mean_squared_error(yte, preds) ** 0.5)
    r2 = float(r2_score(yte, preds))

    
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET)

    model_path = f"{GCS_PREFIX}/models/{ds_nodash}/bike_duration_linear.joblib"
    metrics_path = f"{GCS_PREFIX}/metrics/{ds_nodash}/metrics.json"

    local_model = "/tmp/bike_duration_linear.joblib"
    joblib.dump(model, local_model)
    bucket.blob(model_path).upload_from_filename(local_model)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "rows_used": int(len(df))}
    local_metrics = "/tmp/metrics.json"
    with open(local_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    bucket.blob(metrics_path).upload_from_filename(local_metrics)

    summary = {
        "project_id": PROJECT_ID,
        "dataset": BQ_DATASET,
        "table": BQ_TABLE,
        "gcs_model": f"gs://{GCS_BUCKET}/{model_path}",
        "gcs_metrics": f"gs://{GCS_BUCKET}/{metrics_path}",
        **metrics,
    }
    logging.info("Training complete: %s", json.dumps(summary))
    return summary  # goes to XCom

def log_completion():
    context = get_current_context()
    summary = context["ti"].xcom_pull(task_ids="train_model")
    logging.info("🎉 Model training complete! Summary: %s", json.dumps(summary))

# ── DAG definition ─────────────────────────────────────────────────────────────
with DAG(
    dag_id="cc_bike_regression_dag_v1",
    start_date=datetime(2024, 1, 1),
    schedule="0 0 * * 0",  # weekly
    catchup=False,
    tags=["composer", "ml", "bigquery", "gcs"],
    default_args={"owner": "rakesh", "retries": 1},
) as dag:

    extract_features = BigQueryInsertJobOperator(
        task_id="extract_features",
        configuration={"query": {"query": SQL_EXTRACT, "useLegacySql": False}},
        location=BQ_LOCATION,  # EU
        params={"project_id": PROJECT_ID, "bq_dataset": BQ_DATASET, "bq_table": BQ_TABLE},
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    notify_log = PythonOperator(
        task_id="notify_log",
        python_callable=log_completion,
    )

    maybe_email = EmptyOperator(task_id="maybe_email")
    if ENABLE_EMAIL:
        maybe_email = EmailOperator(
            task_id="email_summary",
            to=EMAIL_TO,
            subject="Composer ML Pipeline Done",
            html_content="""
            <p>Check GCS for model & metrics.</p>
            """,
        )

    extract_features >> train >> [notify_log, maybe_email]
