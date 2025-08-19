# Satyamohanproject
youtoube link: https://youtu.be/YDWrS3beClQ
Cloud Composer ML Pipeline – London Bike Duration Prediction
📄 Overview

This project implements an automated ML pipeline using Google Cloud Composer. The pipeline:

Extracts data from BigQuery (bigquery-public-data.london_bicycles.cycle_hire)

Trains a Linear Regression model to predict bike trip duration

Logs completion metrics to Cloud Logging and optionally sends an email notification

Data and artifacts are persisted for reproducibility and monitoring.

🛠️ Environment Setup

Cloud Composer 2 Environment: Configured with necessary permissions

APIs Enabled:

BigQuery API

Cloud Composer API

Service Account Roles:

BigQuery Job User

Storage Object Admin

Python Packages Installed: pandas, scikit-learn, joblib, google-cloud-storage, google-cloud-bigquery

📊 Data & Model

Target Variable: trip_duration_min (bike trip duration in minutes)

Features:

hour_of_day

day_of_week

month

is_weekend

Model Type: Linear Regression (scikit-learn)

Data Filter: Only trips between 1 minute and 2 hours

SQL Query Used for Feature Extraction:

CREATE OR REPLACE TABLE `{{ params.project_id }}.{{ params.bq_dataset }}.{{ params.bq_table }}` AS
SELECT
  rental_id,
  start_date,
  duration AS duration_sec,
  duration / 60.0 AS trip_duration_min,
  EXTRACT(HOUR FROM start_date) AS hour_of_day,
  EXTRACT(DAYOFWEEK FROM start_date) AS day_of_week,
  EXTRACT(MONTH FROM start_date) AS month,
  IF(EXTRACT(DAYOFWEEK FROM start_date) IN (1,7), 1, 0) AS is_weekend
FROM `bigquery-public-data.london_bicycles.cycle_hire`
WHERE duration BETWEEN 60 AND 7200;
⚙️ DAG Structure
Task 1: extract_features

Runs the above SQL query in BigQuery

Saves results to a project dataset table (ml_pipeline_demo.cycle_hire_features)

Task 2: train_model

Reads data from BigQuery table

Trains Linear Regression on selected features

Evaluates metrics: MAE, RMSE, R2
