# Scalable Smart Home Energy Prediction API

This project demonstrates the end-to-end development and deployment of a machine learning regression model that predicts household appliance energy consumption using environmental sensor data and engineered time features. The solution includes model training, preprocessing pipeline design, local API implementation, Docker containerization, and deployment to Google Cloud Platform (GCP) using Cloud Run.

The final system is serverless, scalable, secure, and production-ready.

---

# Live Deployment

Health Check:
https://energy-api-117171702986.northamerica-northeast2.run.app/health

Swagger UI:
https://energy-api-117171702986.northamerica-northeast2.run.app/docs

for health check: Expand /health  >> Click Try it out  >> Execute >> response appears

for prediction: Expand /predict >> Paste API Key:energy-2026-secure-key
Click Try it out  >> Paste request body in JSON format >> Execute >> response appears

API Key:
energy-2026-secure-key

Note: In real-world production systems, API keys should never be exposed publicly. It is included here only for assignment testing purposes.
Request body example:
{
  "date": "2016-01-11 20:00:00",
  "lights": 10,
  "T1": 19.89,
  "RH_1": 47.6,
  "T2": 19.2,
  "RH_2": 44.79,
  "T3": 19.79,
  "RH_3": 44.73,
  "T4": 19.0,
  "RH_4": 45.57,
  "T5": 17.17,
  "RH_5": 55.2,
  "RH_6": 39.0,
  "T7": 17.2,
  "RH_7": 41.0,
  "T8": 18.2,
  "RH_8": 48.0,
  "T9": 17.0,
  "RH_9": 45.0,
  "T_out": 6.6,
  "Press_mm_hg": 733.5,
  "RH_out": 92.0,
  "Windspeed": 7.0,
  "Visibility": 63.0,
  "Tdewpoint": 5.3
}

# Project Overview

Goal:
Predict appliance energy consumption (Wh) using indoor and outdoor environmental measurements and time-based features.

Final Model:
Random Forest Regressor integrated within a Scikit-learn Pipeline.

Cloud Platform:
Google Cloud Platform (GCP)

Services Used:
- Cloud Run (serverless container hosting)
- Cloud Storage (model artifact storage)
- Cloud Build (CI/CD)
- Artifact Registry (container image storage)


# How the System Works

1. User sends HTTPS request with API key.
2. Cloud Run hosts a Docker container running FastAPI.
3. At container startup:
   - The trained model artifact is downloaded from Google Cloud Storage.
   - The model is loaded into memory once.
4. For each prediction request:
   - The API computes time-based features from the provided date.
   - The preprocessing pipeline is applied.
   - The Random Forest model generates a prediction.
   - JSON response is returned.


# API Usage

## Health Check

GET /health

Example:

curl https://energy-api-117171702986.northamerica-northeast2.run.app/health

Response:
{"status": "ok"}

---

## Prediction Request

POST /predict

Header:
x-api-key: energy-2026-secure-key

Example curl request:

curl -X POST "https://energy-api-117171702986.northamerica-northeast2.run.app/predict" \
-H "Content-Type: application/json" \
-H "x-api-key: energy-2026-secure-key" \
-d '{
  "date": "2016-01-11 17:00:00",
  "lights": 10,
  "T1": 19.89,
  "RH_1": 47.6,
  "T2": 19.2,
  "RH_2": 44.79,
  "T3": 19.79,
  "RH_3": 44.73,
  "T4": 19.0,
  "RH_4": 45.57,
  "T5": 17.17,
  "RH_5": 55.2,
  "RH_6": 39.0,
  "T7": 17.2,
  "RH_7": 41.0,
  "T8": 18.2,
  "RH_8": 48.0,
  "T9": 17.0,
  "RH_9": 45.0,
  "T_out": 6.6,
  "Press_mm_hg": 733.5,
  "RH_out": 92.0,
  "Windspeed": 7.0,
  "Visibility": 63.0,
  "Tdewpoint": 5.3
}'

Example response:

{
  "prediction": 100.2
}


# Local Setup Instructions

## 1. Clone Repository

git clone https://github.com/fatemehabedin2/AIG-Capstone-model-deployment.git
cd AIG-Capstone-model-deployment


## 2. Create Virtual Environment

python -m venv .venv

Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate


## 3. Install Dependencies

pip install -r requirements.txt


## 4. Set Environment Variables

Create a .env file or export variables:

API_KEY=energy-2026-secure-key
GCS_BUCKET_NAME=energy-model-folio-451800-p5-4821
MODEL_BLOB_NAME=energy_rf_pipeline.joblib


## 5. Run Locally

uvicorn app.main:app --host 0.0.0.0 --port 8000

Open:
http://127.0.0.1:8000/docs


# Docker Local Test

Build image:

docker build -t energy-api:1.0 .

Run container:

docker run -p 8000:8080 \
-e API_KEY="energy-2026-secure-key" \
-e GCS_BUCKET_NAME="energy-model-folio-451800-p5-4821" \
-e MODEL_BLOB_NAME="energy_rf_pipeline.joblib" \
energy-api:1.0


# Deployment Configuration (GCP)

- Region: northamerica-northeast2 (Toronto)
- Memory: 1 GiB
- Minimum instances: 0
- Billing: Request-based
- Public access: Enabled
- Model stored in private Cloud Storage bucket
- Lifecycle rule enabled for cost control


# Model Development Summary

- Dataset: Appliances Energy Prediction (Kaggle)
- Removed redundant feature: T6
- Removed irrelevant features: rv1, rv2
- Engineered time features from date column
- Stratified train-test split using quantile binning
- Final Model: Random Forest Regressor
- Serialized pipeline using joblib
- Model artifact stored in GCS (not GitHub)


# Security Measures

- API key authentication required
- No retraining during inference
- Model loaded once at startup
- Secrets stored via environment variables
- Model artifact not committed to GitHub



# Author

Bibi F. Abidin Do  
Machine Learning Model Deployment Assignment  
