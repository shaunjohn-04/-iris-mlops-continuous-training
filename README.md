# Iris MLOps – Continuous Training & Drift Monitoring

An end-to-end MLOps project that demonstrates machine learning model training, experiment tracking, model registration, data drift detection, automated retraining, API deployment, and Docker containerization using the Iris dataset.

## Project Overview

This project implements an Iris flower classification system using a Random Forest machine learning model.

It uses MLflow to track experiments and manage model versions, Evidently and statistical testing to detect data drift, FastAPI to serve predictions, and Docker to package the application for deployment.

When data drift is detected, the retraining pipeline automatically trains a new model, evaluates its accuracy, and registers it in MLflow if it meets the defined accuracy threshold.

## Key Features

- **Model Training:** Random Forest classifier trained on the Iris dataset.
- **Experiment Tracking:** MLflow tracks model parameters, metrics, and training runs.
- **Model Registry:** Registers trained models and manages the `champion` model alias.
- **Data Drift Detection:** Uses Evidently and the Kolmogorov-Smirnov (KS) statistical test.
- **Automated Retraining:** Triggers model retraining when significant data drift is detected.
- **Model Evaluation:** Checks model accuracy against a predefined threshold of 90%.
- **REST API:** FastAPI provides health-check and prediction endpoints.
- **Docker Deployment:** Packages the application and trained model into a Docker image.
- **Automated Testing:** Pytest validates API endpoints, drift detection, and retraining functionality.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, Random Forest |
| Experiment Tracking | MLflow |
| Data Drift Monitoring | Evidently, SciPy |
| API Framework | FastAPI |
| Containerization | Docker |
| Testing | Pytest |
| Dataset | Iris Dataset |
| Version Control | Git, GitHub |

## Project Structure

```text
Mlops-demo/
│
├── model/
│   ├── MLmodel
│   ├── conda.yaml
│   ├── model.skops
│   ├── python_env.yaml
│   └── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── monitor_drift.py
│   └── retrain_pipeline.py
│
├── tests/
│   ├── test_app.py
│   └── test_retrain.py
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── app.py
├── train.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## Project Workflow

1. Train the initial Random Forest model using the Iris dataset.
2. Track training experiments and metrics using MLflow.
3. Register the trained model and assign the `champion` alias.
4. Monitor incoming data for statistical drift.
5. Generate an HTML drift report using Evidently.
6. Use the KS test to identify drifted features.
7. Trigger the retraining pipeline when the drift threshold is reached.
8. Train and evaluate a new Random Forest model.
9. Register the new model if its accuracy is at least 90%.
10. Serve predictions through the FastAPI application.
11. Package the application and model using Docker.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Mlops-demo
```

### 2. Create a Virtual Environment

```bash
python -m venv v1
```

Activate it on Windows:

```powershell
.\v1\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Initial Model

```bash
python train.py
```

This trains the initial model and registers it with MLflow.

### 5. Run the Drift Monitoring Script

```bash
python src/monitor_drift.py
```

This checks for data drift and generates:

```text
drift_report.html
```

Open the HTML report in your browser to inspect the drift results.

### 6. Run the Continuous Retraining Pipeline

```bash
python -m src.retrain_pipeline
```

The pipeline checks for data drift and triggers retraining when drift is detected.

### 7. Start the FastAPI Application

```bash
uvicorn app:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API health and model availability |
| POST | `/predict` | Predicts the Iris flower class |

### Sample Prediction Request

Endpoint:

```text
POST /predict
```

JSON request:

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

Sample response:

```json
{
    "prediction": 0
}
```

## Docker Deployment

### Build the Docker Image

```bash
docker build -t iris-mlops:v1 .
```

### Run the Container

```bash
docker run -d -p 8000:8000 --name iris-mlops-container iris-mlops:v1
```

### Check Running Containers

```bash
docker ps
```

### Test the API

```text
http://localhost:8000/docs
```

### Docker Hub Image

The Docker image is published on Docker Hub.

**Image:** `shaunjohnmathew/iris-mlops:v1`

**Docker Hub Repository:** https://hub.docker.com/r/shaunjohnmathew/iris-mlops

Pull the image:

```bash
docker pull shaunjohnmathew/iris-mlops:v1
```

## Testing

The project uses Pytest to validate its functionality.

Run the test suite:

```bash
pytest -v
```

Current local test results:

```text
6 passed
```

The tests cover:

- API health-check functionality.
- Valid prediction requests.
- Invalid input validation.
- Data drift detection.
- Model training and registration.
- Automatic retraining pipeline execution.

## Future Enhancements

- Integrate Prometheus and Grafana for model and application monitoring.
- Implement CI/CD using GitHub Actions.
- Deploy the application to AWS.
- Integrate real-time production data monitoring.
- Add model performance tracking and automated alerts.

## Author

**Shaun John Mathew**

GitHub: https://github.com/shaunjohn-04

---

*This project demonstrates an end-to-end MLOps workflow combining machine learning, experiment tracking, drift monitoring, automated retraining, API development, and containerization.*
