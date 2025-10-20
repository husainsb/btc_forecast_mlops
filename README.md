# Bitcoin Price Forecasting with LSTM & MLOps

This project demonstrates an **end-to-end MLOps pipeline** for forecasting Bitcoin prices using **LSTM (Long Short-Term Memory)** models, integrated with **MLflow**, **FastAPI**, **Docker**, **Kubernetes**, and **CI/CD via GitHub Actions**.

The project fetches the data in realtime, trains the model, logs & register it using MLflow, it then builds the Docker image for the inference as FastAPI App and pushes it to Dockerhub. This image can be pulled by anyone and can use it straight away either on local machine or on Kubernetes deployment. 

---

### **📌 Project Overview**
- **Data Source:** CoinGecko API (daily Bitcoin price data)
- **Model:** LSTM implemented using **TensorFlow/Keras**
- **Experiment Tracking & Model Registry:** **MLflow**
- **Deployment:** FastAPI app containerized with **Docker**, deployed on **Kubernetes**
- **Load Testing:** **Locust**
- **CI/CD:** GitHub Actions for automated Docker image build & push to Docker Hub
- **Scalability:** Horizontal scaling via Kubernetes based on traffic

---

### **📂 Project Structure**
```
project/
│
├── .github/workflows/ci-cd.yml       # GitHub Actions pipeline
├── artifacts/                        # Trained model & scalers
│   ├── model.keras
│   ├── scaler_x.pkl
│   └── scaler_y.pkl
├── codes/
│   ├── __main__.py                   # Entry point & orchestration of entire pipeline
│   ├── app.py                        # FastAPI app deployment
│   ├── credentials.env               # API credentials & MySQL DB connection
│   ├── data_prep.py                  # Data preprocessing
│   ├── fetch_data.py                 # Fetch data from CoinGecko
│   ├── locustfile.py                 # Load testing
│   ├── predict.py                    # Prediction logic
│   ├── train_model.py                # LSTM model training
│   └── train_pyfunc_model.py         # Logging of MLflow PyFunc model(Champion)
├── k8s_folder/
│   ├── fastapi-app-deployment.yaml   # Kubernetes Deployment
│   └── fastapi-app-expose.yaml       # Kubernetes Service
├── Dockerfile                        # Containerization via Docker
├── MLmodel                           # MLflow model metadata
├── python_env.yaml                   # Virtual environment
├── python_model.pkl                  # MLflow PyFunc model
└── requirements.txt                  # Python dependencies
```

---

### **⚙️ Workflow**
1. **Data Pipeline**
   - Fetch Bitcoin price data from CoinGecko API and park it in MySQL DB
   - Preprocess and scale data
2. **Model Training**
   - Train LSTM model using TensorFlow/Keras
   - Log model and metrics to MLflow
   - Convert the base model to Pyfunc model with dependencies like data pre-processing logic and scaling
   - Register this Pyfunc model as **Champion**
3. **Deployment**
   - Build FastAPI app for inference
   - Containerize it with Docker
   - Deploy on local Kubernetes for scalability
4. **CI/CD**
   - GitHub Actions builds Docker image and pushes to Docker Hub. The architecture is 'arm64' as my machine is MacOS
   - Pull the image from Dockerhub and instantiate a K8s deployment & service on top of this
     [![CI/CD Pipeline](https://github.com/husainsb/btc_forecast_mlops/actions/workflows/ci-cd.yml/badge.svg?branch=fastapi_app)](https://github.com/husainsb/btc_forecast_mlops/actions/workflows/ci-cd.yml)
5. **Load Testing**
   - Inference will be done using Predict method of model
   - Stress test the App using Locust. E.g: Peak concurrency of 1000 requests(users).
   - Obeserve the Load Balancing of K8s service and its impact on API response time & Requests per Second (RPS)
6. **Horizontal Scaling w/o Downtime**
   - Kubernetes can scale pods w/o causing the downtime and traffic will be routed using built-in Load Balancer

---

### **🚀 How to Run**
#### **Local**
#### **Pull the image from Dockerhub & Run the container**
```bash
# This image is built for arm64 or MacOS version
docker pull husainsb/fastapi-mlflow:fastapi_app
docker run -p 5001:8000 husainsb/fastapi-mlflow:fastapi_app
```

#### **Kubernetes Deployment & Serving**
```bash
cd k8s_folder/   # point to directory containing K8s related files
kubectl apply -f k8s_folder/fastapi-app-deployment.yaml
kubectl apply -f k8s_folder/fastapi-app-expose.yaml

# check whether Pods and services are up
kubectl get events
kubectl get pods
kubectl get deployment
kubectl config view

# horizontally scale the app to include 5 pods
kubectl scale --replicas=5 deployment/fastapi-deployment

# delete service & deployment to free up resources
kubectl delete service fastapi-service  
kubectl delete deployment fastapi-deployment 
```
![K8s Deployment](/images/k8s_deployment.png "Kubernetes Deployment")

#### **Stress testing of App**
```bash
source .venv/bin/activate   # replace with your Python env directory
cd codes/  # path to directory of locustfile.py
locust
# This will initiate Locust app on localhost:8089 by default
# Set the parameters like URL, No. of peak users, Ramp-up users etc to test the App
```
![Locust Initialization](/images/locust_intialization.png "Initial parameters for Stress testing")
![Stress Test](/images/locust_stats.png "Statistics related to Stress testing")
![Performance Charts](/images/locust_charts.png "Performance Charts while Stress testing")
---

### **✅ Features**
✔ Automated Data Fetching  
✔ LSTM Model Training & MLflow Tracking  
✔ FastAPI Deployment with Docker & Kubernetes  
✔ CI/CD with GitHub Actions  
✔ Load Testing with Locust  
✔ Horizontal Scaling via K8s

---

### **📈 Future Enhancements**
- Provision to detect model/data/performance drift
- Add **Prometheus + Grafana** for monitoring

