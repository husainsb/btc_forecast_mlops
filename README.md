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
     [![CI/CD Pipeline](https://github.com/husainsb/btc_forecast_mlops/actions/workflows/ci-cd.yml/badge.svg?branch=fastapi_app)](https://github.com/husainsb/btc_forecast_mlops/actions/workflows/ci-cd.yml)
5. **Load Testing**
   - Inference will be done using Predict method of model
   - Stress test the App using Locust. Peak concurrency of 1000 requests(users).
   - Obeserve the Load Balancing of K8s service and its impact on API response time & Requests per Second (RPS)
6. **Horizontal Scaling w/o Downtime**
   - Kubernetes scales pods based on traffic and w/o causing the downtime

---

### **🚀 How to Run**
#### **Local**
```bash
# Clone repo
git clone <repo-url>
cd project

# Install dependencies
pip install -r requirements.txt

# Run FastAPI app
uvicorn codes.app:app --reload
```

#### **Docker**
```bash
docker build -t <your-dockerhub-username>/bitcoin-forecast .
docker run -p 8000:8000 <your-dockerhub-username>/bitcoin-forecast
```

#### **Kubernetes**
```bash
kubectl apply -f k8s_folder/fastapi-app-deployment.yaml
kubectl apply -f k8s_folder/fastapi-app-expose.yaml
```

---

### **✅ Features**
✔ Automated Data Fetching  
✔ LSTM Model Training & MLflow Tracking  
✔ FastAPI Deployment with Docker & Kubernetes  
✔ CI/CD with GitHub Actions  
✔ Load Testing with Locust  
✔ Horizontal Scaling  

---

### **📈 Future Enhancements**
- Add **Prometheus + Grafana** for monitoring
- Implement **Airflow** for orchestration


