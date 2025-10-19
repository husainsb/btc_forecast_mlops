# Bitcoin Price Forecasting with LSTM & MLOps

This project demonstrates an **end-to-end MLOps pipeline** for forecasting Bitcoin prices using **LSTM (Long Short-Term Memory)** models, integrated with **MLflow**, **FastAPI**, **xDocker**, **Kubernetes**, and **CI/CD via GitHub Actions**.

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
│   ├── __main__.py                   # Entry point
│   ├── app.py                        # FastAPI app
│   ├── credentials.env               # API credentials
│   ├── data_prep.py                  # Data preprocessing
│   ├── fetch_data.py                 # Fetch data from CoinGecko
│   ├── locustfile.py                 # Load testing
│   ├── predict.py                    # Prediction logic
│   ├── train_model.py                # LSTM model training
│   └── train_pyfunc_model.py         # MLflow PyFunc model
├── k8s_folder/
│   ├── fastapi-app-deployment.yaml   # Kubernetes Deployment
│   └── fastapi-app-expose.yaml       # Kubernetes Service
├── Dockerfile                        # Containerization
├── MLmodel                           # MLflow model metadata
├── python_env.yaml                   # Conda environment
├── python_model.pkl                  # MLflow PyFunc model
└── requirements.txt                  # Python dependencies
```

---

### **⚙️ Workflow**
1. **Data Pipeline**
   - Fetch Bitcoin price data from CoinGecko API
   - Preprocess and scale data
2. **Model Training**
   - Train LSTM model using TensorFlow/Keras
   - Log model and metrics to MLflow
   - Register best model as **Champion**
3. **Deployment**
   - Build FastAPI app for inference
   - Containerize with Docker
   - Deploy on Kubernetes for scalability
4. **CI/CD**
   - GitHub Actions builds Docker image and pushes to Docker Hub
5. **Load Testing**
   - Stress test API using Locust
6. **Horizontal Scaling**
   - Kubernetes scales pods based on traffic

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
- Add **Hyperparameter Tuning** with Optuna

[![CI/CD Pipeline](https://github.com/husainsb/btc_forecast_mlops/actions/workflows/ci-cd.yml/badge.svg?branch=fastapi_app)](https://github.com/husainsb/btc_forecast_mlops/actions/workflows/ci-cd.yml)
