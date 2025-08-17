# Dockerfile
FROM python:3.12.11-slim

# Install required packages
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install mlflow tensorflow pandas

# Copy the model into the container
COPY . /model
WORKDIR /model

# Expose port for MLflow model serving
EXPOSE 8080

# Run the MLflow model server
CMD ["mlflow", "models", "serve", "-m", ".", "-h", "0.0.0.0", "-p", "8080","--no-conda"]
