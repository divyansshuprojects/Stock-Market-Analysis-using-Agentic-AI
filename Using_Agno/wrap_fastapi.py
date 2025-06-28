# Import necessary modules
import os
import mlflow
import uvicorn
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Any
import json

# Initialize FastAPI application
app = FastAPI(title="MLflow Model Prediction API")

# Configure MLflow and Minio settings
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"  # Minio endpoint
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"  # Minio username
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"  # Minio password

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Create a class to handle the model
class MLflowModel:
    def __init__(self, model_name, model_version):
        """
        Initialize the MLflow model
        model_name: Name of the registered model in MLflow
        model_version: Version of the model to load
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the model from MLflow"""
        try:
            model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{self.model_name}/{self.model_version}"
            )
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, data):
        """
        Make predictions using the loaded model
        data: Pandas DataFrame to perform predictions on
        """
        predictions = self.model.predict(data)
        return predictions.tolist() if hasattr(predictions, 'tolist') else predictions

# Initialize the model
model = MLflowModel(model_name="sample_model_ml", model_version="1")

@app.get("/")
def root():
    """Root endpoint"""
    return {"message": "MLflow Model Prediction API"}

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Make predictions on data in a CSV file
    """
    # Check if the file is a CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Only CSV files are accepted."
        )
    
    try:
        # Create a temporary file to store the uploaded CSV
        with open(file.filename, "wb") as f:
            f.write(await file.read())
        
        # Read the CSV file
        data = pd.read_csv(file.filename)
        
        # Remove the temporary file
        os.remove(file.filename)
        
        # Make predictions
        predictions = model.predict(data)
        
        # Return the predictions
        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict/json")
async def predict_json(data: Dict[str, List[Any]] = Body(...)):
    """
    Make predictions on data sent as JSON
    Example input:
    {
        "open": [100.0],
        "high": [120.0],
        ...
    }
    """
    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Return the predictions
        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("wrap_fastapi:app", host="0.0.0.0", port=8000, reload=True)