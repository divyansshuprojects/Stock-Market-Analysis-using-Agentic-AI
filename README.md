# MLflow Sample Model Deployment for Trading Predictions

## Project Overview
This project demonstrates a comprehensive machine learning model deployment workflow for trading predictions using MLflow, featuring two model approaches:
- Human Learn (HL) Model: Rule-based trading strategy prediction
- Machine Learning (ML) Model: Ensemble-based trading prediction

## Key Features
- MLflow tracking and model registry
- Two model types: rule-based(human-learn) and machine learning
- Streamlit-based model testing and fine-tuning
- Performance metrics visualization
- Model versioning and alias management


## Deployment Steps

### 1. Start MLflow Tracking Server
```bash
mlflow server --host 127.0.0.1 --port 8080
```

### 2. Train Models
```bash
python training_models.py
```
- Trains Human Learn and Machine Learning models
- Logs models to MLflow tracking server
- Creates model registry entries

### 3. Register Models in MLflow UI
**Note:** Model registry via programmatic is going to be deprecated. Use registration via UI. We click on runs, then Click on register model then register with below names.


Model Names:
- Human Learn Model: `hl_trading_model`
- Machine Learning Model: `ml_trading_model`

### 4. Test Model Performance
```bash
streamlit run sample_models/test_and_register/tester.py
```
- Visualize actual vs predicted returns
- Compare model performance metrics

### 5. Fine-Tune and Manage Models
```bash
streamlit run sample_models/tune/tuning_streamlit.py
```
- Interactive model fine-tuning using stream lit app.
- Promote best-performing models to production
- Manage model versions and aliases

## Model Workflow
1. Data Preparation
2. Model Training
3. Performance Evaluation
4. Model Registration
5. Deployment and Inference
6. Continuous Fine-tuning

