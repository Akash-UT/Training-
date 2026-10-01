# ML-Based Apparel Demand Forecasting

## Project Objective

Build a machine learning system to forecast apparel sales demand using historical monthly sales data.

## Dataset

The dataset contains monthly sales demand for multiple apparel product types.

## Product Types

- BOOTS
- CARDIGAN
- JACKET
- JEANS
- SANDALS
- SHIRT
- SNEAKERS
- SWEATER
- TOP
- VEST

## Pipeline

The project follows:

1. Data Cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Chronological Train/Test Split
5. Categorical Encoding
6. Model Training
7. Model Evaluation
8. Model Comparison
9. Automatic Model Selection
10. Final Model Saving
11. Prediction
12. Model Interpretation

## Feature Engineering

The forecasting model uses:

- Year
- Month
- Quarter
- Month Sin
- Month Cos
- Time Index
- Lag 1
- Lag 2
- Lag 3
- Lag 6
- Lag 12
- Rolling Mean 3
- Rolling Mean 6
- Rolling Mean 12
- Rolling Standard Deviation 3
- Rolling Standard Deviation 6
- Year-over-Year Change
- Product Type

## Models

The following configurations are compared:

1. Random Forest - Original Target
2. XGBoost - Original Target
3. CatBoost - Original Target
4. XGBoost - Log Target
5. CatBoost - Log Target

## Evaluation Metrics

- MAE
- RMSE
- R²
- MAPE
- WAPE

Generalization is evaluated using:

- Train/Test MAE Ratio
- Train/Test RMSE Ratio
- Train/Test R² Gap

## Model Selection

Model selection combines:

- Predictive performance
- Generalization performance

The selection score uses:

- 60% predictive performance
- 40% generalization performance

## Saved Model Artifacts

After model experiments, the selected model is saved as:

models/final_model.pkl

The preprocessing pipeline is saved as:

models/preprocessor.pkl

Feature names are saved as:

models/feature_names.pkl

Model information and metrics are saved as:

models/model_metadata.json

## Running the Project

First run:

python experiments/model_experiments.py

Then run:

python main.py