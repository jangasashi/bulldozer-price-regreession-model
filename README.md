Bulldozer Price Regression Model 🚜

This project predicts the future sale price of bulldozers using historical auction data from the Kaggle Bluebook for Bulldozers dataset. It applies machine learning and feature engineering techniques to build a robust regression model capable of supporting real-world auction pricing and business forecasting.

📌 Project Overview

Goal: Predict the sale price of bulldozers based on historical auction data and machine features.

Dataset: Kaggle Bluebook for Bulldozers
 with 400K+ records.

Problem Type: Regression / Time Series Forecasting.

Evaluation Metric: Root Mean Squared Logarithmic Error (RMSLE).

🛠️ Tools & Technologies

Programming & Libraries: Python, Pandas, NumPy, Scikit-learn, XGBoost

Data Visualization: Matplotlib, Seaborn

Model Deployment: Joblib for saving trained models and preprocessing pipelines

Workflow: Scikit-learn Pipelines, ColumnTransformer

🔑 Key Steps

Data Preprocessing

Handled missing values with imputation strategies.

Parsed date columns into saleYear, saleMonth.

Encoded categorical variables for model compatibility.

Feature Engineering

Created new variables such as machineAge to capture equipment depreciation.

Extracted time-based trends for auction seasonality.

Exploratory Data Analysis (EDA)

Visualized sales trends, correlations, and feature importance.

Identified key variables influencing bulldozer pricing.

Modeling & Optimization

Trained multiple regression models (Random Forest, XGBoost).

Tuned XGBoost using RandomizedSearchCV.

Achieved RMSLE < 0.25 on the test dataset.

Deployment-Ready Pipeline

Built an end-to-end pipeline with preprocessing + model training.

Exported with Joblib for integration into dashboards or APIs.

📊 Results & Insights

The optimized XGBoost model outperformed baseline models.

RMSLE score achieved: < 0.25 (competitive with Kaggle benchmarks).

Feature importance showed machine age and sale year as strong predictors of price.

🚀 How to Run

Clone the repository:

git clone https://github.com/jangasashi/bulldozer-price-regreession-model.git
cd bulldozer-price-regreession-model


Install dependencies:

pip install -r requirements.txt


Run the notebook or script to train and evaluate models.

📌 Future Improvements

Deploy model via Flask API or Streamlit app for interactive use.

Experiment with LSTM or Time-Series Deep Learning models for improved trend capture.

Automate feature engineering and hyperparameter tuning with MLflow or Optuna.

👤 Author

Janga Sashi Hari Krishna

Data Scientist | Python | SQL | Machine Learning | Power BI

LinkedIn : https://www.linkedin.com/in/sashi-hari-krishna-janga-96aa28223/
GitHub: https://github.com/jangasashi
