<div align="center">
  <img src="banner.svg" alt="Bulldozer Price Prediction Banner" width="100%"/>
</div>

<br/>

<div align="center">

  [![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

  <br/>

  ![RMSLE](https://img.shields.io/badge/RMSLE-%3C%200.25-FFD200?style=flat-square)
  ![Records](https://img.shields.io/badge/Dataset-400K%2B%20Records-F7971E?style=flat-square)
  ![Source](https://img.shields.io/badge/Source-Kaggle%20Bluebook-20BEFF?style=flat-square&logo=kaggle)
  ![Author](https://img.shields.io/badge/Author-jangasashi-1a1200?style=flat-square)

</div>

<br/>

## What is This?

A **machine learning regression model** that predicts the future auction sale price of heavy construction equipment (bulldozers) using historical sale data from Kaggle's Bluebook for Bulldozers competition.

This isn't just a notebook exercise — I built a **production-ready scikit-learn pipeline** with preprocessing, feature engineering, and hyperparameter tuning via RandomizedSearchCV, achieving **RMSLE < 0.25** on the validation set. The pipeline is structured so it can be plugged directly into a pricing dashboard or API.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Feature Engineering](#feature-engineering)
- [Model & Results](#model--results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Author](#author)

---

## Problem Statement

> *"Given historical auction records of bulldozers sold at auction, predict the future sale price of a bulldozer."*

This is a **time-series regression problem** — you're predicting a continuous value (price in USD), and the evaluation metric is **Root Mean Squared Logarithmic Error (RMSLE)** — chosen because price errors matter proportionally, not absolutely (being off by $5K on a $10K machine is worse than being off by $5K on a $100K machine).

---

## Dataset

| Property | Detail |
|---|---|
| **Competition** | Kaggle — Bluebook for Bulldozers |
| **Records** | 400,000+ auction sale records |
| **Time Range** | Historical auction data spanning multiple years |
| **Target Variable** | `SalePrice` — auction sale price in USD |
| **Metric** | RMSLE (Root Mean Squared Logarithmic Error) |

The data includes machine specifications (model, year made, equipment series), sale metadata (date, location, auctioneer), and usage details — making feature engineering a significant part of the challenge.

---

## Approach

```
Raw Auction Data (400K+ rows)
         │
         ▼
  Exploratory Data Analysis
  (distributions, nulls, outliers)
         │
         ▼
  Feature Engineering
  (machine age, date decomposition,
   ordinal encoding, null imputation)
         │
         ▼
  Scikit-Learn Pipeline
  (ColumnTransformer → XGBoost)
         │
         ▼
  RandomizedSearchCV
  (hyperparameter optimization)
         │
         ▼
  Validation: RMSLE < 0.25
         │
         ▼
  Deployment-Ready Pipeline
  (saved via Joblib)
```

---

## Feature Engineering

Key transformations that improved model performance:

| Transformation | Description |
|---|---|
| **Machine Age** | `YearMade` subtracted from `saleYear` — age of equipment at time of sale |
| **Date Decomposition** | `saledate` split into `saleYear`, `saleMonth`, `saleDay`, `saleDayOfWeek` |
| **Ordinal Encoding** | Size/condition categorical columns converted to ordered numerics |
| **Null Imputation** | Strategic fill strategies: median for numerics, "Unknown" for categoricals |
| **Rare Category Grouping** | Low-frequency categories collapsed into "Other" to reduce noise |

---

## Model & Results

**Model:** XGBoost Regressor inside a scikit-learn `Pipeline` with `ColumnTransformer` preprocessing

**Tuned via:** `RandomizedSearchCV` with cross-validation

| Metric | Score |
|---|---|
| **RMSLE (Validation)** | **< 0.25** |
| Training Strategy | Time-based train/validation split (no data leakage) |
| Hyperparameters | Tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample` |

> Why RMSLE and not RMSE? Because the target (`SalePrice`) spans a wide range — from cheap small equipment to expensive heavy machinery. RMSLE penalizes errors proportionally, making it a fairer measure for skewed price distributions.

---

## Tech Stack

<div align="center">

| Library | Role |
|---|---|
| `XGBoost` | Primary regression model — gradient boosted trees |
| `scikit-learn` | Pipeline, ColumnTransformer, RandomizedSearchCV |
| `pandas` | Data loading, date parsing, feature engineering |
| `numpy` | Numerical operations, log transformations |
| `matplotlib` & `seaborn` | EDA visualizations, feature importance plots |
| `joblib` | Serialize the trained pipeline for deployment |

</div>

---

## Project Structure

```
bulldozer-price-regression-model/
│
├── bulldozer-price-regreession-model.ipynb   # Full notebook: EDA → Pipeline → Evaluation
└── README.md
```

> Note: The dataset is not included in the repo (Kaggle terms). Download it from the [Bluebook for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers) competition page and place in the working directory.

---

## Getting Started

### 1. Install Dependencies
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn joblib jupyter
```

### 2. Download Data
Download from [Kaggle Bluebook for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data):
- `TrainAndValid.csv` — training data
- `Test.csv` — test set

### 3. Run the Notebook
```bash
git clone https://github.com/jangasashi/bulldozer-price-regreession-model.git
cd bulldozer-price-regreession-model
jupyter notebook bulldozer-price-regreession-model.ipynb
```

### 4. Load the Saved Pipeline (if exported)
```python
import joblib

pipeline = joblib.load('bulldozer_price_pipeline.pkl')
prediction = pipeline.predict(new_data)
print(f"Predicted Sale Price: ${prediction[0]:,.2f}")
```

---

## Author

<div align="center">

**Janga Sashi Harikrishna**

[![GitHub](https://img.shields.io/badge/GitHub-jangasashi-181717?style=for-the-badge&logo=github)](https://github.com/jangasashi)
[![Kaggle](https://img.shields.io/badge/Kaggle-Bluebook%20Challenge-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/c/bluebook-for-bulldozers)
[![Email](https://img.shields.io/badge/Email-shashikrish.143%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:shashikrish.143@gmail.com)

</div>

---

<div align="center">
  <sub>Built to understand how structured ML pipelines scale to real-world regression problems. Drop a ⭐ if this was useful!</sub>
</div>
