# RiskForge — Credit Default Risk Prediction System
## Overview

This project is an end-to-end machine learning system built to predict the probability of loan default using structured financial data.

Instead of focusing only on model accuracy, the goal of this project was to build a complete production-oriented workflow starting from raw relational datasets all the way to deployment and inference.

The project includes:

* data analysis
* feature engineering
* model training and evaluation
* explainability using SHAP
* FastAPI-based deployment
* Docker support
* deployment and monitoring design

The final system can take applicant information as input and return:

* default probability
* risk category
* final approval recommendation

---

## Problem Statement

Financial institutions deal with significant losses when high-risk applicants are approved without proper risk assessment.

The goal of this project was to build a machine learning system capable of estimating the likelihood of default so that lending decisions can become:

* faster
* more consistent
* more scalable
* more data-driven

The project focuses on balancing business trade-offs between:

* approving risky applicants
* rejecting good customers

---

## Dataset

Dataset used:
Home Credit Default Risk Dataset

Dataset size:

* ~300K loan applicants
* multiple relational tables

Target variable:

* TARGET = 1 → customer defaulted
* TARGET = 0 → customer did not default

Class imbalance:

* around 8% defaults

The dataset contains:

* applicant demographics
* employment details
* credit bureau history
* previous applications
* installment payment records

Because the data is relational, significant aggregation and feature engineering were required before modeling.

---

## Dataset Access

The original dataset is not included in this repository due to size limitations.

Dataset source:
Home Credit Default Risk
[https://www.kaggle.com/competitions/home-credit-default-risk](https://www.kaggle.com/competitions/home-credit-default-risk)

After downloading, place the files inside:

```text
data/raw/
```

Required files:

* `application_train.csv`
* `bureau.csv`
* `previous_application.csv`
* `installments_payments.csv`

Additional files can also be added for extended feature engineering.

---

## Feature Engineering

A major part of the project focused on transforming raw relational data into meaningful customer-level features.

Feature engineering included:

* aggregating bureau history
* previous loan behavior
* installment payment trends
* repayment delays
* approval/refusal ratios
* credit utilization patterns

Several behavioral indicators were created to better capture repayment risk.

Final dataset:

* approximately 164 engineered features

Missing values were handled using:

* numerical defaults
* categorical fallback values

Basic outlier reduction was also applied using percentile capping.

---

## Tech Stack

Languages & Libraries:

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* XGBoost
* SHAP
* FastAPI
* Joblib
* Docker

---

## Project Structure

```text
riskforge-credit-ml/
│
├── notebooks/      # EDA, feature engineering, modeling
├── src/            # reusable ML/inference utilities
├── deployment/     # FastAPI + Docker deployment
├── reports/        # executive summary and deployment design
├── models/         # runtime model directory
├── data/           # raw/interim/processed data
├── requirements.txt
└── README.md
```

---

## Models Evaluated

Multiple models were trained and compared during experimentation:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

The objective was not only to maximize ROC-AUC, but also to:

* reduce overfitting
* maintain interpretability
* ensure deployment feasibility

---

## Final Model

Selected Model:
LightGBM

Why LightGBM was selected:

* best validation ROC-AUC performance
* stable train vs validation behavior
* strong performance on tabular financial data
* faster inference compared to heavier ensemble models

Final validation ROC-AUC:
~0.778

---

## Model Evaluation

Evaluation included:

* ROC-AUC analysis
* Precision-Recall analysis
* confusion matrix evaluation
* threshold tuning

Threshold tuning was important because business decisions depend heavily on:

* false positives
* false negatives
* risk tolerance

The project explored different threshold strategies for:

* approval
* review
* rejection

---

## Explainability

Model explainability was implemented using SHAP.

This helped identify the most influential drivers behind model predictions.

Important features included:

* EXT_SOURCE risk scores
* payment delay behavior
* credit utilization
* employment stability
* income-related variables

Both:

* global explanations
* local prediction explanations

were analyzed.

The explainability stage was important to ensure the system remains interpretable and suitable for financial decision support.

---

## Inference Pipeline

A reusable inference pipeline was created for prediction serving.

Pipeline flow:
Input → preprocessing → model → probability → decision

The inference system:

* aligns incoming features automatically
* handles missing inputs
* applies preprocessing consistently
* generates probability scores
* maps probabilities into risk bands

Output includes:

* default probability
* Low / Medium / High risk category
* final decision:

  * Approve
  * Review
  * Reject

---

## API Deployment

A FastAPI application was built to expose the model as a real-time prediction service.

Features:

* prediction endpoint
* interactive Swagger documentation
* reusable inference pipeline
* deployment-ready structure

Local API execution:
`uvicorn deployment.app:app --reload`

Swagger API docs:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Docker Deployment

Docker support was added to make the system portable and deployment-ready.

The deployment setup includes:

* Dockerfile
* .dockerignore
* isolated deployment dependencies

Docker build:
docker build -t riskforge-api ./deployment

Docker run:
docker run -p 8000:8000 riskforge-api

---

## Model Serving Strategy

Since the trained model exceeded GitHub upload limits, the model is loaded dynamically during application startup.

This approach:

* keeps the repository lightweight
* avoids large binary uploads
* supports scalable deployment

The model is downloaded automatically and cached locally before inference begins.

---

## Monitoring & Deployment Design

The deployment design also considers production-oriented concerns such as:

* batch scoring
* drift monitoring
* retraining triggers
* monitoring strategy
* deployment scalability

The system design includes:

* monitoring prediction distributions
* tracking feature drift
* identifying model degradation over time

Potential future monitoring tools:

* Evidently AI
* Prometheus
* Grafana

---

## Example Prediction

Example API response:

{
"probability": 0.71,
"risk_band": "High Risk",
"decision": "Reject"
}

---

## Limitations

Current limitations include:

* no strict temporal validation
* limited real-time monitoring
* simple missing value handling
* no probability calibration
* limited input validation

These limitations were documented intentionally to keep the project realistic and transparent.

---

## Future Improvements

Potential future improvements:

* Pydantic-based schema validation
* automated retraining pipeline
* drift detection integration
* CI/CD pipeline
* authentication & rate limiting
* cloud deployment on AWS/Render
* model calibration
* stronger feature selection

---

## Key Takeaways

This project helped strengthen understanding of:

* end-to-end ML workflows
* relational feature engineering
* model evaluation and thresholding
* explainable AI
* deployment using FastAPI
* Docker-based deployment
* production-oriented ML system design
