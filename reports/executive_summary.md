# Credit Default Risk — Executive Summary

## 1. Business Problem

Predict the probability of a loan applicant will default.

Goal: Reduce financial losses by improving approval decisions and risk segmentation.

---

## 2. Data Used
- Source: Home Credit Dataset
- Size: ~300K applicants
- Features: ~164 engineered features
- Data includes:
    - Applicant details (income, employment, demographics)
    - Credit bureau history
    - Previous loan behavior
    - installment payment behavior

Target:

- TARGET = 1 → default
- TARGET = 0 → non-default

Class imbalance: ~8% defaults

---

## 3. Feature Engineering
- Aggregated bureau, previous application, and installment data per customer
- Created behavioral features:
    - Payment delay
    - Credit utilization
    - Approval/refusal ratio
- Handled missing values:
    - Numerical → 0
    - Categorical → "Unknown"
- Applied light percentile capping to reduce extreme outliers

---

## 4. Models Evaluated
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

---

## 5. Best Model
**LightGBM**

Reason:
- Best validation performance
- Handles tabular + imbalanced data effectively
- Lower overfitting compared to Random Forest

---

## 6. Performance
- ROC-AUC: ~0.778
- Stable train vs validation performance
- Good ranking ability for high-risk customers

Threshold tuning:

- Default(0.5) not optimal
- Best F1 threshold 0.65

---

## 7. Key Insights
- External risk scores(`EXT_SOURCE_*`) are strongest predictors
- Payment delays and credit utilization strongly increase default risk
- Higher income and stable employment reduce risk

---

## 8. Trade-offs
- Higher recall → more false positives (rejecting good customers)
- Higher precision → more false negatives (approving risky customers)

Business decision depends on:
- Risk tolerance
- Cost of default vs lost opportunity

---

## 9. Inference & Deployment
- Built end-to-end pipeline:
    - Input → preprocessing → model → probability → decision
- Deployment features:
    - FastAPI based prediction service
    - Runtime model loading using `joblib`
    - Schema-aligned preprocessing
    - Risk band classification:
        - Low Risk
        - Medium Risk
        - High Risk

---

## 10. Limitations
- No strict temporal filtering (possible data leakage risk)
- Missing values filled with simple defaults
- No real-time feature validation
- Model probabilities are not calibrated

---

## 11. Future Improvements
- Add temporal validation
- Improve feature selection
- Add input validation (Pydantic)
- Deploy with monitoring and logging
- Add drift detection and monitoring
- Add Pydantic-based input validation
- Automate retraining pipeline
---

## 12. Business Impact
- Enables automated risk scoring
- Improves loan approval consistency
- Reduce default exposure
- Supports scalable decision making
- Provides interpretable and production-ready ML workflow

