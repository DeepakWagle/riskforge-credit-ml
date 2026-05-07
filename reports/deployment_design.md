# Credit Default Risk — Deployment Design

## 1. Deployment Architecture

### System Flow

Client → FastAPI API → Preprocessing Pipeline → LightGBM Model → Prediction Response

### Components

* FastAPI inference service
* Preprocessing pipeline
* LightGBM prediction model
* Risk band & decision
* Runtime model loading

---

## 2. API Layer

### Endpoints

#### `/predict`

Accepts applicant data and returns:

* Default probability
* Risk band
* Final decision

#### `/health`

Used for service health monitoring.

---

## 3. Model Loading

### Strategy

* Model not stored in GitHub repository
* Downloaded dynamically at application startup
* Loaded using `joblib`

### Benefits

* Avoids repository size limitations
* Keeps deployment lightweight
* Supports scalable cloud deployment

---

## 4. Inference Pipeline

### Steps

1. Receive input request
2. Validate schema
3. Align feature columns
4. Apply preprocessing pipeline
5. Generate prediction probability
6. Map output:

   * Risk band
   * Decision
7. Return response

---

## 5. Prediction Output

### Response Includes

* Probability score
* Risk category:

  * Low Risk
  * Medium Risk
  * High Risk
* Final recommendation:

  * Approve
  * Review
  * Reject

---

## 6. Monitoring

### Data Monitoring

Track:

* Missing values
* Feature distribution shifts
* Unseen categories

### Model Monitoring

Track:

* Average prediction probability
* Approval/rejection rates
* Prediction class distribution

### Performance Monitoring

If labels become available:

* ROC-AUC
* Precision / Recall
* Default rate trends

---

## 7. Drift Detection

### Data Drift

Examples:

* Income distribution changes
* Credit utilization increases

### Concept Drift

Examples:

* Same customer profile begins defaulting more frequently

### Detection Methods

* Train vs production distribution comparison
* Threshold-based alerts
* Percentile and mean shift tracking

---

## 8. Retraining Strategy

Retrain when:

* Validation performance drops
* Drift thresholds exceeded
* Business KPIs degrade

### Suggested Policy

* Scheduled retraining every 1–3 months
* Event-triggered retraining after significant drift

---

## 9. Reliability & Scalability

### Reliability

* Health monitoring endpoint
* Graceful handling of invalid inputs
* Cached model after initial download

### Scalability

* Stateless API architecture
* Cloud deployment compatible
* Suitable for horizontal scaling

---

## 10. Risks & Limitations

* No real-time drift monitoring implemented
* Model calibration not applied
* Limited input validation
* Manual retraining process

---

## 11. Future Improvements

* Add Pydantic schema validation
* Add logging and monitoring dashboards
* Implement automated retraining
* Add authentication and rate limiting
* Use Docker for containerized deployment

---

## 12. Deployment Summary

* FastAPI-based prediction service implemented
* Runtime model loading enabled
* Batch and API inference supported
* Monitoring and retraining strategy defined
* System designed for scalable production deployment
