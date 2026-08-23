# Product Requirements Document (PRD)

# Fraud Transaction Detection System

**Document Version:** 1.0
**Status:** Product Definition
**Product Type:** Machine Learning Web Application
**Platform:** Web
**Primary Technology:** Python, Flask, Scikit-learn, XGBoost, SHAP
**Target Users:** Data scientists, ML students, developers, analysts, and portfolio reviewers

---

## 1. Product Overview

The **Fraud Transaction Detection System** is a machine-learning-powered web application designed to estimate the fraud risk of credit-card transactions.

The platform combines an end-to-end machine-learning pipeline with an interactive Flask dashboard. Users can analyze individual transactions, upload transaction datasets for batch prediction, review model performance, investigate prediction explanations, monitor high-risk transactions, review historical predictions, and generate reports.

The system is designed as a **portfolio-grade ML application**, demonstrating how a complete fraud-risk workflow can be implemented from data preparation and model training through deployment and visualization.

> **Product Disclaimer:** This system provides fraud-risk estimates for educational and demonstration purposes. It is not a certified banking fraud-detection platform and must not be treated as proof that a transaction is fraudulent.

---

# 2. Product Vision

Build a professional, transparent, and easy-to-use fraud-risk analysis platform that demonstrates how machine learning can be used to identify potentially suspicious financial transactions while providing users with meaningful model explanations and performance information.

The product should make complex ML outputs understandable through:

* Risk scores
* Risk levels
* Clear visualizations
* Model metrics
* Explainable AI
* Transaction history
* Monitoring dashboards
* Downloadable reports

---

# 3. Problem Statement

Traditional machine-learning experiments often stop after training a model and printing evaluation metrics.

This creates several problems:

* Users cannot easily interact with the model.
* Individual transactions cannot be analyzed through a proper interface.
* Batch datasets require manual scripts.
* Model performance is difficult for non-technical users to understand.
* Predictions may lack explanations.
* There is no centralized transaction history.
* High-risk transactions are difficult to monitor.
* Results are not easily exportable.

The Fraud Transaction Detection System addresses these problems by combining the ML pipeline with a complete web-based analysis and monitoring platform.

---

# 4. Product Goals

## Primary Goals

1. Provide reliable fraud-risk predictions from trained ML models.
2. Support individual transaction analysis.
3. Support batch CSV prediction.
4. Compare multiple ML models objectively.
5. Handle severe class imbalance correctly.
6. Optimize the prediction threshold using validation data.
7. Provide explainable predictions using SHAP.
8. Maintain prediction history.
9. Provide high-risk transaction monitoring.
10. Generate downloadable reports.
11. Provide a secure and responsive web interface.
12. Keep ML training and application logic modular.

## Secondary Goals

* Demonstrate production-style Flask architecture.
* Provide transparent model evaluation.
* Make the system easy to retrain with compatible data.
* Provide a foundation for future real-time fraud detection.
* Demonstrate responsible ML practices.

---

# 5. Non-Goals

The initial version will **not** attempt to provide:

* Real banking transaction processing
* Real-time payment authorization
* Actual financial institution integration
* Customer identity verification
* Live card-network integration
* Production banking compliance
* Guaranteed fraud detection
* Automated blocking of real transactions
* Real customer profiling
* Production-grade distributed infrastructure

These capabilities may be considered in future versions.

---

# 6. Target Users

## 6.1 ML Student

Needs to:

* Understand model performance.
* Experiment with different algorithms.
* Analyze predictions.
* Learn about class imbalance.
* Understand SHAP explanations.

## 6.2 Data Scientist

Needs to:

* Compare models.
* Inspect metrics.
* Analyze false positives and false negatives.
* Test different thresholds.
* Review prediction distributions.

## 6.3 Developer

Needs to:

* Integrate the prediction API.
* Upload transaction batches.
* Retrieve prediction results.
* Extend the platform.

## 6.4 Portfolio Reviewer

Needs to:

* Quickly understand the project.
* See meaningful ML results.
* Inspect the architecture.
* Explore model explainability.
* Evaluate security and engineering quality.

---

# 7. User Roles

### Initial Version

The first version supports a single application user context.

### Future Version

Multi-user support may introduce:

* Administrator
* Data Scientist
* Analyst
* Reviewer

Future authorization must ensure that one user's transaction history and reports cannot be accessed by another user.

---

# 8. Core User Journeys

## Journey 1 — Analyze a Transaction

```text
Open Dashboard
      ↓
Go to Analyze Transaction
      ↓
Enter V1–V28 + Time + Amount
      ↓
Submit Transaction
      ↓
Validate Input
      ↓
Load Active Model
      ↓
Generate Fraud Probability
      ↓
Generate Risk Score
      ↓
Assign Risk Level
      ↓
Store Prediction
      ↓
Display Result
      ↓
Optional SHAP Explanation
```

---

## Journey 2 — Batch Prediction

```text
Open Batch Prediction
      ↓
Upload CSV
      ↓
Validate File
      ↓
Validate Required Columns
      ↓
Process Transactions
      ↓
Generate Predictions
      ↓
Calculate Risk Scores
      ↓
Generate Summary
      ↓
Store Prediction History
      ↓
Display Results
      ↓
Download CSV
```

---

## Journey 3 — Review Model Performance

```text
Open Model Performance
      ↓
View Active Model
      ↓
View Test Metrics
      ↓
Compare Four Models
      ↓
Review Confusion Matrix
      ↓
Review ROC / PR Curves
      ↓
Review Feature Importance
      ↓
View Selected Threshold
```

---

## Journey 4 — Investigate a High-Risk Transaction

```text
Open Monitoring
      ↓
View High/Critical Transactions
      ↓
Select Transaction
      ↓
View Transaction Details
      ↓
View Probability + Risk Score
      ↓
Request Explanation
      ↓
View SHAP Contributions
```

---

## Journey 5 — Generate Report

```text
Open Reports
      ↓
Select Report Type
      ↓
Select Filters / Date Range
      ↓
Generate Report
      ↓
Validate Report Data
      ↓
Generate CSV or PDF
      ↓
Download Report
```

---

# 9. Functional Requirements

## FR-001 — Dashboard

The dashboard shall provide a high-level overview of the application.

### Required Information

* Active model
* Dataset transaction count
* Fraud count
* Fraud percentage
* Total predictions
* Fraud predictions
* High-risk predictions
* Critical-risk predictions
* Recent transactions
* Recent alerts

### Acceptance Criteria

* Dashboard loads without manual data entry.
* Statistics are calculated dynamically.
* Prediction statistics come from the live prediction database.
* Dataset statistics come from the available dataset/model metadata.
* Missing model information is handled gracefully.

---

# 10. FR-002 — Manual Transaction Analyzer

The application shall provide a form for manually entering transaction features.

### Input Fields

* Time
* V1
* V2
* V3
* V4
* V5
* V6
* V7
* V8
* V9
* V10
* V11
* V12
* V13
* V14
* V15
* V16
* V17
* V18
* V19
* V20
* V21
* V22
* V23
* V24
* V25
* V26
* V27
* V28
* Amount

### Validation

The system shall:

* Require all mandatory fields.
* Reject non-numeric values.
* Reject malformed requests.
* Validate expected feature names.
* Apply the same preprocessing used during training.

### Output

The result shall contain:

* Fraud probability
* Risk score
* Risk level
* Prediction
* Active model
* Classification threshold
* Optional explanation

---

# 11. FR-003 — Risk Scoring

The system shall convert the model probability into a normalized risk score.

### Risk Score

```text
0–100
```

### Risk Levels

| Score Range | Risk Level |
| ----------: | ---------- |
|        0–19 | Very Low   |
|       20–39 | Low        |
|       40–59 | Medium     |
|       60–79 | High       |
|      80–100 | Critical   |

The exact boundaries shall remain configurable.

### Acceptance Criteria

* Every valid prediction receives a risk score.
* Risk score is deterministic for the same model output.
* Risk level corresponds to the configured range.
* Score must remain between 0 and 100.

---

# 12. FR-004 — Batch CSV Prediction

Users shall be able to upload a CSV containing transaction data.

### Requirements

The system shall:

* Accept CSV files only.
* Enforce maximum file size.
* Validate required columns.
* Reject malformed files.
* Prevent path traversal.
* Process valid transactions.
* Generate predictions.
* Generate risk scores.
* Generate risk levels.
* Provide summary statistics.
* Allow users to download results.

### Output Columns

The generated CSV should contain:

* Original transaction features
* Prediction
* Fraud probability
* Risk score
* Risk level

---

# 13. FR-005 — Model Comparison

The application shall compare:

1. Logistic Regression
2. Random Forest
3. Extra Trees
4. XGBoost

### Metrics

* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC

### Requirements

The comparison page shall:

* Display all available models.
* Display validation metrics.
* Identify the selected model.
* Show the model-selection metric.
* Clearly indicate the active model.

The system must never assume a specific model is the winner.

---

# 14. FR-006 — Model Performance

The model performance page shall display:

* Active model
* Test metrics
* Validation metrics
* Selected threshold
* Confusion matrix
* ROC curve
* Precision-Recall curve
* Feature importance where supported

### Metrics

```text
Precision
Recall
F1 Score
ROC-AUC
PR-AUC
Specificity
False Positive Rate
False Negative Rate
```

---

# 15. FR-007 — Explainable AI

The application shall support transaction-level explanations using SHAP.

### Requirements

The explanation system shall:

* Receive a transaction.
* Generate the model prediction.
* Calculate feature contributions.
* Rank influential features.
* Present positive and negative contributions.
* Handle unsupported models gracefully.

### Failure Behavior

If SHAP is unavailable:

```text
Prediction continues successfully.
Explanation is marked unavailable.
No application crash occurs.
```

---

# 16. FR-008 — Transaction History

The application shall store prediction records in SQLite.

### Stored Information

Each prediction should include:

* Prediction ID
* Timestamp
* Transaction features
* Fraud probability
* Prediction
* Risk score
* Risk level
* Model information where appropriate

### Filters

Users shall be able to filter:

* All
* Legitimate
* Fraud
* High Risk
* Critical

### Transaction Details

Selecting a transaction shall display the complete stored prediction information.

---

# 17. FR-009 — Fraud Monitoring

The monitoring page shall provide visibility into potentially suspicious predictions.

### Required Information

* Recent high-risk transactions
* Recent critical transactions
* Fraud prediction count
* High-risk count
* Critical count
* Risk distribution
* Recent activity

### Alerts

The system shall identify transactions that meet configured high-risk or critical-risk conditions.

---

# 18. FR-010 — Reports

The application shall generate:

### CSV Reports

CSV reports may contain:

* Transaction information
* Prediction statistics
* Risk statistics
* Model information
* Selected filters

### PDF Reports

PDF reports should provide a human-readable summary containing:

* Report title
* Generation timestamp
* Summary statistics
* Model information
* Risk distribution
* Fraud statistics
* Relevant transaction information

---

# 19. FR-011 — Settings

The Settings page shall provide application-level controls.

### Initial Settings

* Clear prediction history
* Display application information
* Display active model information

### Data Management

Users shall receive a confirmation step before clearing prediction history.

---

# 20. FR-012 — About Page

The About page shall explain:

* Project purpose
* Dataset
* Machine-learning models
* Technology stack
* Limitations
* Disclaimer
* Author information

---

# 21. Machine Learning Requirements

## ML-001 — Dataset Validation

Before training, the system shall verify:

* Dataset exists.
* Required columns exist.
* Data types are compatible.
* Target column exists.
* Missing values are handled.
* Dataset is sufficiently populated.

---

## ML-002 — Data Splitting

The system shall use a stratified:

```text
70% Train
15% Validation
15% Test
```

split.

The split must preserve class distribution as much as reasonably possible.

---

## ML-003 — Preprocessing

All preprocessing must be fitted using training data only.

The preprocessing pipeline must then be applied consistently to:

* Validation data
* Test data
* Manual predictions
* Batch predictions

This prevents preprocessing leakage.

---

## ML-004 — Imbalance Handling

Supported strategies:

```text
class_weight
random_undersample
smote
smote_tomek
```

Resampling must occur only on training data.

---

## ML-005 — Model Training

The system shall train all four configured models using the same training split and preprocessing approach.

---

## ML-006 — Model Selection

The default selection metric shall be:

```text
Validation PR-AUC
```

The winning model shall be persisted as the active model.

---

## ML-007 — Threshold Optimization

The threshold shall be optimized using validation predictions.

Supported objectives:

```text
max_f1
prioritize_recall
balanced
```

The test set must not influence threshold selection.

---

## ML-008 — Final Evaluation

After model and threshold selection:

1. Lock the configuration.
2. Evaluate once on test data.
3. Save test metrics.
4. Persist model metadata.

The test set shall not be used for further tuning.

---

# 22. Model Artifacts

The training system shall generate:

```text
models/
├── fraud_model.pkl
├── preprocessing_pipeline.pkl
├── feature_columns.pkl
└── model_metadata.json
```

### Metadata

The metadata file should contain information such as:

* Model name
* Training timestamp
* Dataset information
* Feature columns
* Selection metric
* Validation metrics
* Test metrics
* Threshold
* Threshold objective
* Imbalance strategy
* Model version

---

# 23. API Requirements

All APIs shall:

* Validate input.
* Return JSON where applicable.
* Use appropriate HTTP status codes.
* Avoid exposing stack traces.
* Avoid exposing secrets.
* Avoid exposing internal filesystem paths.
* Handle expected failures gracefully.

### HTTP Status Examples

```text
200 — Successful request
400 — Invalid input
404 — Resource not found
413 — Upload too large
429 — Rate limit exceeded
500 — Unexpected server error
503 — Model unavailable
```

---

# 24. Security Requirements

## SEC-001 — CSRF

Browser-facing state-changing requests shall use CSRF protection.

## SEC-002 — Rate Limiting

Prediction endpoints shall be rate limited to reduce abuse.

## SEC-003 — File Upload Security

The application shall:

* Restrict file extensions.
* Enforce upload size.
* Use secure filenames.
* Prevent path traversal.
* Store uploads in controlled directories.
* Remove temporary files when appropriate.

## SEC-004 — Secrets

Secrets must be loaded from environment variables.

No API keys or secret credentials may be committed to source control.

## SEC-005 — Error Handling

Production responses must never expose:

* Python tracebacks
* Internal paths
* Database details
* Environment variables
* Credentials

## SEC-006 — Database

All user-controlled database values must be safely parameterized.

---

# 25. Privacy Requirements

The application should minimize stored information.

Prediction history shall only contain information necessary for application functionality.

Users shall be able to clear prediction history.

The system shall clearly communicate that the dataset is anonymized and that the application is a demonstration system.

---

# 26. UI/UX Requirements

## Design Direction

The dashboard should have a modern analytics-focused design.

### Visual Characteristics

* Clean
* Professional
* Data-centric
* Responsive
* Minimal visual clutter
* Consistent cards
* Clear status indicators
* Accessible charts
* Dark/light theme

### Navigation

Primary navigation:

```text
Dashboard
Analyze
Batch Prediction
Transactions
Monitoring
Analytics
Model
Explain
Reports
Settings
About
```

---

# 27. Dashboard Components

The main dashboard should contain:

### KPI Cards

* Total Transactions
* Fraud Transactions
* Fraud Rate
* Total Predictions
* High-Risk Predictions
* Critical Predictions

### Charts

* Risk distribution
* Fraud vs legitimate
* Recent prediction activity
* Model performance summary

### Alert Section

Display the latest high/critical transactions.

### Quick Actions

* Analyze Transaction
* Batch Prediction
* View Monitoring
* View Model Performance

---

# 28. Analytics Requirements

The analytics page should provide:

* Transaction distribution
* Fraud distribution
* Risk distribution
* Model metrics
* Confusion matrix
* ROC curve
* Precision-Recall curve
* Feature importance

Charts should use live data generated from the model or database.

No analytical visualization should use hardcoded prediction results.

---

# 29. Performance Requirements

## Application Performance

For normal local usage:

* Dashboard should load quickly.
* Individual prediction should respond within a reasonable interactive timeframe.
* API responses should not perform unnecessary expensive operations.
* Batch processing should provide progress or clear processing feedback for large files.

## ML Performance

Model training performance is not required to be real-time.

Training may take significantly longer depending on:

* Hardware
* Dataset size
* XGBoost configuration
* Resampling strategy

---

# 30. Reliability Requirements

The system shall continue functioning when optional components fail.

Examples:

### SHAP unavailable

Prediction still works.

### Model unavailable

Dashboard can load, but prediction endpoints return `503`.

### Invalid CSV

The upload is rejected with a user-friendly validation message.

### Invalid transaction

The request is rejected without crashing the server.

---

# 31. Database Requirements

The initial implementation shall use:

```text
SQLite
```

The database shall store prediction history.

The architecture should keep database operations isolated from business logic so SQLite can later be replaced with PostgreSQL.

---

# 32. File Management

### Dataset

```text
data/creditcard.csv
```

### Temporary Uploads

```text
uploads/
```

### Generated Reports

```text
exports/
```

### Model Artifacts

```text
models/
```

Generated and sensitive files should be excluded from version control where appropriate.

---

# 33. Configuration Requirements

Important configuration options should be environment-based or centralized.

Examples:

```text
SECRET_KEY
DATABASE_URL
IMBALANCE_STRATEGY
THRESHOLD_OBJECTIVE
MODEL_SELECTION_METRIC
MAX_UPLOAD_SIZE
RATE_LIMIT
```

The system should provide safe defaults for development.

---

# 34. Testing Requirements

The project should include automated tests for:

### ML

* Dataset validation
* Preprocessing
* Model loading
* Prediction
* Threshold behavior
* Metric calculations

### API

* Valid prediction
* Invalid prediction
* Missing fields
* Batch upload
* Invalid file
* Model unavailable
* Transaction retrieval
* Report generation

### Security

* CSRF protection
* Upload restrictions
* Path traversal
* Rate limiting
* Sanitized errors

### Database

* Prediction creation
* Prediction retrieval
* Filtering
* History deletion

---

# 35. Acceptance Criteria

The product is considered functionally complete when:

* [ ] Dataset can be loaded and validated.
* [ ] Four ML models can be trained.
* [ ] Validation metrics are calculated.
* [ ] Best model is automatically selected.
* [ ] Classification threshold is optimized.
* [ ] Test evaluation occurs only after model selection.
* [ ] Model artifacts are saved successfully.
* [ ] Flask application starts successfully.
* [ ] Individual transactions can be analyzed.
* [ ] Fraud probability is displayed.
* [ ] Risk score is displayed.
* [ ] Risk level is displayed.
* [ ] Batch CSV prediction works.
* [ ] Batch results can be downloaded.
* [ ] Model comparison is available.
* [ ] Confusion matrix is displayed.
* [ ] ROC and PR data are available.
* [ ] Feature importance is displayed where supported.
* [ ] SHAP explanations work when available.
* [ ] SHAP failure does not break prediction.
* [ ] Prediction history is stored.
* [ ] Transaction filtering works.
* [ ] High-risk monitoring works.
* [ ] CSV reports can be generated.
* [ ] PDF reports can be generated.
* [ ] CSRF protection is active.
* [ ] Rate limiting is active.
* [ ] Upload security is implemented.
* [ ] Secrets are not hardcoded.
* [ ] Errors do not expose sensitive information.
* [ ] Automated tests pass.

---

# 36. Success Metrics

Since this is primarily a portfolio and educational system, success should be measured through both ML quality and software quality.

### ML Success

* Strong validation PR-AUC
* Meaningful fraud recall
* Controlled false-positive rate
* Stable test performance
* Correct threshold optimization
* No data leakage

### Engineering Success

* Clean modular architecture
* Reliable Flask APIs
* Successful batch processing
* Automated tests
* Secure file handling
* Clear error handling
* Reproducible model training

### UX Success

* Users can analyze a transaction without technical knowledge.
* Model results are understandable.
* Risk levels are easy to interpret.
* Model explanations are accessible.
* Reports are easy to generate.

---

# 37. Architecture

The system should follow a layered architecture:

```text
┌──────────────────────────────────────────────┐
│                  Web Browser                 │
│ Dashboard / Analyze / Batch / Analytics     │
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                Flask Routes                  │
│ Pages + REST API + Request Validation       │
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│               Service Layer                  │
│ Prediction / Reports / Analytics / SHAP     │
└───────────────┬─────────────────┬────────────┘
                │                 │
                ▼                 ▼
┌──────────────────────┐   ┌───────────────────┐
│ ML Model + Pipeline  │   │ SQLite Database   │
│ Scaler + Metadata    │   │ Prediction Log    │
└───────────┬──────────┘   └───────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│              Training Pipeline               │
│ Data → Split → Preprocess → Train → Evaluate│
└──────────────────────────────────────────────┘
```

---

# 38. Future Product Roadmap

## Phase 1 — Core ML

* Dataset processing
* Four models
* Evaluation
* Model selection
* Threshold optimization
* Artifact persistence

## Phase 2 — Web Application

* Flask dashboard
* Manual analyzer
* Batch prediction
* Prediction history

## Phase 3 — Explainability & Analytics

* SHAP
* Model comparison
* Confusion matrix
* ROC/PR curves
* Feature importance
* Monitoring

## Phase 4 — Reporting & Security

* CSV reports
* PDF reports
* CSRF
* Rate limiting
* Upload security
* Error sanitization

## Phase 5 — Advanced Platform

* Multi-user authentication
* Role-based access control
* PostgreSQL
* Redis
* Real-time transaction ingestion
* Kafka
* Model versioning
* Drift detection
* Automated retraining
* Notification system

---

# 39. Future Production Architecture

A future production version could evolve toward:

```text
Transaction Sources
       ↓
API Gateway
       ↓
Kafka / Event Stream
       ↓
Fraud Feature Service
       ↓
Redis / PostgreSQL
       ↓
ML Inference Service
       ↓
Risk Engine
       ↓
 ┌─────┼─────────┐
 ↓     ↓         ↓
Alerts Review  Dashboard
       ↓
Human Analyst
```

This architecture would allow the system to move from an educational batch-oriented application toward a real-time fraud-risk platform.

---

# 40. Product Risks

| Risk                      | Impact   | Mitigation                                           |
| ------------------------- | -------- | ---------------------------------------------------- |
| Severe class imbalance    | High     | PR-AUC, stratification, imbalance strategies         |
| Data leakage              | Critical | Train-only preprocessing and resampling              |
| Dataset outdated          | High     | Clearly communicate limitations                      |
| False positives           | High     | Threshold optimization and precision monitoring      |
| False negatives           | High     | Recall monitoring and configurable objectives        |
| SHAP incompatibility      | Medium   | Graceful fallback                                    |
| Malicious uploads         | High     | File validation and path protection                  |
| Model unavailable         | Medium   | Graceful `503` responses                             |
| Hardcoded metrics         | High     | Generate all metrics dynamically                     |
| Prediction history growth | Medium   | Clear-history controls and future retention policies |

---

# 41. Ethical and Responsible ML Considerations

The system should clearly communicate that:

* A model prediction is not a fact.
* Fraud probability is an estimate.
* False positives can negatively affect legitimate users.
* False negatives can allow fraudulent activity.
* Dataset limitations can affect generalization.
* Model performance should be evaluated continuously in real-world deployments.

The system should therefore be presented as a **risk estimation and ML demonstration platform**, rather than an autonomous fraud adjudication system.

---

# 42. Definition of Done

The project is considered complete when:

1. The complete ML pipeline runs successfully.
2. Dataset leakage is prevented.
3. Four models are trained and compared.
4. The best model is selected using validation PR-AUC.
5. Threshold optimization occurs only on validation data.
6. Final test metrics are generated once.
7. Model artifacts are persisted.
8. Flask dashboard is functional.
9. Individual prediction works.
10. Batch prediction works.
11. Risk scoring works.
12. Transaction history works.
13. Monitoring works.
14. SHAP explanation works or fails gracefully.
15. Model comparison works.
16. Analytics work.
17. CSV/PDF reports work.
18. Security protections are implemented.
19. Automated tests pass.
20. Documentation clearly explains limitations and setup.

---

# 43. Final Product Definition

The **Fraud Transaction Detection System** is an end-to-end machine-learning application that demonstrates the complete lifecycle of a fraud-risk model:

```text
Data
 ↓
Validation
 ↓
Preprocessing
 ↓
Class-Imbalance Handling
 ↓
Model Training
 ↓
Model Comparison
 ↓
Model Selection
 ↓
Threshold Optimization
 ↓
Final Evaluation
 ↓
Model Persistence
 ↓
Flask API
 ↓
Interactive Dashboard
 ↓
Prediction
 ↓
Explainability
 ↓
Monitoring
 ↓
Reporting
```

The primary objective is not simply to train a fraud classifier, but to demonstrate how a machine-learning model can be transformed into a **usable, explainable, secure, and maintainable web application**.
