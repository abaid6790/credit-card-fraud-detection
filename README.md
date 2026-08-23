# Fraud Transaction Detection System

A machine-learning-powered **fraud-risk detection platform** built with Python and Flask. The system provides an end-to-end workflow for training fraud-detection models, analyzing individual and batch transactions, estimating fraud risk, comparing model performance, explaining predictions with SHAP, monitoring high-risk transactions, and generating reports.

> **Important:** This project is a **fraud-risk estimator for educational and portfolio purposes**. It is not a certified production banking fraud-detection system and should not be used to make real financial decisions.

---

## Overview

The Fraud Transaction Detection System combines a leakage-safe machine-learning pipeline with a production-style Flask dashboard.

The application trains and evaluates four machine-learning models:

* Logistic Regression
* Random Forest
* Extra Trees
* XGBoost

The models are evaluated using the same validation split, and the best model is selected according to **validation PR-AUC**. The classification threshold is then optimized on the validation set before the final model is evaluated once on the untouched test set.

The trained model is exposed through a Flask web application and API, allowing users to:

* Analyze individual transactions
* Upload CSV files for batch prediction
* View fraud probabilities and risk scores
* Compare machine-learning models
* Explore performance metrics and curves
* Understand individual predictions using SHAP
* Review historical predictions
* Monitor high-risk transactions
* Generate CSV and PDF reports

All predictions, metrics, charts, and dashboard statistics are generated dynamically from the trained models and application data. **No prediction results or performance metrics are hardcoded.**

---

## Key Features

### Transaction Analysis

* Manual transaction analyzer
* Complete `V1–V28`, `Time`, and `Amount` input
* Real-time fraud-risk prediction
* Fraud probability
* 0–100 risk score
* Risk classification from **Very Low → Critical**

### Batch Prediction

* CSV transaction upload
* Automatic validation
* Batch fraud prediction
* Fraud/legitimate transaction summaries
* Risk-level statistics
* Downloadable prediction results

### Machine Learning

* Logistic Regression baseline
* Random Forest
* Extra Trees
* XGBoost
* Automatic model comparison
* Validation-based model selection
* Configurable class-imbalance strategies
* Threshold optimization

### Model Evaluation

The system calculates:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix
* False Positive Rate
* False Negative Rate
* Specificity

Accuracy is intentionally **not used as the primary model-selection metric** because the dataset is extremely imbalanced.

### Explainable AI

The application integrates **SHAP** for per-transaction explanations.

Depending on model compatibility, the system uses:

* SHAP `TreeExplainer`
* Generic SHAP `Explainer` fallback

If SHAP is unavailable or unsupported for the active model, prediction continues normally and the explanation is simply omitted.

### Transaction History

* Stores prediction history in SQLite
* Filter by:

  * All
  * Legitimate
  * Fraud
  * High Risk
  * Critical
* View complete transaction details
* Clear prediction history from Settings

### Fraud Monitoring

* Recent high-risk transactions
* Critical-risk alerts
* Fraud-risk statistics
* Recent prediction activity

### Reports

* CSV report generation
* PDF report generation
* Batch prediction exports
* Model and transaction statistics

### Security

* CSRF protection
* Rate limiting
* Secure file handling
* Path-traversal protection
* Upload-size limits
* Parameterized SQL queries
* Environment-based secrets
* Sanitized API errors
* No stack traces exposed to users

### UI

* Responsive Flask dashboard
* Dark/light theme
* Modern dashboard interface
* Analytics visualizations
* Model-performance pages
* Transaction monitoring interface

---

## Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle.

**Dataset:** Credit Card Fraud Detection — Machine Learning Group, ULB

[Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?utm_source=chatgpt.com)

The dataset contains:

* **284,807** credit-card transactions
* **492** fraudulent transactions
* Approximately **0.172%** fraud rate
* European card transactions
* Anonymized PCA-transformed features

The dataset contains the following main fields:

* `Time`
* `V1` through `V28`
* `Amount`
* `Class`

The raw dataset is **not committed to Git**.

See `data/README.md` for dataset preparation instructions and additional information.

---

## Machine Learning Pipeline

The training workflow follows a leakage-safe process:

```text
Load Dataset
      ↓
Validate Data
      ↓
Clean Data
      ↓
Exploratory Data Analysis
      ↓
Analyze Class Imbalance
      ↓
Separate Features / Target
      ↓
Stratified 70 / 15 / 15 Split
      ↓
Fit Preprocessing on Training Data Only
      ↓
Apply Imbalance Strategy to Training Data Only
      ↓
Train 4 Models
      ↓
Evaluate on Validation Set
      ↓
Select Best Model by Validation PR-AUC
      ↓
Optimize Classification Threshold
      ↓
Lock Model + Threshold
      ↓
Evaluate Once on Test Set
      ↓
Save Model Artifacts + Metadata
```

### Data Split

The dataset is divided using a stratified:

* **70% Training**
* **15% Validation**
* **15% Test**

The validation set is used for:

* Model comparison
* Model selection
* Threshold optimization

The test set remains untouched until the final evaluation.

> The test set is used exactly once after model and threshold selection are complete. It is never used for tuning.

---

## Handling Class Imbalance

Fraud detection is a highly imbalanced classification problem. With only around **0.17% fraudulent transactions**, standard accuracy can be misleading.

The training pipeline supports multiple imbalance strategies through:

```text
IMBALANCE_STRATEGY
```

Available options:

```text
class_weight
random_undersample
smote
smote_tomek
```

The default strategy is:

```text
class_weight
```

Any resampling is performed **only on the training split**.

Validation and test data remain in their original distributions.

---

## Machine Learning Models

The project trains four different classification algorithms.

### 1. Logistic Regression

Used as an interpretable baseline model.

Advantages:

* Simple
* Fast
* Interpretable
* Useful baseline for comparison

### 2. Random Forest

An ensemble of decision trees that can capture nonlinear relationships between transaction features.

### 3. Extra Trees

An extremely randomized tree ensemble that provides another strong tree-based comparison.

### 4. XGBoost

A gradient-boosted tree model designed for high-performance classification and commonly effective for structured/tabular datasets.

The application does **not assume which model will perform best**.

Every model is evaluated using the same validation data, and the best model is selected automatically.

---

## Model Selection

The default model-selection metric is:

```text
PR-AUC
```

The model with the highest validation PR-AUC is selected as the active model.

This is particularly useful for fraud detection because the positive class is extremely rare.

The selection criterion can be configured in:

```text
ml/train.py
```

---

## Evaluation Metrics

The system evaluates model performance using several metrics.

### Precision

Measures how many transactions predicted as fraud were actually fraudulent.

### Recall

Measures how many actual fraudulent transactions were successfully detected.

### F1 Score

Balances precision and recall.

### ROC-AUC

Measures ranking performance across classification thresholds.

### PR-AUC

Measures precision-recall performance and is particularly informative for highly imbalanced datasets.

### Confusion Matrix

Provides:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Additional Metrics

The application also calculates:

* False Positive Rate
* False Negative Rate
* Specificity

Accuracy is intentionally not used as the main selection criterion.

For example, a model that predicts every transaction as legitimate could achieve approximately **99.8% accuracy** while detecting essentially no fraud.

---

## Threshold Optimization

The probability threshold is configurable through:

```text
THRESHOLD_OBJECTIVE
```

Available objectives include:

```text
max_f1
prioritize_recall
balanced
```

The default objective is:

```text
max_f1
```

The threshold is optimized using the **validation dataset**, not the test dataset.

After optimization, the threshold is locked before final test evaluation.

The selected threshold is also displayed on the Model Performance page.

---

## Fraud Risk Scoring

The application converts the model's fraud probability into a **0–100 risk score**.

Transactions are then assigned a risk level such as:

```text
Very Low
Low
Medium
High
Critical
```

The risk score is intended to make model output easier to understand through the dashboard.

> A high-risk prediction indicates elevated model-estimated risk. It does not prove that a transaction is fraudulent.

---

## Explainable AI

The application uses **SHAP (SHapley Additive exPlanations)** to provide transaction-level explanations.

For supported tree-based models, the system uses:

```text
TreeExplainer
```

If necessary, it falls back to:

```text
Explainer
```

SHAP explanations show which features contributed toward the model's prediction and help users understand why a particular transaction received its risk estimate.

SHAP is optional. If the package is unavailable or the active model cannot be explained, normal prediction functionality continues.

---

## Web Dashboard

The Flask application contains the following pages:

| Page                 | Purpose                             |
| -------------------- | ----------------------------------- |
| `/`                  | Main dashboard                      |
| `/analyze`           | Analyze an individual transaction   |
| `/batch`             | Upload and predict CSV transactions |
| `/transactions`      | View prediction history             |
| `/transactions/<id>` | View transaction details            |
| `/monitoring`        | Monitor high-risk transactions      |
| `/analytics`         | View analytics and statistics       |
| `/model`             | Model performance and comparison    |
| `/explain`           | Explain model predictions           |
| `/reports`           | Generate reports                    |
| `/settings`          | Application and history settings    |
| `/about`             | Project information                 |

---

## API

The application provides JSON-based API endpoints for prediction, analytics, transactions, monitoring, and reporting.

| Method | Endpoint                      | Description                              |
| ------ | ----------------------------- | ---------------------------------------- |
| `POST` | `/api/predict`                | Predict a single transaction             |
| `POST` | `/api/predict?explain=true`   | Predict with SHAP explanation            |
| `POST` | `/api/predict/batch`          | Upload CSV and generate predictions      |
| `GET`  | `/api/batch/download/<token>` | Download batch prediction results        |
| `GET`  | `/api/dashboard`              | Dashboard summary                        |
| `GET`  | `/api/analytics`              | Analytics summary                        |
| `GET`  | `/api/transactions`           | Retrieve prediction history              |
| `GET`  | `/api/transactions/<id>`      | Retrieve one prediction                  |
| `POST` | `/api/transactions/clear`     | Clear prediction history                 |
| `GET`  | `/api/model`                  | Active model metadata and test metrics   |
| `GET`  | `/api/model/comparison`       | Compare all four models                  |
| `GET`  | `/api/alerts`                 | Retrieve high/critical-risk transactions |
| `POST` | `/api/reports/csv`            | Generate CSV report                      |
| `POST` | `/api/reports/pdf`            | Generate PDF report                      |

All API endpoints validate their inputs and return sanitized JSON responses.

The application does not expose:

* Stack traces
* Internal file paths
* API keys
* Secrets
* Sensitive server configuration

---

## Installation

### 1. Clone the Repository

```bash
git clone <this-repo>
cd fraud-transaction-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Copy the example environment file:

**Windows:**

```bash
copy .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

Then update the required configuration, especially:

```text
SECRET_KEY
```

Never commit `.env` or other secrets to Git.

---

## Dataset Preparation

### Step 1 — Download the Dataset

Download `creditcard.csv` from Kaggle:

[Download Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?utm_source=chatgpt.com)

### Step 2 — Place the Dataset

Put the file here:

```text
data/creditcard.csv
```

The expected structure is:

```text
fraud-transaction-detection/
└── data/
    ├── creditcard.csv
    └── README.md
```

The CSV file is excluded from version control.

---

## Training the Models

Run:

```bash
python ml/train.py
```

The training script will:

1. Validate the dataset
2. Load and clean the data
3. Analyze class imbalance
4. Create the train/validation/test splits
5. Fit preprocessing using training data
6. Apply the selected imbalance strategy
7. Train all four models
8. Evaluate the models on validation data
9. Select the best model
10. Optimize the classification threshold
11. Evaluate the locked model on the test set
12. Save the trained artifacts and metadata

Generated artifacts include:

```text
models/
├── fraud_model.pkl
├── preprocessing_pipeline.pkl
├── feature_columns.pkl
└── model_metadata.json
```

The generated model files should not normally be committed to Git unless intentionally versioning trained artifacts.

---

## Running the Application

After training:

```bash
python run.py
```

Open:

```text
http://localhost:5000
```

If no trained model exists, the Flask application can still start, but prediction endpoints return:

```text
503 Service Unavailable
```

until the model-training step has been completed.

---

## Project Structure

```text
fraud-transaction-detection/
│
├── app/
│   ├── routes/
│   ├── services/
│   ├── templates/
│   └── static/
│
├── ml/
│   ├── dataset loading
│   ├── preprocessing
│   ├── EDA
│   ├── training
│   └── evaluation
│
├── models/
│   └── generated model artifacts
│
├── data/
│   ├── creditcard.csv
│   └── README.md
│
├── uploads/
│   └── temporary batch uploads
│
├── exports/
│   └── generated reports and results
│
├── instance/
│   └── SQLite database
│
├── tests/
│   └── automated tests
│
├── requirements.txt
├── run.py
├── .env.example
├── .gitignore
└── README.md
```

---

## Security and Privacy

The application includes several security measures designed for a portfolio-quality web application.

### Web Security

* CSRF protection
* Rate limiting
* Secure request validation
* Sanitized error responses

### File Security

* Secure filenames
* Path-traversal protection
* Upload-size restrictions
* Controlled download paths
* Temporary upload handling

### Database Security

* Parameterized SQL queries
* Controlled database operations
* No raw SQL interpolation from user input

### Secret Management

Sensitive configuration is stored using environment variables:

```text
.env
```

Secrets should never be hardcoded or committed to GitHub.

### Prediction History

The application stores only the transaction information required for its dashboard and prediction-history functionality.

Prediction history can be cleared from:

```text
Settings → Clear Prediction History
```

---

## Testing

Automated tests are located in:

```text
tests/
```

Run the test suite with:

```bash
pytest
```

Tests are intended to cover important application behavior, validation, API responses, model functionality, and security-related functionality.

---

## Limitations

This project has several important limitations.

### Dataset Limitations

The model is trained using a **2013 dataset from a single European card processor**.

Therefore, the learned patterns may not represent:

* Modern fraud techniques
* Other countries
* Other payment networks
* Different financial institutions
* Current transaction behavior

### Anonymized Features

`V1–V28` are anonymized PCA components.

Their individual values do not have straightforward real-world interpretations such as:

* Merchant type
* Device type
* Customer location
* Purchase category

### Missing Real-World Signals

The dataset does not provide important production fraud signals such as:

* Merchant information
* Device fingerprinting
* IP address
* Geolocation
* Customer history
* Transaction velocity
* Account behavior
* Previous fraud events

### Prediction Interpretation

A fraud prediction is a **risk estimate**, not proof of fraud.

Likewise, a legitimate prediction does not guarantee that a transaction is safe.

### Production Limitations

This project is not intended to replace a production banking fraud system.

A real-world system would typically require:

* Real-time transaction streams
* Low-latency inference
* Behavioral features
* Velocity rules
* Device fingerprinting
* Geographic analysis
* Continuous monitoring
* Data-drift detection
* Model retraining
* Human review
* Audit logging
* High-availability infrastructure
* Stronger authentication and authorization
* Production-grade databases and messaging systems

---

## Future Improvements

Potential improvements include:

* Real-time transaction processing
* Kafka-based transaction streams
* Redis caching and rate limiting
* PostgreSQL production database
* Customer behavioral profiling
* Transaction velocity features
* Device fingerprinting
* Geographic anomaly detection
* Autoencoder-based anomaly detection
* Isolation Forest
* Model drift detection
* Automated model retraining
* Email alerts
* SMS alerts
* Telegram notifications
* Webhook integrations
* Human fraud-review workflows
* Multi-user authentication
* Role-based access control
* Advanced audit logging
* Model versioning
* Production monitoring

The current architecture separates the machine-learning pipeline from the Flask application and service layer, making these improvements possible without completely redesigning the project.

---

## Technology Stack

### Backend

* Python
* Flask
* SQLite
* SQLAlchemy

### Machine Learning

* scikit-learn
* XGBoost
* imbalanced-learn

### Explainable AI

* SHAP

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Charting libraries used by the dashboard

### Reporting

* CSV
* PDF

### Security

* Flask-WTF / CSRF protection
* Rate limiting
* Secure file handling
* Environment-based configuration

---

## Why PR-AUC Matters for Fraud Detection

Fraud detection is a classic **imbalanced classification problem**.

Only a tiny fraction of the transactions in this dataset are fraudulent. Because of that, accuracy can give a misleading impression of model quality.

For example:

```text
Fraudulent transactions: ~0.17%
Legitimate transactions: ~99.83%
```

A model that predicts every transaction as legitimate could therefore achieve extremely high accuracy while completely failing at the actual task: detecting fraud.

For this reason, the project emphasizes:

```text
Precision
Recall
F1
PR-AUC
ROC-AUC
False Positive Rate
False Negative Rate
Specificity
```

with **PR-AUC as the default model-selection metric**.

---

## Disclaimer

This project is intended for **education, experimentation, and portfolio demonstration**.

It demonstrates how an end-to-end machine-learning fraud-risk application can be designed, trained, evaluated, explained, and deployed through a Flask web interface.

It should **not** be used as a real banking, financial, compliance, or fraud-prevention system without substantial additional engineering, validation, security controls, monitoring, and domain-specific testing.

---

## Author

**Abaid-ur-Rehman**

Built as a portfolio project demonstrating:

* End-to-end machine learning
* Imbalanced classification
* Leakage-safe model evaluation
* Model comparison
* Threshold optimization
* Explainable AI
* Flask application development
* REST API design
* Data visualization
* Security-conscious web development
* Automated testing
* Production-style project architecture
