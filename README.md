# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project detects fraudulent credit card transactions using Machine Learning. Due to the highly imbalanced nature of fraud data, SMOTE (Synthetic Minority Oversampling Technique) was used to balance the dataset before model training.

Three machine learning models were trained and compared:

* Logistic Regression
* Random Forest
* XGBoost

The final model selected for deployment was XGBoost due to its superior ROC-AUC score and strong fraud detection performance.

---

## Dataset Information

Dataset: Credit Card Fraud Detection Dataset

* Total Transactions: 284,807
* Fraudulent Transactions: 492
* Legitimate Transactions: 284,315
* Fraud Percentage: 0.1727%

Features:

* Time
* V1 to V28 (PCA-transformed anonymous features)
* Amount

Target Variable:

* Class = 0 → Legitimate Transaction
* Class = 1 → Fraudulent Transaction

---

## Technologies Used

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* SMOTE

### Data Visualization

* Matplotlib
* Seaborn

### Web Development

* Flask
* HTML
* CSS

---

## Project Workflow

1. Data Loading
2. Data Exploration
3. Exploratory Data Analysis (EDA)
4. Feature Scaling
5. Handling Class Imbalance using SMOTE
6. Model Training
7. Model Evaluation
8. Feature Importance Analysis
9. Model Saving
10. Flask Web Application Development

---

## Model Performance

### Logistic Regression

* Accuracy: 97.91%
* Recall: 90.82%
* F1 Score: 0.13
* ROC-AUC: 0.973

### Random Forest

* Precision: 0.83
* Recall: 0.83
* F1 Score: 0.83
* ROC-AUC: 0.969

### XGBoost (Selected Model)

* Accuracy: 99.93%
* Recall: 86.73%
* F1 Score: 0.81
* ROC-AUC: 0.981

---

## Feature Importance

Feature importance analysis showed that features such as V14 were among the strongest predictors of fraudulent transactions.

Since the dataset was anonymized using PCA, the exact real-world meaning of V1–V28 is not available.

---

## Flask Web Application

The web application allows users to:

* Upload CSV files containing transactions
* Upload Excel (.xlsx) files containing transactions
* Analyze transactions using the trained XGBoost model
* View fraud detection statistics
* View detected fraudulent transactions

---

## Project Structure

Credit-Card-Fraud-Detection/

│

├── app.py

├── fraud_model.pkl

├── scaler.pkl

├── requirements.txt

├── README.md

│

├── templates/

│ └── index.html

│

├── static/

│ └── style.css

│

├── notebook/

│ └── fraud_detection.ipynb

│

└── images/

├── homepage.png

├── dashboard.png

├── feature_importance.png

└── model_comparison.png

---

## Installation

Clone the repository:

git clone <repository-url>

Move into the project directory:

cd Credit-Card-Fraud-Detection

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

Open browser:

http://127.0.0.1:5000

---

## Future Improvements

* Real-time fraud detection
* Fraud risk scoring
* Interactive dashboard
* Cloud deployment
* REST API integration

---

## Author

Abaid-ur-Rehman

Machine Learning | Python | Flask
