from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Columns used during training
EXPECTED_COLUMNS = [
    'Time',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
    'V8', 'V9', 'V10', 'V11', 'V12', 'V13',
    'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
    'V26', 'V27', 'V28',
    'Amount'
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    # Read uploaded file
    try:

        if file.filename.lower().endswith('.csv'):
            data = pd.read_csv(filepath)

        elif file.filename.lower().endswith('.xlsx'):
            data = pd.read_excel(filepath)

        else:
            return "Only CSV and XLSX files are supported."

    except Exception as e:
        return f"File reading error: {str(e)}"

    # Validate columns
    if list(data.columns) != EXPECTED_COLUMNS:

        return (
            "Invalid file columns.<br><br>"
            f"Expected:<br>{EXPECTED_COLUMNS}<br><br>"
            f"Found:<br>{list(data.columns)}"
        )

    try:

        # Make predictions
        predictions = model.predict(data)

        total_transactions = len(predictions)

        fraud_transactions = int(predictions.sum())

        legitimate_transactions = (
            total_transactions -
            fraud_transactions
        )

        fraud_percentage = round(
            (fraud_transactions /
             total_transactions) * 100,
            2
        )

        # Add prediction column
        result_df = data.copy()

        result_df['Prediction'] = predictions

        fraud_rows = result_df[
            result_df['Prediction'] == 1
        ]

        fraud_table = fraud_rows.to_html(
            classes='table table-striped',
            index=False
        )

        return render_template(
            'index.html',
            total=total_transactions,
            frauds=fraud_transactions,
            legitimate=legitimate_transactions,
            fraud_percent=fraud_percentage,
            fraud_table=fraud_table
        )

    except Exception as e:
        return f"Prediction error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)