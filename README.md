# Customer Churn Prediction System

A complete Python machine-learning project that predicts whether a customer is likely to discontinue a service. It includes data preprocessing, feature selection, classification, evaluation, a saved trained model, and an interactive Streamlit interface.

## Features
- Data preprocessing for numerical and categorical variables
- Missing-value handling, scaling and one-hot encoding
- Feature selection with Mutual Information
- Logistic Regression and Random Forest comparison
- Accuracy, precision, recall, F1-score and ROC-AUC evaluation
- Saved trained model (`models/churn_model.pkl`)
- Interactive customer risk prediction using Streamlit
- Performance report
- Jupyter Notebook

## Project Structure
```text
customer_churn_prediction_system/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── customer_churn.csv
├── models/
│   ├── churn_model.pkl
│   └── model_info.json
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
├── reports/
│   └── performance_report.md
└── src/
    └── train_model.py
```

## Installation
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Train the model
From the project root:
```bash
python src/train_model.py
```

The script creates/uses the dataset, compares the classifiers, selects the best model using ROC-AUC, and saves the model.

## Run the interface
```bash
streamlit run app.py
```

## Dataset
The included CSV is a synthetic business dataset created for a reproducible academic/demo project. It contains realistic customer-behavior variables such as tenure, charges, support tickets, login activity, contract type, payment method and satisfaction. For a production deployment, replace it with real historical customer data while keeping the same target definition (`churn`).

## Machine Learning Workflow
1. Load customer history.
2. Separate target (`churn`) from predictors.
3. Split data using stratification.
4. Impute missing values.
5. Scale numerical variables.
6. One-hot encode categorical variables.
7. Apply mutual-information feature selection.
8. Train Logistic Regression and Random Forest.
9. Evaluate using classification metrics and ROC-AUC.
10. Save the best pipeline.
11. Use the saved pipeline for interactive predictions.

## Business Use
The predicted probability can be converted into risk tiers:
- Low: below 40%
- Medium: 40–70%
- High: 70% or above

Businesses can use these tiers to prioritize retention campaigns, customer support and targeted offers.

## Important Note
This project is intended for learning and demonstration. A real business system should validate the model on representative historical data, monitor drift, check fairness, calibrate probabilities, and integrate predictions with approved retention policies.
