# Customer Churn Prediction — Performance Report

Best model: **Logistic Regression**

## Model comparison

| Model               |   Accuracy |   Precision |   Recall |       F1 |   ROC_AUC |
|:--------------------|-----------:|------------:|---------:|---------:|----------:|
| Logistic Regression |      0.785 |    0.66586  | 0.78125  | 0.718954 |  0.867985 |
| Random Forest       |      0.782 |    0.676316 | 0.730114 | 0.702186 |  0.861374 |

## Confusion matrix

[[510, 138], [77, 275]]

## Classification report

```
              precision    recall  f1-score   support

      Retain       0.87      0.79      0.83       648
       Churn       0.67      0.78      0.72       352

    accuracy                           0.79      1000
   macro avg       0.77      0.78      0.77      1000
weighted avg       0.80      0.79      0.79      1000

```

## Dataset
- Rows: 5,000
- Churn rate: 35.16%
- Train/test split: 80/20 stratified
- Random seed: 42