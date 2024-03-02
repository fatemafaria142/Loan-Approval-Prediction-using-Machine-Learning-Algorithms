# Loan Approval Prediction using Machine Learning Algorithms

## Overview

This project focuses on predicting loan approval using the XGBClassifier machine learning algorithm. The dataset used for this analysis is available [here](https://www.kaggle.com/datasets/yaminh/applicant-details-for-loan-approve/data).

## Results Summary

| Model Description                          | Accuracy | Precision | Recall | F1 Score | Log Loss |
| ------------------------------------------ | -------- | --------- | ------ | -------- | ---------|
| **Only using XGBClassifier**               | 0.9140   | 0.7337    | 0.5310 | 0.6161   | 0.1794   |
| **XGBClassifier using Undersampling**      | 0.8995   | 0.5658    | 0.9750 | 0.7160   | 0.3103   |
| **XGBClassifier using Oversampling**       | 0.9206   | 0.6275    | 0.9584 | 0.7584   | 0.2754   |
| **XGBClassifier using SMOTE + ENN**        | 0.8990   | 0.7318    | 0.3517 | 0.4751   | 0.4187   |
| **XGBClassifier using SMOTE + Tomek Links**| 0.9206   | 0.6275    | 0.9584 | 0.7584   | 0.2754   |

## Observations

- The baseline XGBClassifier performs well, showing high accuracy, precision, and F1 Score.
- Undersampling of the majority class results in lower precision but very high recall, indicating a higher tendency to correctly predict the minority class.
- Oversampling the minority class improves precision while maintaining high recall.
- SMOTE + ENN results in a trade-off between precision and recall, with a lower overall performance.
- SMOTE + Tomek Links achieves a balanced improvement in precision, recall, and F1 Score.

