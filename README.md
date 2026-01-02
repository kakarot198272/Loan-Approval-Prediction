# Loan Approval Prediction

A machine learning pipeline to predict loan approval decisions using
structured applicant data. The project explores classical ML models,
ensemble methods, and cost-sensitive decision-making to balance
profit and risk.

## Problem Statement
Financial institutions must decide whether to approve or reject loan
applications. Incorrect approvals can lead to defaults, while incorrect
rejections lead to missed revenue opportunities. This project models
loan approval as a binary classification problem and optimizes decisions
using business-aware cost functions.

## Dataset
- Source: Kaggle Playground Series S4E10
- Files:
  - `train.csv`: labeled loan applications
  - `test.csv`: unlabeled applications for inference

> Dataset files are not committed to this repository.

## Project Structure
## Methods
- Exploratory Data Analysis (EDA)
- Feature preprocessing with imputation and encoding
- Models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine
  - Neural Networks (MLP)
- Ensemble learning:
  - Soft Voting
  - Stacking
- Cost-sensitive threshold optimization

## Evaluation Metrics
- ROC-AUC
- Precision / Recall / F1-score
- Balanced Accuracy
- Cost-sensitive profit curves

## How to Run
```bash
pip install -r requirements.txt
python main.py


## Results Summary

The table below reports validation performance across multiple machine learning models.
ROC-AUC is used as the primary metric due to class imbalance in loan approvals.

| Model                | ROC-AUC | Accuracy | Precision | Recall | F1-score |
|---------------------|--------:|---------:|----------:|-------:|---------:|
| XGBoost             | 0.949   | 0.931    | 0.734     | 0.809  | 0.770    |
| Random Forest       | 0.931   | 0.951    | 0.935     | 0.705  | 0.804    |
| Neural Network (MLP)| 0.929   | 0.946    | 0.916     | 0.681  | 0.781    |
| SVM                 | 0.920   | 0.904    | 0.628     | 0.790  | 0.700    |
| AdaBoost            | 0.915   | 0.923    | 0.793     | 0.625  | 0.699    |
| Logistic Regression | 0.907   | 0.844    | 0.473     | 0.834  | 0.604    |

**Key takeaway:**  
Tree-based models (XGBoost, Random Forest) outperform linear baselines, highlighting
nonlinear relationships in borrower risk profiles.

