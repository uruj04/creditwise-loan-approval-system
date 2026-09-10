# CreditWise — Loan Approval Predictor

## Overview
A machine learning classifier that predicts whether a loan application will be approved (1) or rejected (0) based on applicant financial and demographic features. Multiple models were trained and compared to identify the best-performing approach.

## Dataset
- Source: Kaggle — "Loan Approval Dataset"
- Size: 1,000 applicants, 20 raw features (income, credit score, DTI ratio, employment status, education, property area, etc.)
- Missing data: ~5% missing values across most columns, handled via imputation
- Target variable: Loan_Approved (1 = approved, 0 = not approved)

## Approach
- Preprocessing: Missing value imputation, one-hot encoding of categorical features (employment status, marital status, property area, education level, gender, employer category)
- Feature engineering: Created derived features including DTI_Ratio_sq, Credit_Score_sq, and log-transformed Applicant_Income to better capture non-linear relationships — resulting in 28 total features
- Train/test split with feature scaling for distance-based models
- Models trained and compared:
  - Logistic Regression
  - K-Nearest Neighbors (k=5)
  - Gaussian Naive Bayes

## Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 88.0%    | 0.785     | 0.836  | 0.810    |
| KNN (k=5)            | 78.5%    | 0.673     | 0.574  | 0.619    |
| Naive Bayes          | 86.0%    | 0.811     | 0.705  | 0.754    |

**Best model: Logistic Regression** — highest performance across accuracy, recall, and F1 score.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook credit_wise.ipynb
```

## Contributors
Mohammad Uruj Faizan (solo project)
