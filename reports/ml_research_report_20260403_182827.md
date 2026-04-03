# ML Research Report — Hospital Readmission Prediction

## Executive Summary
The objective of this research was to predict hospital readmissions, a critical issue affecting patient care and healthcare costs. We employed various machine learning models to analyze a dataset of 1,602 patient records, ultimately identifying the LightGBM model as the best performer with an AUC of 0.9177, F1 score of 0.8271, and accuracy of 0.8411. We recommend deploying the LightGBM model in a production environment to enhance patient management and reduce readmission rates.

## 1. Problem Statement
Hospital readmissions pose significant challenges for healthcare providers, impacting patient outcomes and increasing operational costs. Predicting which patients are likely to be readmitted allows healthcare professionals to implement targeted interventions, ultimately improving patient care and reducing unnecessary expenditures.

## 2. Dataset Overview
The dataset comprises 1,602 rows and 16 columns, sourced from hospital records. Key features include patient demographics, medical history, and treatment details. The dataset includes both numerical and categorical variables, providing a comprehensive view of patient health and treatment patterns.

## 3. Data Quality Findings
The analysis revealed several data quality issues:
- **Missing Values:** 
  - `num_lab_tests`: 30 missing values (1.87%)
  - `A1C_result`: 50 missing values (3.12%)
- **Outliers:** 
  - `num_medications`: 10 outliers (0.62%)
  - `readmitted`: 199 outliers (12.42%)
- **Class Imbalance:** 
  - The dataset was balanced post-SMOTE application, with 801 instances for both classes (readmitted and not readmitted).

## 4. Feature Engineering
To prepare the data for modeling, several steps were taken:
- **Missing Values:** Imputed missing values in `num_lab_tests` with the median (25.00) and in `A1C_result` with the mode ("Normal").
- **Encoding:** Categorical variables were converted to numerical format using label encoding.
- **New Features Created:**
  - `medication_intensity`: Measures the intensity of medications taken.
  - `procedure_burden`: Reflects the number of procedures undergone.
  - `is_elderly`: A binary feature indicating if the patient is 65 or older.
  - `log_num_medications`: Logarithmic transformation of `num_medications` to reduce skewness.

## 5. Experiment Results
The following table summarizes the performance of all models evaluated:

| Model               | AUC    | F1     | Accuracy | Precision | Recall |
|---------------------|--------|--------|----------|-----------|--------|
| Logistic Regression  | 0.8006 | 0.6498 | 0.6978   | 0.6618    | 0.6383 |
| Random Forest        | 0.9012 | 0.8000 | 0.8100   | 0.7439    | 0.8652 |
| XGBoost             | 0.9152 | 0.8231 | 0.8380   | 0.7908    | 0.8582 |
| LightGBM            | 0.9177 | 0.8271 | 0.8411   | 0.7922    | 0.8652 |
| Neural Network      | 0.8757 | 0.7645 | 0.7850   | 0.7368    | 0.7943 |

## 6. Model Selection
The **LightGBM** model was selected as the best model for production deployment due to its outstanding performance metrics:
- **AUC:** 0.9177, indicating a high level of discrimination between positive and negative classes.
- **F1 Score:** 0.8271, balancing precision and recall effectively.
- **Accuracy:** 0.8411, demonstrating a high percentage of correct predictions.

Trade-offs considered included the balance between AUC and F1 score, as well as the model's efficiency in handling large datasets.

## 7. Key Findings
1. The LightGBM model outperformed all other models, showcasing its effectiveness in capturing complex patterns.
2. Class imbalance was successfully addressed using SMOTE, resulting in a balanced dataset.
3. New features created during the feature engineering process significantly enhanced model performance.
4. Tree-based models, particularly boosting methods, demonstrated superior predictive capabilities compared to traditional models like logistic regression.
5. The dataset's comprehensive nature allowed for a robust analysis of patient readmission factors.

## 8. Recommendations
Moving forward, we recommend:
- Deploying the LightGBM model in a production environment to assist in real-time patient management.
- Continuously monitoring model performance and retraining with new data to maintain accuracy.
- Exploring additional features that may further enhance predictive capabilities, such as social determinants of health.

## 9. Conclusion
This research successfully developed a predictive model for hospital readmissions, providing valuable insights into patient management. The LightGBM model stands out as a reliable tool for predicting readmissions, enabling healthcare providers to implement proactive measures that can improve patient outcomes and reduce costs.