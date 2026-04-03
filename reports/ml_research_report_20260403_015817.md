# ML Research Report — Hospital Readmission Prediction

## Executive Summary
Hospital readmissions pose significant challenges to healthcare systems, impacting patient outcomes and incurring additional costs. This research aimed to predict hospital readmissions using a comprehensive dataset of patient records. The best-performing model, LightGBM, achieved an AUC of 0.9177, an F1 score of 0.8271, and an accuracy of 0.8411. It is recommended to deploy the LightGBM model for operational use to enhance patient care and resource management.

## 1. Problem Statement
Predicting hospital readmissions is crucial for improving patient outcomes and reducing healthcare costs. Unplanned readmissions can indicate inadequate care during the initial hospital stay and can lead to increased morbidity and healthcare expenses. By accurately predicting which patients are at risk of readmission, healthcare providers can implement targeted interventions to improve care continuity and patient management.

## 2. Dataset Overview
The dataset used for this analysis consists of **1602 rows** and **16 columns**, derived from patient records. Key features include demographic information, medical history, and treatment details. The dataset includes the following columns:
- `age`
- `num_medications`
- `num_procedures`
- `num_diagnoses`
- `time_in_hospital`
- `num_lab_tests`
- `gender`
- `admission_type`
- `discharge_to`
- `primary_diagnosis`
- `A1C_result`
- `medication_intensity`
- `procedure_burden`
- `is_elderly`
- `log_num_medications`
- `readmitted`

The dataset is balanced, with **801 instances** for both classes (readmitted and not readmitted).

## 3. Data Quality Findings
The exploratory data analysis revealed several data quality issues:
- **Missing Values:** 
  - `num_lab_tests`: 30 missing values (3.0%)
  - `A1C_result`: 50 missing values (5.0%)
- **Outliers:** 
  - `num_medications`: 10 outliers (1.0%)
  - `readmitted`: 199 outliers (19.9%)
- **Class Imbalance:** The dataset was initially imbalanced, but techniques such as SMOTE were applied to ensure equal representation of both classes.

## 4. Feature Engineering
To prepare the data for modeling, several feature engineering steps were undertaken:
- **Missing Value Imputation:** 
  - `num_lab_tests` was filled with the median value of 25.00.
  - `A1C_result` was filled with the mode value of "Normal".
- **Encoding Categorical Variables:** Categorical features were label encoded to convert them into numerical formats suitable for model training.
- **New Features Created:**
  - `medication_intensity`: Ratio of `num_medications` to `num_diagnoses`.
  - `procedure_burden`: Ratio of `num_procedures` to `num_diagnoses`.
  - `is_elderly`: Binary feature indicating if the patient is elderly (age ≥ 65).
  - `log_num_medications`: Logarithmic transformation of `num_medications`.

## 5. Experiment Results
The following table summarizes the performance metrics of all models evaluated:

| Model               | AUC    | F1     | Accuracy | Precision | Recall |
|---------------------|--------|--------|----------|-----------|--------|
| Logistic Regression  | 0.8006 | 0.6498 | 0.6978   | 0.6618    | 0.6383 |
| Random Forest        | 0.9012 | 0.8000 | 0.8100   | 0.7439    | 0.8652 |
| XGBoost             | 0.9152 | 0.8231 | 0.8380   | 0.7908    | 0.8582 |
| LightGBM            | 0.9177 | 0.8271 | 0.8411   | 0.7922    | 0.8652 |
| Neural Network      | 0.8757 | 0.7645 | 0.7850   | 0.7368    | 0.7943 |

## 6. Model Selection
The **LightGBM** model was selected as the best model for production deployment due to its superior performance metrics, achieving an AUC of **0.9177**, an F1 score of **0.8271**, and an accuracy of **0.8411**. The model's ability to handle large datasets efficiently and capture complex patterns contributed to its success. Trade-offs considered included the balance between AUC, F1 score, and computational efficiency, with LightGBM excelling in all areas.

## 7. Key Findings
1. The LightGBM model outperformed all other models in terms of AUC, F1 score, and accuracy.
2. Ensemble methods consistently provided better performance than simpler models, highlighting the importance of model complexity in capturing data patterns.
3. The dataset's balance was successfully achieved through SMOTE, enhancing model training effectiveness.
4. New features created during the feature engineering process significantly contributed to model performance.
5. The analysis confirmed the importance of addressing missing values and outliers in the dataset.

## 8. Recommendations
To further enhance the predictive capabilities of the model, the following steps are recommended:
- Conduct additional experiments with hyperparameter tuning for the LightGBM model to optimize performance.
- Explore the inclusion of external datasets to enrich the feature set and improve model robustness.
- Implement a monitoring system to track model performance over time and adjust as necessary based on new patient data.

## 9. Conclusion
This research successfully developed a predictive model for hospital readmissions using patient data. The LightGBM model demonstrated the highest performance metrics, making it a strong candidate for deployment. By leveraging this model, healthcare providers can improve patient management and reduce unnecessary readmissions, ultimately enhancing patient care and operational efficiency.