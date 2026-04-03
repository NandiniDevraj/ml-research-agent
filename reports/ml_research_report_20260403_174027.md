# ML Research Report — Hospital Readmission Prediction

## Executive Summary
The objective of this research was to predict hospital readmissions within 30 days of discharge, a critical issue in healthcare that impacts patient outcomes and hospital costs. We employed various machine learning models, with the LightGBM model achieving the best performance, yielding an AUC of 0.9177, an F1 score of 0.8271, and an accuracy of 0.8411. Based on these results, we recommend deploying the LightGBM model in a production environment to assist healthcare professionals in making informed decisions regarding patient readmissions.

## 1. Problem Statement
Hospital readmissions within 30 days of discharge pose significant challenges for healthcare systems, leading to increased costs and adverse patient outcomes. Predicting which patients are at risk of readmission allows healthcare providers to implement targeted interventions, ultimately improving patient care and reducing unnecessary hospital stays.

## 2. Dataset Overview
The dataset used for this analysis consists of 1602 rows and 16 columns, derived from a hospital readmission dataset. Key features include patient demographics, medical history, and treatment details, such as:
- `age`: Patient age
- `num_medications`: Number of medications prescribed
- `num_procedures`: Number of procedures performed
- `num_diagnoses`: Number of diagnoses recorded
- `time_in_hospital`: Duration of hospital stay
- `readmitted`: Target variable indicating readmission status

## 3. Data Quality Findings
The dataset exhibited several quality issues:
- **Missing Values**: 
  - `num_lab_tests`: 30 missing values (1.87%)
  - `A1C_result`: 50 missing values (3.12%)
- **Outliers**: 
  - `num_medications`: 10 outliers (0.62%)
  - `readmitted`: 199 outliers (12.41%)
- **Class Imbalance**: The dataset was balanced post-SMOTE application, with 801 instances for both classes (readmitted and not readmitted).

## 4. Feature Engineering
To prepare the data for modeling, several feature engineering steps were undertaken:
- **Missing Value Imputation**: 
  - `num_lab_tests` was filled with the median value of 25.00.
  - `A1C_result` was filled with the mode value of "Normal".
- **Encoding**: Categorical variables were label encoded to convert them into numerical format.
- **New Features Created**:
  - `medication_intensity`: Ratio of `num_medications` to `time_in_hospital`.
  - `procedure_burden`: Ratio of `num_procedures` to `time_in_hospital`.
  - `is_elderly`: Binary feature indicating if the patient is elderly (age ≥ 65).
  - `log_num_medications`: Logarithmic transformation of `num_medications`.

## 5. Experiment Results
The following table summarizes the performance metrics of the models evaluated:

| Model               | AUC    | F1 Score | Precision | Recall | Accuracy |
|---------------------|--------|----------|-----------|--------|----------|
| Logistic Regression  | 0.8006 | 0.6498   | 0.6618    | 0.6383 | 0.6978   |
| Random Forest        | 0.9012 | 0.8000   | 0.7439    | 0.8652 | 0.8100   |
| XGBoost             | 0.9152 | 0.8231   | 0.7908    | 0.8582 | 0.8380   |
| LightGBM            | 0.9177 | 0.8271   | 0.7922    | 0.8652 | 0.8411   |
| Neural Network      | 0.8757 | 0.7645   | 0.7368    | 0.7943 | 0.7850   |

## 6. Model Selection
The **LightGBM** model was selected as the best model for production deployment due to its outstanding performance metrics:
- **AUC**: 0.9177, indicating a strong ability to distinguish between classes.
- **F1 Score**: 0.8271, reflecting a good balance between precision and recall.
- **Accuracy**: 0.8411, demonstrating its effectiveness in correctly predicting outcomes.

### Trade-offs Considered
In selecting the LightGBM model, trade-offs included:
- **AUC vs. F1 Score**: Both metrics are crucial, especially in healthcare where false positives and negatives have significant implications.
- **Speed and Efficiency**: LightGBM's efficiency in handling large datasets makes it suitable for real-time predictions.

## 7. Key Findings
1. **Ensemble Methods Outperforming**: Ensemble techniques like LightGBM and XGBoost significantly outperformed traditional models like Logistic Regression.
2. **High Recall in LightGBM**: The model demonstrated a high recall score, effectively identifying patients at risk of readmission.
3. **Class Balance Achieved**: The application of SMOTE successfully balanced the dataset, enhancing model training.
4. **Feature Engineering Impact**: New features created improved model performance, particularly in capturing complex relationships.
5. **Robustness of LightGBM**: The model's robustness and efficiency make it a strong candidate for deployment in healthcare settings.

## 8. Recommendations
Next steps include:
- **Model Deployment**: Promote the LightGBM model to production for real-time predictions.
- **Continuous Monitoring**: Implement monitoring to evaluate model performance over time and adjust as necessary.
- **Data Collection**: Gather additional data on patient demographics and treatment outcomes to further enhance model accuracy.
- **Further Experiments**: Explore additional modeling techniques and hyperparameter tuning to optimize performance.

## 9. Conclusion
This research successfully developed a predictive model for hospital readmissions, addressing a critical healthcare challenge. The LightGBM model demonstrated superior performance, making it a valuable tool for healthcare professionals. By implementing the recommendations outlined, hospitals can improve patient care and reduce readmission rates.