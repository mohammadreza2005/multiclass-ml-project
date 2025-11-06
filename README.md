# Full Pipeline Practice: COVID-19 Hospital Length of Stay (LOS) Prediction

## Overview
This project focuses on building a full machine learning pipeline to predict whether COVID-19 patients will have an extended hospital stay (greater than 5 days) based on admission data. It serves as a practice exercise for designing, training, and evaluating binary classification models using scikit-learn.

The goal is to create an early-warning system that helps clinicians optimize resource allocation, such as beds, ICU capacity, isolation rooms, and staffing, especially during surges.

## Objective
Design, train, and evaluate a machine learning model that, at the moment of admission, predicts if a patient hospitalized with COVID-19 will remain in the hospital for more than 5 days.

### Target Variable
| Column | Meaning | Type |
|--------|---------|------|
| Stay_Days (alias: stayday) | 0 = stay ≤ 5 days<br>1 = stay > 5 days | Binary classification target |

## Dataset
The dataset contains 318,439 records with 18 columns, capturing patient, hospital, and admission details.

| #  | Feature                  | Description                                                                 | Suggested Treatment          |
|----|--------------------------|-----------------------------------------------------------------------------|------------------------------|
| 1  | case_id                 | Unique encounter identifier. No clinical meaning beyond indexing.           | Drop or use as index.        |
| 2  | Hospital                | Encoded ID of admitting hospital (1-32). Captures fixed effects like capacity, protocols, and resources. | Categorical (one-hot or target encoding). |
| 3  | Hospital_type           | Encoded facility type (e.g., teaching, private, government). Reflects organizational characteristics impacting LOS. | Categorical                  |
| 4  | Hospital_city           | Encoded city code of the hospital. May proxy local prevalence or regulations. | Categorical                  |
| 5  | Hospital_region         | Encoded geographic region/zone of the hospital (e.g., rural/urban clusters). | Categorical                  |
| 6  | Available_Extra_Rooms_in_Hospital | Number of spare isolation/negative-pressure rooms at admission. Higher availability can shorten delays. | Numeric (integer)            |
| 7  | Department              | Primary clinical department responsible (e.g., radiotherapy, anesthesia, gynecology, TB & Chest disease). Signals comorbidities or resources in COVID context. | Categorical                  |
| 8  | Ward_Type               | Code for ward category (letters R, S, Q, P, T, U). Often maps to cost tier and infrastructure (e.g., R=Regular, S=semi-private, Q=quarantine). | Categorical                  |
| 9  | Ward_Facility           | Facility rating within the ward (letters A-F, A=most equipped). Higher grades offer advanced monitoring, affecting stay length. | Categorical                  |
| 10 | Bed_Grade               | Numerical bed grade (1-7). Correlates with equipment level and cost. ~0.03% missing values to impute. | Numeric (ordinal)            |
| 11 | patientid               | De-identified patient code for grouping multiple admissions. For single-admission models, exclude or use for patient-aware cross-validation to avoid leakage. | Index/grouping key           |
| 12 | City_Code_Patient       | Encoded residence city of the patient. Influences LOS via socio-economic and travel factors. ~1.4% missing. | Categorical                  |
| 13 | Type of Admission       | Administrative channel: Emergency, Trauma, Urgent. Reflects acuity and pre-hospital stabilization. | Categorical                  |
| 14 | Illness_Severity        | Clinician-rated severity: Extreme, Moderate, Minor. Strong signal for LOS.  | Ordinal categorical (Extreme > Moderate > Minor) |
| 15 | Patient_Visitors        | Number of visitors recorded/allowed at admission (0-13). Proxy for social support. | Numeric                      |
| 16 | Age                     | Age bucket (0-10, 11-20, ..., 91-100). Convert to ordered category or midpoint numeric. | Ordinal/numeric              |
| 17 | Admission_Deposit       | Initial deposit in local currency. Correlates with insurance or bed class.  | Numeric (continuous)         |
| 18 | Stay_Days               | Target variable (see above).                                                | Binary                       |

## Model Building Tips
- Treat as a strict binary classification task with positive class "stay > 5 days" (Stay_Days == 1).
- Use scikit-learn exclusively for pre-processing, training, tuning, and evaluation.
- Start with Logistic Regression:
  - Employ a Pipeline with ColumnTransformer for encoding/standardization.
  - Try L1 (lasso) and L2 (ridge) regularization; use `class_weight='balanced'` for imbalance.
- Advance to tree-based models:
  - **Decision Tree**: Quick for interpretability; prune to avoid overfitting (tune max_depth, min_samples_leaf).
  - **Random Forest**: Solid baseline; tune n_estimators, max_features, min_samples_leaf.
  - **Gradient-Boosted Trees** (GradientBoostingClassifier, HistGradientBoostingClassifier, or XGBoostClassifier if allowed): Often top performer; grid-search learning_rate, n_trees, max_depth.
  - **Extra Trees** (ExtraTreesClassifier): Similar to Random Forest, often faster.

## Evaluation Metrics
Keep reporting simple with these core metrics:

| Metric          | scikit-learn Helper          | Description                                                                 |
|-----------------|------------------------------|-----------------------------------------------------------------------------|
| Accuracy       | accuracy_score              | Overall fraction of correct predictions (both classes).                     |
| Precision      | precision_score             | Proportion of flagged ">5-day stay" predictions that were actually >5 days. |
| Recall         | recall_score                | Proportion of actual >5-day stays caught by the model.                      |
| F1-score       | f1_score                    | Harmonic mean balancing precision and recall.                               |
| Confusion Matrix | confusion_matrix (visualize with ConfusionMatrixDisplay) | Raw counts of TP, FP, TN, FN to inspect errors.                             |

## Requirements
- Python 3.x
- scikit-learn (for the full pipeline)


## Usage
1. Load and preprocess the dataset (handle missing values, encode categoricals).
2. Split into train/test sets (consider patient-aware splitting via patientid).
3. Build and tune models using pipelines and grid search.
4. Evaluate on test set with the core metrics.
5. Visualize confusion matrix for insights.

For example code snippets, refer to scikit-learn documentation or implement based on the tips above.

## License
This project is for educational purposes. No specific license applied.
