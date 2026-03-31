#  Diabetes Prediction Using Machine Learning

## Overview

This project predicts whether a person is diabetic or not using a **Random Forest Classifier** trained on the Pima Indians Diabetes dataset. It demonstrates a complete machine learning workflow from data preprocessing to model evaluation and prediction.

---

##  Dataset

* Source: Pima Indians Diabetes dataset (publicly available)
* Features:

  * Pregnancies
  * Glucose
  * BloodPressure
  * SkinThickness
  * Insulin
  * BMI
  * DiabetesPedigreeFunction
  * Age
* Target:

  * **Outcome (0 = Non-Diabetic, 1 = Diabetic)**

---

##  Project Workflow

### 1. Data Preprocessing

* Replace zero values with missing values (`NaN`)
* Fill missing values using median

### 2. Train-Test Split

* 80% training and 20% testing data

### 3. Feature Scaling

* Standardization using **StandardScaler**

### 4. Model Training

* Algorithm: **Random Forest Classifier**
* Parameters:

  * `n_estimators = 100`
  * `max_depth = 6`
  * `class_weight = balanced`

### 5. Model Evaluation

* Accuracy Score
* ROC-AUC Score
* Classification Report

---

##  Visualizations

* Confusion Matrix
* ROC Curve
* Feature Importance Plot

All results are saved as:

```
diabetes_results.png
```

---

##  Prediction Function

You can predict diabetes for a new patient using:

```python
predict_patient({
  "Pregnancies": 2,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
})
```

### Output:

* Prediction: Diabetic / Non-Diabetic
* Probability Score

---

##  Requirements

Install required libraries using:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

##  How to Run

1. Clone the repository
2. Run the Python script
3. View results and generated plots
4. Use the prediction function for custom input

---

##  Conclusion

This project shows how machine learning can help in early detection of diabetes using health data. The Random Forest model provides reliable performance and useful insights.

---

##  Future Improvements

* Hyperparameter tuning
* Try advanced models (XGBoost, Neural Networks)
* Deploy as a web application
* Use larger datasets

---

##  Author

VIBHAKAR KRISHNA
