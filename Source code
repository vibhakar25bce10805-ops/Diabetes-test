import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve)

# 1. Load Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness","Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=columns)
print("Shape:", df.shape)
print(df.head())

# 2. Preprocess 
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)
df.fillna(df.median(), inplace=True)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Train / Test Split 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature Scaling 
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 5. Train Model 
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
model.fit(X_train_sc, y_train)

# 6. Evaluate 
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]
print("\n── Model Performance ──")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Visualize 
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"], ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
axes[1].plot([0, 1], [0, 1], color="navy", linestyle="--")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

# Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", ax=axes[2], color="steelblue")
axes[2].set_title("Feature Importance")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("diabetes_results.png", dpi=150)
plt.show()

# 8. Predict a Single Patient
def predict_patient(data: dict) -> None:
    """
    data keys: Pregnancies, Glucose, BloodPressure, SkinThickness,
               Insulin, BMI, DiabetesPedigreeFunction, Age
    """
    sample = pd.DataFrame([data])[X.columns]
    sample_sc = scaler.transform(sample)
    pred  = model.predict(sample_sc)[0]
    prob  = model.predict_proba(sample_sc)[0][1]
    label = "Diabetic" if pred == 1 else "Non-Diabetic"
    print(f"\nPrediction : {label}")
    print(f"Probability: {prob:.2%}")

# Example usage
predict_patient({"Pregnancies": 2,"Glucose": 148,"BloodPressure": 72,"SkinThickness": 35,"Insulin": 0,"BMI": 33.6,"DiabetesPedigreeFunction": 0.627,"Age": 50})
