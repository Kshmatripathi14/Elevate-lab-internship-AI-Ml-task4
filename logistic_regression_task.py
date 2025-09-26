import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

data = pd.read_csv("C:\\Users\\lenovo\\Downloads\\Elevate labs internship\\AI ML\\task4\\data\\breast_cancer_dataset.csv")

print("First 5 rows of dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# ==============================
# 2. Define Features & Target
# ==============================
# Change column names as per your dataset
X = data.drop("target", axis=1)   # all columns except target
y = data["target"]                # binary target column (0/1)

# ==============================
# 3. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 4. Standardize Features
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 5. Train Logistic Regression
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ==============================
# 6. Evaluation
# ==============================
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0,1], [0,1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# ==============================
# 7. Threshold Tuning Example
# ==============================
threshold = 0.3   # try values like 0.4, 0.5, 0.6 etc.
y_pred_custom = (y_prob >= threshold).astype(int)

print(f"\nConfusion Matrix at threshold {threshold}:")
print(confusion_matrix(y_test, y_pred_custom))
print(f"\nClassification Report at threshold {threshold}:\n", classification_report(y_test, y_pred_custom))
 
