import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.neural_network import MLPClassifier


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("diabetes.csv")

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["Outcome"].value_counts())


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# 5. Build ANN model
# --------------------------------------------------

model = MLPClassifier(
    hidden_layer_sizes=(12, 8),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)


# --------------------------------------------------
# 6. Train model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 8. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 9. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 10. Save confusion matrix visualization
# --------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes", "Diabetes"]
)

disp.plot()

plt.title("Diabetes Prediction - Confusion Matrix")
plt.tight_layout()

plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches="tight")

plt.show()

plt.close()


# --------------------------------------------------
# 11. Save ROC curve
# --------------------------------------------------

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("Diabetes Prediction - ROC Curve")
plt.tight_layout()

plt.savefig("results/roc_curve.png", dpi=300, bbox_inches="tight")

plt.show()

plt.close()