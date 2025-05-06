import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load data
X, y = load_iris(return_X_y=True)
labels = load_iris().target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train & Predict
model = GaussianNB().fit(X_train, y_train)
y_pred, y_proba = model.predict(X_test), model.predict_proba(X_test)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:\n", (cm := confusion_matrix(y_test, y_pred)))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba, multi_class='ovr'))

# Plot Confusion Matrix
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(range(3), labels, rotation=45)
plt.yticks(range(3), labels)
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.xlabel('Predicted'), plt.ylabel('True')
plt.tight_layout(), plt.show()

# ROC Curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
colors = ['blue', 'red', 'green']
plt.figure()
for i, color in enumerate(colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, color=color, label=f'{labels[i]} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - Multi-class')
plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
plt.legend(), plt.grid(True), plt.tight_layout(), plt.show()

# Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=10)
print("CV Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

plt.plot(range(1, 11), cv_scores, 'o--', color='purple')
plt.title('Cross-Validation Scores')
plt.xlabel('Fold'), plt.ylabel('Accuracy')
plt.ylim(0.8, 1.05)
plt.grid(True), plt.tight_layout(), plt.show()
