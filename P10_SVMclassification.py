import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load data
iris = load_iris()
X, y = iris.data[:, :2], iris.target  # use only 2 features for 2D plotting
target_names = iris.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM
model = SVC(probability=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(label_binarize(y_test, classes=[0,1,2]), y_score, multi_class='ovr'))

# Confusion matrix plot
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
for i, color in zip(range(3), ['blue', 'red', 'green']):
    plt.plot(fpr[i], tpr[i], color=color, label=f'{target_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--'); plt.title('ROC Curve'); plt.legend(); plt.show()

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=10)
print("Cross-Validation Scores:", cv_scores)
plt.bar(range(1, 11), cv_scores, color='cyan')
plt.axhline(np.mean(cv_scores), color='black', linestyle='--', label=f'Mean: {np.mean(cv_scores):.2f}')
plt.title('CV Accuracy'); plt.xlabel('Fold'); plt.ylabel('Score'); plt.legend(); plt.show()

# Decision boundary
h = .02
xx, yy = np.meshgrid(np.arange(X[:, 0].min()-1, X[:, 0].max()+1, h),
                     np.arange(X[:, 1].min()-1, X[:, 1].max()+1, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', edgecolors='k', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='^', edgecolors='k', label='Test')
plt.xlabel('Sepal Length'); plt.ylabel('Sepal Width'); plt.title('SVM Decision Boundary')
plt.legend(); plt.show()