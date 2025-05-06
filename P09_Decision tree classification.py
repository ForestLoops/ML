import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load data
X, y = load_iris(return_X_y=True)
labels = load_iris().target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier().fit(X_train, y_train)
y_pred, y_proba = model.predict(X_test), model.predict_proba(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:\n", (cm := confusion_matrix(y_test, y_pred)))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba, multi_class='ovr'))

# Confusion Matrix Plot
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.colorbar()
plt.xticks(range(3), labels, rotation=45)
plt.yticks(range(3), labels)
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.xlabel('Predicted'), plt.ylabel('True')
plt.tight_layout(), plt.show()

# ROC Curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
plt.figure()
colors = ['blue', 'red', 'green']
for i, color in enumerate(colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, color=color, label=f'{labels[i]} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - Decision Tree')
plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
plt.legend(), plt.grid(True), plt.tight_layout(), plt.show()

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=10)
print("CV Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Cross-validation Scores Plot
plt.bar(range(1, 11), cv_scores, color='cyan')
plt.axhline(np.mean(cv_scores), color='black', linestyle='--', label=f'Mean: {np.mean(cv_scores):.2f}')
plt.title('CV Accuracy - Decision Tree')
plt.xlabel('Fold'), plt.ylabel('Accuracy')
plt.legend(), plt.grid(True), plt.tight_layout(), plt.show()

# Decision Tree Visualization
plt.figure(figsize=(18, 12))
plot_tree(model, feature_names=load_iris().feature_names, class_names=labels, filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Structure (Iris Dataset)')
plt.show()
