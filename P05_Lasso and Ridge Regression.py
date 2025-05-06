import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load data
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    "Lasso": Lasso(),
    "Ridge": Ridge()
}
colors = {
    "Lasso": ("coral", "red", "orange"),
    "Ridge": ("mediumseagreen", "green", "blue")
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} Regression - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Cross-validation
    cv_scores = -cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    print(f"{name} Regression - CV MSE Scores: {cv_scores}")
    print(f"{name} Regression - Mean CV MSE: {cv_scores.mean():.4f}")

    # Plot: Actual vs Predicted
    plt.scatter(y_test, y_pred, alpha=0.7, color=colors[name][0])
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.title(f'{name} Regression: Actual vs Predicted')
    plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.grid(True)
    plt.show()

    # Plot: Residuals
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7, color=colors[name][1])
    plt.hlines(0, y_pred.min(), y_pred.max(), colors='black', linestyles='dashed')
    plt.title(f'{name} Regression: Residuals vs Predicted')
    plt.xlabel('Predicted'); plt.ylabel('Residuals'); plt.grid(True)
    plt.show()

    # Plot: CV MSE
    plt.plot(range(1, 11), cv_scores, marker='o', linestyle='--', color=colors[name][2])
    plt.title(f'{name} Regression: Cross-Validation MSE')
    plt.xlabel('Fold'); plt.ylabel('MSE'); plt.grid(True)
    plt.show()
