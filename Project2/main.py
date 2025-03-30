# Re-import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from CoordinateDescentLogisticRegression import CoordinateDescentLogisticRegression
from RandomCoordinateDescentLogisticRegression import RandomCoordinateDescentLogisticRegression

# Load and preprocess the Wine dataset
data = load_wine()
X = data.data
y = data.target

# Keep only the first two classes (Binary Classification)
mask = y < 2
X = X[mask]
y = y[mask]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline: Logistic Regression using Scikit-Learn
logreg = LogisticRegression(penalty='none', solver='lbfgs', max_iter=5000)
logreg.fit(X_train, y_train)
L_star = logreg.score(X_test, y_test)  # Baseline accuracy

# Run Coordinate Descent
coord_desc = CoordinateDescentLogisticRegression(learning_rate=0.1, max_iter=1000)
loss_history_cd = coord_desc.fit(X_train, y_train)

# Run Random Coordinate Descent
rand_coord_desc = RandomCoordinateDescentLogisticRegression(learning_rate=0.1, max_iter=1000)
loss_history_rcd = rand_coord_desc.fit(X_train, y_train)

# Plot Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(loss_history_cd, label="Adaptive Coordinate Descent")
plt.plot(loss_history_rcd, label="Random Coordinate Descent", linestyle="dashed")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Reduction Over Iterations")
plt.legend()
plt.show()
