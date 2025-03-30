# Coordinate Descent for Logistic Regression
import numpy as np


class CoordinateDescentLogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=1000, tol=1e-5):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, X, y, w):
        predictions = self.sigmoid(X @ w)
        return -np.mean(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        loss_history = []
        for iteration in range(self.max_iter):
            prev_weights = self.weights.copy()
            for j in range(n_features):
                # Compute gradient for coordinate j
                predictions = self.sigmoid(X @ self.weights)
                gradient = np.dot(X[:, j], predictions - y) / n_samples
                self.weights[j] -= self.learning_rate * gradient

            # Compute loss after update
            loss = self.loss(X, y, self.weights)
            loss_history.append(loss)

            # Check for convergence
            if np.linalg.norm(self.weights - prev_weights) < self.tol:
                break

        return loss_history