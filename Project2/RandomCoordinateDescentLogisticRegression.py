# Random Feature Coordinate Descent
import numpy as np

from CoordinateDescentLogisticRegression import CoordinateDescentLogisticRegression


class RandomCoordinateDescentLogisticRegression(CoordinateDescentLogisticRegression):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        loss_history = []
        for iteration in range(self.max_iter):
            prev_weights = self.weights.copy()

            # Select a coordinate randomly
            j = np.random.randint(n_features)

            # Compute gradient for chosen coordinate
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