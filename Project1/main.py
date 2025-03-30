import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist


# Load MNIST dataset
def load_mnist():
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten images and normalize to [0, 1]
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    return X_train, X_test, y_train, y_test


# Prototype selection using K-means clustering
def select_prototypes(X_train, y_train, M):
    print(f"Selecting {M} prototypes...")
    prototypes = []
    prototype_labels = []

    classes = np.unique(y_train)
    M_per_class = M // len(classes)  # Equal allocation for each class

    for cls in classes:
        print(f"Processing class {cls}...")
        X_cls = X_train[y_train == cls]

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=min(M_per_class, len(X_cls)), random_state=42)
        kmeans.fit(X_cls)

        prototypes.append(kmeans.cluster_centers_)
        prototype_labels.extend([cls] * len(kmeans.cluster_centers_))

    prototypes = np.vstack(prototypes)
    prototype_labels = np.array(prototype_labels)

    # If needed, trim excess prototypes
    if len(prototypes) > M:
        indices = np.random.choice(len(prototypes), M, replace=False)
        prototypes = prototypes[indices]
        prototype_labels = prototype_labels[indices]

    return prototypes, prototype_labels


# Custom implementation of 1-NN classifier
def knn_predict(X_train, y_train, X_test):
    """ Predict labels for X_test using 1-NN classifier with Euclidean distance """
    y_pred = np.zeros(len(X_test), dtype=y_train.dtype)

    for i, x in enumerate(X_test):
        distances = np.linalg.norm(X_train - x, axis=1)  # Compute Euclidean distance
        nearest_index = np.argmin(distances)  # Find nearest neighbor index
        y_pred[i] = y_train[nearest_index]  # Assign the label of the nearest neighbor

    return y_pred


# Evaluate performance using 1-NN classifier
def evaluate_1nn(X_train, y_train, X_test, y_test):
    print("Evaluating 1-NN classifier...")
    y_pred = knn_predict(X_train, y_train, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_mnist()

    # Define prototype sizes to evaluate
    prototype_sizes = [500, 1000, 5000, 10000, 20000]
    results = []

    # Evaluate with full training set
    print("Evaluating with full training set...")
    full_accuracy = evaluate_1nn(X_train, y_train, X_test, y_test)
    results.append(("Full Training Set", len(X_train), full_accuracy))

    # Evaluate with different prototype sizes
    for M in prototype_sizes:
        prototypes, prototype_labels = select_prototypes(X_train, y_train, M)
        prototype_accuracy = evaluate_1nn(prototypes, prototype_labels, X_test, y_test)
        results.append((f"Prototypes (M={M})", M, prototype_accuracy))

    # Save results to a text file
    with open("results.txt", "w") as f:
        f.write("Results:\n")
        for method, size, accuracy in results:
            f.write(f"{method}: Size={size}, Accuracy={accuracy:.4f}\n")

    # Plot results
    methods = [r[0] for r in results]
    sizes = [r[1] for r in results]
    accuracies = [r[2] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, accuracies, marker="o")
    plt.title("Accuracy vs Prototype Set Size")
    plt.xlabel("Number of Prototypes")
    plt.ylabel("Accuracy")
    plt.xticks(sizes, methods, rotation=45)

