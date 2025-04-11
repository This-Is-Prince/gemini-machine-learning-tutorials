import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification # To create sample data

# --- 1. Create Sample Data ---
# Let's create 50 data points with 2 features that are mostly linearly separable
# X will be the features, y will be the class labels (0 or 1)
X, y = make_classification(n_samples=50, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           class_sep=1.5, # How separated the classes are
                           random_state=42) # For reproducible results

# Optional: Check the first 5 data points and labels
# print("Features (X first 5):\n", X[:5])
# print("Labels (y first 5):\n", y[:5])

# --- 2. Choose and Configure the Model ---
# We'll use SGDClassifier. SGD means "Stochastic Gradient Descent", a common way to train models.
# We configure it to behave like a Perceptron:
# - loss='perceptron': Use the Perceptron cost function idea.
# - learning_rate='constant': Keep the learning rate the same.
# - eta0=0.1: This is our learning rate 'α'.
# - max_iter=100: Maximum number of passes over the data (epochs).
# - penalty=None: Don't use any extra 'regularization' for now.
# - random_state=42: Ensures the results are the same each time we run it.
model = SGDClassifier(loss='perceptron', learning_rate='constant', eta0=0.1,
                      max_iter=100, penalty=None, random_state=42)

# --- 3. Train the Model ---
# This is where the learning happens! The .fit() method runs the training loop
# (like the predict -> compare -> update cycle we discussed).
print("Training the model...")
model.fit(X, y)
print("Training complete.")

# --- 4. Inspect the Learned Parameters ---
# The learned weights (w1, w2) are stored in model.coef_
# The learned bias (b) is stored in model.intercept_
weights = model.coef_
bias = model.intercept_
print(f"Learned Weights (w1, w2): {weights}")
print(f"Learned Bias (b): {bias}")

# --- 5. Make Predictions on New Data ---
# Let's pretend we have two new data points
new_data = np.array([
    [2, 2],  # Point 1
    [-1, -1] # Point 2
])

predictions = model.predict(new_data)
print(f"Predictions for new data ([2,2] and [-1,-1]): {predictions}")

# (Note: The classes here are 0 and 1, not -1 and +1, but the principle is the same.
# SGDClassifier handles this mapping internally.)