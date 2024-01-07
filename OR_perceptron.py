import numpy as np

# Define the OR input and corresponding output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])  # OR truth table

# Define the perceptron parameters (weights and bias)
weights = np.random.rand(2) - 0.5  # Initialize weights randomly
bias = np.random.rand(1) - 0.5  # Initialize bias randomly

print('Starting Weights: ', weights)
print('Starting Bias: ', bias)

# Set the learning rate
learning_rate = 0.01

# Define the activation function (step function for a simple perceptron)
def activate(x):
    return 1 if x > 0 else 0

# Train the perceptron
epochs = 10000
for epoch in range(epochs):
    for i in range(len(X)):
        # Forward pass (calculate the weighted sum and apply activation)
        weighted_sum = np.dot(X[i], weights) + bias
        output = activate(weighted_sum)

        # Update weights and bias based on the error
        error = y[i] - output
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

# Test the trained perceptron
print("Trained Weights:", weights)
print("Trained Bias:", bias)

# Predict outputs for OR inputs
predictions = [activate(np.dot(x, weights) + bias) for x in X]
print('OR inputs: ', [list(b) for b in X])
print('Predictions:', predictions)
print('Truth Table: ', list(y))