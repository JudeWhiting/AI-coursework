import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network architecture
input_size = 2  # Number of input features
hidden_layer_size = 2  # Number of neurons in the hidden layer
output_size = 1  # Number of output neurons

# Initialize weights and biases
weights_input_hidden = np.random.rand(input_size, hidden_layer_size)
biases_hidden = np.zeros((1, hidden_layer_size))

weights_hidden_output = np.random.rand(hidden_layer_size, output_size)
biases_output = np.zeros((1, output_size))


X = np.array([[np.random.rand()/2 for _ in range(2)] for _ in range(100)])  #this creates a training set of inputs
y = np.array([[i[0] + i[1]] for i in X])  

# Training the neural network
learning_rate = 0.05
epochs = 100000

for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_output
    predicted_output = sigmoid(output_layer_input)

    # Calculate the error
    error = y - predicted_output

    # Backpropagation
    output_error = error * sigmoid_derivative(predicted_output)
    hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(output_error) * learning_rate
    biases_output += np.sum(output_error) * learning_rate

    weights_input_hidden += X.T.dot(hidden_layer_error) * learning_rate
    biases_hidden += np.sum(hidden_layer_error) * learning_rate

# Test the trained neural network
new_data_point = np.array([0.3, 0.3])  # New input data
hidden_layer_input = np.dot(new_data_point, weights_input_hidden) + biases_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_output
predicted_output = sigmoid(output_layer_input)

print("Input:", new_data_point)
print("Predicted Output:", predicted_output)
