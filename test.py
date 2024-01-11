import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the neural network architecture
X = 2  # Number of input features
HL = 10  # Number of neurons in the hidden layer
Y = 1  # Number of output neurons

# Initialize weights for the vertices attached to the hidden layer and the output layer
HL_weights = np.random.rand(X, HL)
Y_weights = np.random.rand(HL, Y)
HL_biases = np.zeros((1, HL))
Y_biases = np.zeros((1, Y))

training_data = np.array([[np.random.rand() for _ in range(2)] for _ in range(1000)])  
target_outputs = np.array([[i[0] + i[1]] for i in training_data])

# Training the neural network
learning_rate = 0.1
epochs = 100

for i in range(epochs):
    # Forward pass
    for j, inputs in enumerate(training_data):
        inputs = inputs.reshape((1, 2))
        HL_input = np.dot(inputs, HL_weights) + HL_biases
        HL_output = sigmoid(HL_input)

        Y_input = np.dot(HL_output, Y_weights) + Y_biases
        predicted_outputs = Y_input  # Remove sigmoid activation on the output node

        # Calculate the error
        error = target_outputs[j] - predicted_outputs

        # Backpropagation
        output_error = error
        hidden_layer_error = np.dot(output_error, Y_weights.T) * sigmoid(HL_output) * (1 - sigmoid(HL_output))

        # Update weights and biases
        Y_weights += np.dot(HL_output.T, output_error) * learning_rate
        Y_biases += np.sum(output_error) * learning_rate

        HL_weights += np.dot(inputs.T, hidden_layer_error) * learning_rate
        HL_biases += np.sum(hidden_layer_error) * learning_rate

# Test the trained neural network
for k in [[0.1, 0.1], [0.3, 0.2], [0.4, 0.35], [0.3, 0.3], [0.8, 0.6], [0.5, 0.5]]:
    new_data_point = np.array(k)
    HL_input = np.dot(new_data_point, HL_weights) + HL_biases
    HL_output = sigmoid(HL_input)

    Y_input = np.dot(HL_output, Y_weights) + Y_biases
    predicted_outputs = Y_input  # No sigmoid activation on the output node

    print("Input:", new_data_point)
    print("Predicted Output:", predicted_outputs)
