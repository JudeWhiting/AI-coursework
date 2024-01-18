import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_der(x):
    return np.where(x > 0, 1, 0)

# Define the neural network architecture
X = 2  # Number of input features
HL1 = 4  # Number of neurons in the first hidden layer
HL2 = 4  # Number of neurons in the second hidden layer
Y = 1  # Number of output neurons

# Initialize weights for the vertices attached to the hidden layers and the output layer
HL1_weights = np.random.rand(X, HL1)
HL2_weights = np.random.rand(HL1, HL2)
Y_weights = np.random.rand(HL2, Y)
HL1_biases = np.zeros((1, HL1))
HL2_biases = np.zeros((1, HL2))
Y_biases = np.zeros((1, Y))

training_data = np.array([[np.random.rand() for _ in range(2)] for _ in range(1000)])  
target_outputs = np.array([[i[0] + i[1]] for i in training_data])

# Training the neural network
learning_rate = 0.01
epochs = 1000

for i in range(epochs):
    # Forward pass
    for j, inputs in enumerate(training_data):
        inputs = inputs.reshape((1, 2))
        HL1_input = np.dot(inputs, HL1_weights) + HL1_biases
        HL1_output = relu(HL1_input)

        HL2_input = np.dot(HL1_output, HL2_weights) + HL2_biases
        HL2_output = relu(HL2_input)

        Y_input = np.dot(HL2_output, Y_weights) + Y_biases
        predicted_outputs = Y_input 
        
        # Calculate the error
        error = target_outputs[j] - predicted_outputs

        # Backpropagation
        output_error = error
        hidden_layer2_error = np.dot(output_error, Y_weights.T) * relu_der(HL2_output)
        hidden_layer1_error = np.dot(hidden_layer2_error, HL2_weights.T) * relu_der(HL1_output)

        # Update weights and biases
        Y_weights += np.dot(HL2_output.T, output_error) * learning_rate
        Y_biases += np.sum(output_error) * learning_rate

        HL2_weights += np.dot(HL1_output.T, hidden_layer2_error) * learning_rate
        HL2_biases += np.sum(hidden_layer2_error) * learning_rate

        HL1_weights += np.dot(inputs.T, hidden_layer1_error) * learning_rate
        HL1_biases += np.sum(hidden_layer1_error) * learning_rate

# Test the trained neural network
for k in [[0.1, 0.1], [0.3, 0.2], [0.4, 0.35], [0.3, 0.3], [0.8, 0.6], [0.5, 0.5]]:
    new_data_point = np.array(k)
    HL1_input = np.dot(new_data_point, HL1_weights) + HL1_biases
    HL1_output = relu(HL1_input)

    HL2_input = np.dot(HL1_output, HL2_weights) + HL2_biases
    HL2_output = relu(HL2_input)

    Y_input = np.dot(HL2_output, Y_weights) + Y_biases
    predicted_outputs = Y_input  # No activation on the output node

    print("Input:", new_data_point)
    print("Predicted Output:", predicted_outputs)
