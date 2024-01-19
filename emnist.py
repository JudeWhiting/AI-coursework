import numpy as np
import pickle
import matplotlib.pyplot as plt

def relu(x):
    #return np.maximum(0, x)
    return 1 / (1 + np.exp(-x))

def relu_der(x):
    #return np.where(x > 0, 1, 0)
    return x * (1 - x)


# Define the neural network architecture
X = 32 * 32 * 1 * 3  # Number of input features
HL1 = 128  # Number of neurons in the first hidden layer
HL2 = 64  # Number of neurons in the second hidden layer
HL3 = 32  # Number of neurons in the third hidden layer
Y = 2  # Number of output neurons

# Initialize weights for the vertices attached to the hidden layers and the output layer
HL1_weights = np.random.rand(X, HL1)
HL2_weights = np.random.rand(HL1, HL2)
HL3_weights = np.random.rand(HL2, HL3)
Y_weights = np.random.rand(HL3, Y)
HL1_biases = np.zeros((1, HL1))
HL2_biases = np.zeros((1, HL2))
HL3_biases = np.zeros((1, HL3))
Y_biases = np.zeros((1, Y))

training_data = []
for i in range(1, 51):
    with open(f'parts/cifar10/train/airplane/{i:04d}.png', 'rb') as f:
        image = plt.imread(f)  # Stores the image as a 3D numpy array

        image = image.flatten()  # Flatten the image to a 1D numpy array

        image = image / 255  # Normalize the image
        training_data.append(image)

    with open(f'parts/cifar10/train/dog/{i:04d}.png', 'rb') as f:
        image = plt.imread(f)  # Stores the image as a 3D numpy array

        image = image.flatten()  # Flatten the image to a 1D numpy array

        image = image / 255  # Normalize the image
        training_data.append(image)

target_outputs = np.array([[(i+1)%2, i%2] for i in range(len(training_data))])

# Training the neural network
learning_rate = 0.1
epochs = 100

for i in range(epochs):
    # Forward pass
    for j, inputs in enumerate(training_data):
        inputs = inputs.reshape((1, X))
        HL1_input = np.dot(inputs, HL1_weights) + HL1_biases
        HL1_output = relu(HL1_input)

        HL2_input = np.dot(HL1_output, HL2_weights) + HL2_biases
        HL2_output = relu(HL2_input)

        HL3_input = np.dot(HL2_output, HL3_weights) + HL3_biases
        HL3_output = relu(HL3_input)

        Y_input = np.dot(HL3_output, Y_weights) + Y_biases
        predicted_outputs = relu(Y_input)
        
        # Calculate the error
        error = target_outputs[j] - predicted_outputs

        # Backpropagation
        output_error = relu_der(error)
        hidden_layer3_error = np.dot(output_error, Y_weights.T) * relu_der(HL3_output)
        hidden_layer2_error = np.dot(hidden_layer3_error, HL3_weights.T) * relu_der(HL2_output)
        hidden_layer1_error = np.dot(hidden_layer2_error, HL2_weights.T) * relu_der(HL1_output)

        # Update weights and biases
        Y_weights += np.dot(HL3_output.T, output_error) * learning_rate
        Y_biases += np.sum(output_error) * learning_rate

        HL3_weights += np.dot(HL2_output.T, hidden_layer3_error) * learning_rate
        HL3_biases += np.sum(hidden_layer3_error) * learning_rate

        HL2_weights += np.dot(HL1_output.T, hidden_layer2_error) * learning_rate
        HL2_biases += np.sum(hidden_layer2_error) * learning_rate

        HL1_weights += np.dot(inputs.T, hidden_layer1_error) * learning_rate
        HL1_biases += np.sum(hidden_layer1_error) * learning_rate

for j, k in enumerate(training_data):
    new_data_point = np.array(k)
    HL1_input = np.dot(new_data_point, HL1_weights) + HL1_biases
    HL1_output = relu(HL1_input)

    HL2_input = np.dot(HL1_output, HL2_weights) + HL2_biases
    HL2_output = relu(HL2_input)

    HL3_input = np.dot(HL2_output, HL3_weights) + HL3_biases
    HL3_output = relu(HL3_input)

    Y_input = np.dot(HL3_output, Y_weights) + Y_biases
    predicted_outputs = relu(Y_input)  # No activation on the output node

    print("Input:", new_data_point)
    print("Predicted Output:", predicted_outputs)
    print("Target Output:", target_outputs[j])