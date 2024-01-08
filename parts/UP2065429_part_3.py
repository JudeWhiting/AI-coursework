import numpy as np

# Generate synthetic data for training
def generate_data(num_samples=1000):
    X = np.random.rand(num_samples, 2)
    y = np.prod(X, axis=1)
    return X, y

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Neural network model with one hidden layer
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.hidden_activation = sigmoid

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = self.hidden_activation(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output)
        return self.output

    def backward(self, X, y, learning_rate):
        output_error = y - self.output
        output_delta = output_error

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

# Generate training data
X_train, y_train = generate_data()

# Create and train the neural network
input_size = 2
hidden_size = 5
output_size = 1
learning_rate = 0.1
epochs = 1000

model = NeuralNetwork(input_size, hidden_size, output_size)

for epoch in range(epochs):
    # Forward pass
    predictions = model.forward(X_train)

    # Backward pass
    model.backward(X_train, y_train, learning_rate)

    # Calculate and print the training loss
    loss = mean_squared_error(y_train, predictions)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Test the model with new data
test_input = np.array([[0.8, 0.4]])
predicted_output = model.forward(test_input)
print(f"Predicted product of {test_input[0, 0]} and {test_input[0, 1]}: {predicted_output[0]}")
