
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Define the Convolutional Neural Network
class CNN:
    def __init__(self):
        self.weights = None
        self.bias = None

    def initialize_parameters(self):
        # Initialize weights and bias
        self.weights = np.random.randn(16, 3, 3)
        self.bias = np.zeros((16, 1))

    def conv2d(self, image):
        # Perform convolution operation
        conv_output = np.zeros((16, image.shape[1]-2, image.shape[2]-2))
        for i in range(16):
            for j in range(image.shape[1]-2):
                for k in range(image.shape[2]-2):
                    conv_output[i, j, k] = np.sum(image[:, j:j+3, k:k+3] * self.weights[i]) + self.bias[i]
        return conv_output

    def relu(self, x):
        # Apply ReLU activation
        return np.maximum(0, x)

    def max_pool2d(self, x):
        # Perform max pooling
        pool_output = np.zeros((x.shape[0], x.shape[1]//2, x.shape[2]//2))
        for i in range(x.shape[0]):
            for j in range(0, x.shape[1], 2):
                for k in range(0, x.shape[2], 2):
                    pool_output[i, j//2, k//2] = np.max(x[i, j:j+2, k:k+2])
        return pool_output

    def flatten(self, x):
        # Flatten the input
        return x.reshape(x.shape[0], -1)

    def linear(self, x):
        # Perform linear transformation
        return np.dot(x, self.weights.reshape(16 * 30 * 30, 2)) + self.bias.reshape(2,)

    def forward(self, x):
        # Forward pass through the network
        conv_output = self.conv2d(x)
        relu_output = self.relu(conv_output)
        pool_output = self.max_pool2d(relu_output)
        flattened_output = self.flatten(pool_output)
        linear_output = self.linear(flattened_output)
        return linear_output

# Load the trained model
with open('model.pkl', 'rb') as file:
    my_NN = pickle.load(file)

# Process and classify images
for i in range(1, 11):
    with open(f'parts/cifar10/train/airplane/{i:04d}.png', 'rb') as f:
        image = plt.imread(f)  # Stores the image as a 3D numpy array

        image = image.transpose(2, 0, 1)  # Transpose the image to match the shape of the weights

        image = image.reshape(3, 32, 32)  # Reshape the image to match the shape of the weights

        image = image / 255  # Normalize the image


        output = my_NN.forward(image)  # Perform forward pass through the network
        print(output)
