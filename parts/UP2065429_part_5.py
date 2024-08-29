import numpy as np
import matplotlib.pyplot as plt

#framework for the convolutional neural network
class CNN:

    def __init__(self): #initialises the weights and biases
        self.weights = np.random.randn(16, 3, 3)
        self.biases = np.zeros((16, 1))



    def relu(self, x): #relu activation function, used for the forward pass
        return np.maximum(0, x)

    def relu_der(x): #relu derivative, used for backpropagation
        return np.where(x > 0, 1, 0)



    def convolute(self, image):
        #this code runs the image through 16 filters and returns the output
        conv_output = np.zeros((16, image.shape[1]-2, image.shape[2]-2))
        for i in range(16):
            for j in range(image.shape[1]-2):
                for k in range(image.shape[2]-2):
                    conv_output[i, j, k] = np.sum(image[:, j:j+3, k:k+3] * self.weights[i]) + self.biases[i]
        return conv_output


    def flatten(self, x): #flattens the input
        return x.reshape(x.shape[0], -1)

    def linear(self, x): #linear transformation
        return np.dot(x, self.weights.reshape(16 * 30 * 30, 2)) + self.biases.reshape(2,)



    def FF(self, x): #forward pass through the entire network
        conv_output = self.convolute(x)
        relu_output = self.relu(conv_output)
        flattened_output = self.flatten(relu_output)
        linear_output = self.linear(flattened_output)
        return linear_output



    def train_nn(self, images, truth_table, learning_rate, num_epochs):
        #trains the NN
        for epoch in range(num_epochs):
            for image, label in zip(images, truth_table):
                #normalise the image
                image = image.transpose(2, 0, 1)
                image = image.reshape(3, 32, 32)
                image = image / 255 

                #Forward pass
                output = self.FF(image)

                #gradient descent
                gradients = self.GD(output, label)

                #updates weights and biases
                self.update_parameters(gradients, learning_rate)

    def GD(self, output, label):
        #work out the gradients for gradient descent
        return 2 * (output - label)

    def update_parameters(self, gradients, learning_rate):
        #update weights and bias using gradient descent
        self.weights -= learning_rate * gradients.reshape(16, 3, 3)
        self.biases -= learning_rate * np.mean(gradients, axis=(1, 2)).reshape(16, 1)

#loads the training data
images = []
truth_table = []
test_images = []
test_truth_table = []
#successfully loads the images of airplanes and dogs
for i in range(1, 101):
    with open(f'cifar10/train/airplane/{i:04d}.png', 'rb') as f:
        #converts the image to a numpy array, and stores it in a list
        image = plt.imread(f)
        images.append(image)
        #adds the truth table values corresponding to 'airplane' to a list
        label = np.array([1, 0])
        truth_table.append(label)

    with open(f'/cifar10/train/dog/{i:04d}.png', 'rb') as f:
        image = plt.imread(f)
        images.append(image)
        #adds the truth table values corresponding to 'dog' to a list
        label = np.array([0, 1])
        truth_table.append(label)

for i in range(1, 11):
    with open(f'cifar10/test/airplane/{i:04d}.png', 'rb') as f:
        #converts the image to a numpy array, and stores it in a list
        image = plt.imread(f)
        test_images.append(image)
        #adds the truth table values corresponding to 'airplane' to a list
        label = np.array([1, 0])
        test_truth_table.append(label)

    with open(f'cifar10/test/dog/{i:04d}.png', 'rb') as f:
        image = plt.imread(f)
        test_images.append(image)
        #adds the truth table values corresponding to 'dog' to a list
        label = np.array([0, 1])
        test_truth_table.append(label)



#proof that the images and truth table values have been loaded correctly
print(images)
print(truth_table,'\n')
#proof that the test images and test truth table values have been loaded correctly
print(test_images)
print(test_truth_table)


#initialises the CNN, tries to train the NN
my_NN = CNN()
my_NN.train_nn(images, truth_table, 0.01, 10) #numbers represent the learning rate and number of epochs

#tests the NN
for image, label in zip(test_images, test_truth_table):
    #normalise the image
    image = image.transpose(2, 0, 1)
    image = image.reshape(3, 32, 32)
    image = image / 255 
    #forward pass
    output = my_NN.FF(image)
    print('predicted output:', output)
    print('target output:', label)