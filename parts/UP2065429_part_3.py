import numpy as np

def neuron(inputs, weights, bias):
    return 1.0/(1+np.exp(-(np.dot(inputs, weights) + bias)))

X = 2
HL = [8]
Y = 1
L = [X] + HL + [Y]

weights = np.random.rand(HL[0] , X) - 0.5
biases = np.random.rand(HL[0]) - 0.5
'''
def train_neuron(inputs, target_outputs, W, B, epochs=100, learning_rate = 0.1):
    for epoch in range(epochs):
        for i in range(len(inputs)):
            error = target_outputs[i] - neuron(inputs[i], W, B)
            W += learning_rate * error * inputs[i]
            B += learning_rate * error
    return W, B
'''
def train_nn(inputs, target_outputs, W, B, epochs=100, learning_rate=0.1):
    
    for i in range (epochs):   #training loop for as many epochs as we need

        S_errors=0    #variable to carry the error we need to report to the user

        for j, input in enumerate (inputs):  #iterate through the traning data and inputs
            t=target_outputs[j]

            output = FF(input)   #use the network calculations for forward calculations. 

            e=t-output    #obtain the overall Network output error

            BP(e)      # use that error to do the back propagation 

            GD(learning_rate)     #Do gradient descent 

            S_errors+=msqe(t,output)   #update the overall error to show the user. 

def FF(inputs, target_outputs, W, B, epochs=100, learning_rate=0.1):
    out=inputs  #the input layer output is just the input
    
    #self.out[0]=x  #begin the linking of outputs to the class variable for back propagation. (begin with the input layer.

    for i, w in enumerate(W): #go through (iterate) the network layers via the weights variable

        Xnext=np.dot(out, w)    #calculate product between weights and output for the next output
        out=sigmoid(Xnext)  #use the activation function as we must per theory. 
        self.out[i+1]=out      #pass the result to the clas variable to preserve for later (when we do the back propagation. 

    return out #return the outputs of the layers. 


training_inputs = np.array([[np.random.rand()/2 for _ in range(2)] for _ in range(1000)])  #this creates a training set of inputs
targets = np.array([[i[0] * i[1]] for i in training_inputs])     #this creates a training set of outputs

data = np.array([[0,0],[0,1],[1,0],[1,1]]) #training data
OR_truth_table = np.array([0,1,1,1]) #truth table for OR



