import Activation as ac
import numpy as np


'''
class NeuralNetwork:
    def __init__(self, x, y):
        self.layer1_activation = ac.ReLuActivation()
        self.layer2_activation = ac.SigmoidActivation()

        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
    
    def feedforward(self):
        self.layer1 = self.layer1_activation.normal(np.dot(self.input, self.weights1))
        self.output = self.layer2_activation.normal(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.layer2_activation.derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.layer1_activation.derivative(self.output), self.weights2.T) *
                                            self.layer1_activation.derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
'''


class NeuralNetwork:
    def __init__(self, x, y): 
        #relu 3, relu 90, relu 90, relu 40, relu 24, softmax 11
        self.layer1_activation = ac.ReLuActivation()
        self.layer2_activation = ac.ReLuActivation()
        self.layer3_activation = ac.ReLuActivation()
        self.layer4_activation = ac.ReLuActivation()
        self.layer5_activation = ac.ReLuActivation()
        self.layer6_activation = ac.SigmoidActivation()

        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],3) 
        self.weights2   = np.random.rand(3,90)  
        self.weights3   = np.random.rand(90,90)
        self.weights4   = np.random.rand(90,40)
        self.weights5   = np.random.rand(40,24)               
        self.weights6   = np.random.rand(24,11)
        self.y          = y
        self.output     = np.zeros(self.y.shape)
    
    def feedforward(self):
        self.layer1 = self.layer1_activation.normal(np.dot(self.input, self.weights1))
        self.layer2 = self.layer2_activation.normal(np.dot(self.layer1, self.weights2))
        self.layer3 = self.layer3_activation.normal(np.dot(self.layer2, self.weights3))
        self.layer4 = self.layer4_activation.normal(np.dot(self.layer3, self.weights4))
        self.layer5 = self.layer5_activation.normal(np.dot(self.layer4, self.weights5))
        self.output = self.layer6_activation.normal(np.dot(self.layer5, self.weights6))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights6 = np.dot(self.layer5.T, (2*(self.y - self.output) * self.layer6_activation.derivative(self.output)))

        d_weights5 = np.dot(self.layer4.T, (np.dot(2*(self.y - self.output) * self.layer5_activation.derivative(self.output), self.weights6.T) *
                                            self.layer5_activation.derivative(self.layer5)))

        d_weights4 = np.dot(self.layer3.T, (np.dot(2*(self.y - self.output) * self.layer4_activation.derivative(self.output), self.weights5.T) *
                                            self.layer4_activation.derivative(self.layer4))) #todo zagnieździć

        d_weights3 = np.dot(self.layer2.T, (np.dot(2*(self.y - self.output) * self.layer3_activation.derivative(self.output), self.weights4.T) *
                                            self.layer3_activation.derivative(self.layer3))) #todo zagnieździć

        d_weights2 = np.dot(self.layer1.T, (np.dot(2*(self.y - self.output) * self.layer2_activation.derivative(self.output), self.weights3.T) *
                                            self.layer2_activation.derivative(self.layer2))) #todo zagnieździć

        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.layer1_activation.derivative(self.output), self.weights2.T) *
                                            self.layer1_activation.derivative(self.layer1))) #todo zagnieździć

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3
        self.weights4 += d_weights4
        self.weights5 += d_weights5
        self.weights6 += d_weights6
