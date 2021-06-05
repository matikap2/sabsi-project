import numpy as np
import Activation as ac

class NewerNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(1)

        std = 1e-4
        self.W1 = np.random.randn(input_size, hidden_size) * std
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * std
        self.b1 = np.zeros(output_size)

    @staticmethod
    def _linear_forward(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        return Z

    @staticmethod
    def _linear_forward_activation(self, A_prev, W, b, activation: ac.Activation):
        Z = self._linear_forward(A_prev, W, b)
        A_curr = activation.normal(Z)
        return A_curr
    
    def loss(self, X_input, y_output=None, reg=0.0):
        N, D = X_input.shape

        # Forward pass, class scores for input
        fc1 = self._linear_forward_activation(X_input, self.W1, self.b1, ac.ReLuActivation())
        probabilities = self._linear_forward_activation(fc1, self.W2, self.b2, ac.SoftmaxActivation())
        
        # Cross-entropy loss and regularization
        loss = np.sum(-np.log(probabilities[np.arange(N), y_output]))
        loss /= N
        loss += 0.5 * reg * (np.sum(self.W2 * self.W2) + 
                        np.sum(self.W1 * self.W1))

        # Backpropagation
        dW1 = {}
        db1 = {}
        dW2 = {}
        db2 = {}

        # Gradient on scores (fc2)
        dscores = probabilities
        dscores[np.arange(N), y_output] -= 1
        dscores /= N

        # Gradient W2
        dW2 = np.dot(fc1.T, dscores)

        # Gradient b2
        db2 = np.sum(dscores, axis=0)

        # Gradient W1
        dhidden = np.dot(dscores, self.W2.T)
        dhidden[fc1 <= 0] = 0
        dW1 = np.dot(X_input.T, dhidden)

        # Gradient b1
        db1 = np.sum(dhidden, axis=0)

        # Reguarization
        dW2 += reg * self.W2
        dW1 += reg * self.W1

        grads = (dW1, db1, dW2, db2)

        return loss, grads