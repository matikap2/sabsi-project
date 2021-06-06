import numpy as np
import Activation as ac


class NewerNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, std = 1e-4):
        np.random.seed(1)

        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2']= np.random.randn(hidden_size, output_size) * std
        self.params['b2'] = np.zeros(output_size)

    @staticmethod
    def _linear_forward(self, A_prev, W, b):
        Z = np.dot(A_prev, W) + b
        return Z

    @staticmethod
    def _linear_forward_activation(self, A_prev, W, b, activation: ac.Activation):
        Z = self._linear_forward(A_prev, W, b)
        A_curr = activation.normal(Z)
        return A_curr, Z

    @staticmethod
    def _forward_pass(self, X_input, params):
        # Unpack params dictionary
        W1, b1 = params['W1'], params['b1']
        W2, b2 = params['W2'], params['b2']

        # Perform forward pass
        # 1) Fully connected layer + ReLu activation function
        # 2) Fully connected layer + Softmax activation function
        layers = {}
        layers['fc1_relu'], layers['fc1'] = self._linear_forward_activation(X_input, W1, b1, ac.ReLuActivation())
        layers['fc2_softmax'], layers['fc2'] = self._linear_forward_activation(fc1_relu, W2, b2, ac.SoftmaxActivation())

        return layers

    @staticmethod
    def _cross_entropy_loss(self, X_input, y_output, params, layers, reg=0.0):
        # Unpack params dictionary
        W1, W2 = params['W1'], params['W2']

        # Get size of input data
        N = X_input.shape[0]

        # Calculate
        probabilities = np.copy(layers['fc2_softmax'])
        loss = np.sum(-np.log(probabilities[np.arange(N), y_output]))
        loss /= N
        loss += 0.5 * reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

        return loss

    @staticmethod
    def _backpropagate(self, X_input, y_output, params, layers, reg=0.0):
        # Unpack params dictionary
        W1, b1 = params['W1'], params['b1']
        W2, b2 = params['W2'], params['b2']

        # Get size of input data
        N = X_input.shape[0]

        # Gradient on scores (fc2_softmax)
        dscores = np.copy(layers['fc2_softmax'])
        dscores[np.arange(N), y_output] -= 1
        dscores /= N
        
        grads = {}

        # Gradient W2
        grads['W2'] = np.dot(layers['fc1_relu'].T, dscores)

        # Gradient b2
        grads['b2'] = np.sum(dscores, axis=0)

        # Gradient W1
        dhidden = np.dot(dscores, W2.T)
        dhidden[layers['fc1_relu'] <= 0] = 0
        grads['W1'] = np.dot(X_input.T, dhidden)

        # Gradient b1
        grads['b1'] = np.sum(dhidden, axis=0)

        # Reguarization
        grads['W2'] += reg * W2
        grads['W1'] += reg * W1

        return grads
    
    def loss(self, X_input, y_output=None, reg=0.0):
        N, D = X_input.shape

        # Forward pass, class scores for input
        layers = self._forward_pass(X_input, self.params)

        # Cross-entropy loss (Softmax loss) and regularization
        loss = self._cross_entropy_loss(X_input, y_output, self.params, layers, reg)

        # Backpropagation
        grads = self._backpropagate(X_input, y_output, self.params, layers, reg)

        return loss, grads

    def train(self, X_train, y_train, X_validation, y_validation,
                learning_rate=1e-3, learning_rate_decay=0.95,
                reg=1e-5, num_iters=100,
                batch_size=200, verbose=False):
        # TODO
        pass

    def predict(self, X_input):
        # Predict score for X_input using loaded weights
        layers = self._forward_pass(X_input, self.params)
        y_prediction = np.argmax(layers['fc2'], axis=1)

        return y_prediction

    def save_model(self):
        # TODO: Add saving to file
        return self.params

    def load_model(self, params):
        # TODO: Add loading from file
        self.params = params