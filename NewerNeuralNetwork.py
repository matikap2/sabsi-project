import numpy as np
import Activation as ac
import matplotlib.pyplot as plt
from DatasetLoader import DatasetLoader
from ImageLoader import ImageLoader

# Implementation based on assignments from Stanford CS231n Convolutional Neural Networks for Visual Recognition course.
# http://cs231n.stanford.edu/
# https://cs231n.github.io/

class NewerNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        np.random.seed(1)

        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * std
        self.params['b2'] = np.zeros(output_size)

     
    def _linear_forward(self, A_prev, W, b):
        Z = np.dot(A_prev, W) + b
        return Z

     
    def _linear_forward_activation(self, A_prev, W, b, activation: ac.Activation):
        Z = self._linear_forward(A_prev, W, b)
        A_curr = activation.normal(Z)
        return A_curr, Z

     
    def _forward_pass(self, X_input, params):
        # Unpack params dictionary
        W1, b1 = params['W1'], params['b1']
        W2, b2 = params['W2'], params['b2']

        # Perform forward pass
        # 1) Fully connected layer + ReLu activation function
        # 2) Fully connected layer + Softmax activation function
        layers = {}
        layers['fc1_relu'], layers['fc1'] = self._linear_forward_activation(X_input, W1, b1, ac.ReLuActivation())
        layers['fc2_softmax'], layers['fc2'] = self._linear_forward_activation(layers['fc1_relu'], W2, b2, ac.SoftmaxActivation())

        return layers

     
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

        num_train = X_train.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # Mini batch
            sample_indices = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X_train[sample_indices]
            y_batch = y_train[sample_indices]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y_batch, reg=reg)
            loss_history.append(loss)

            # Update params
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2']

            if verbose and it % 10 == 0:
                print(f'iteration {it} / {num_iters}: loss {loss}')

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_validation) == y_validation).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {'loss_history': loss_history, 'train_acc_history': train_acc_history, 'val_acc_history': val_acc_history}

    def predict(self, X_input):
        # Predict score for X_input using loaded weights
        layers = self._forward_pass(X_input, self.params)
        y_prediction = np.argmax(layers['fc2'], axis=1)

        return y_prediction

    def save_model(self, file_name=None):
        if file_name is not None:
            with open(filename, mode='wx') as f:
                pass

        return self.params

    def load_model(self, params):
        # TODO: Add loading from file
        self.params = params


def main():
    DATASET_FOLDER = 'sabsi-project/datasets/'
    DATASET_COLORS = 'colors.csv'
    DATASET_IMAGE = 'flags.jpg'
    DATASET_CLASSIFIERS = 'flags_classified.txt'

    DATASET_COLORS_LABELS = ('Red', 'Green', 'Blue', 
                            'Yellow', 'Orange', 'Pink', 
                            'Purple', 'Brown', 'Grey', 
                            'Black', 'White') 

    data = DatasetLoader(DATASET_FOLDER + DATASET_COLORS)
    X_in, y_out = data.load_data()

    image = ImageLoader(DATASET_FOLDER + DATASET_IMAGE)
    image.load_image()

    imported_colour = []
    with open(DATASET_FOLDER + DATASET_CLASSIFIERS) as file :
        imported_colour = list(map(int, list(file.read())))     

    X_image = np.array(image._rgb_data)
    y_image = np.array(imported_colour)

    X_train = np.array(X_in[:9000])
    y_train = np.array(y_out[:9000])
    X_val = np.array(X_in[9000:])
    y_val = np.array(y_out[9000:])

    nn = NewerNeuralNetwork(3, 10, len(DATASET_COLORS_LABELS))

    stats = nn.train(X_train, y_train, X_val, y_val, learning_rate=7e-3, reg=1e-6, num_iters=165, verbose=False)

    print('Final training loss: ', stats['loss_history'][-1])

    # Predict on the training set
    train_accuracy = (nn.predict(X_train) == y_train).mean()
    
    # Predict on the validation set
    val_accuracy = (nn.predict(X_val) == y_val).mean()

    print(f'train accuracy: {train_accuracy} val accuracy: {val_accuracy}')

    # Predict on picture
    picture_accuracy = (nn.predict(X_image) == y_image).mean()
    print(f'picture_accuracy: {picture_accuracy}')

    # plot the loss history
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()