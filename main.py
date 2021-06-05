import numpy as np
from DatasetLoader import DatasetLoader
from ImageLoader import ImageLoader
import NeuralNetwork

DATASET_FOLDER = 'sabsi-project/datasets/'
DATASET_COLORS = 'colors.csv'
TEST_IMAGE = 'test.jpg'
DATASET_COLORS_LABELS = ('Red', 'Green', 'Blue', 
                        'Yellow', 'Orange', 'Pink', 
                        'Purple', 'Brown', 'Grey', 
                        'Black', 'White') 
 


def main():
    data = DatasetLoader(DATASET_FOLDER + DATASET_COLORS)
    image = ImageLoader(DATASET_FOLDER + TEST_IMAGE)
    #image.print_rgb_data()

    #test
    X = np.array([[1,1,1],
                [1,1,0],
                [0,0,0],
                [1,0,1],
                [0,0,1],
                [0,1,0]])
    y = np.array([[1],[1],[0],[0],[0],[1]])
    NN = NeuralNetwork.NeuralNetwork(X, y)
    for i in range (1500):
        NN.feedforward()
        NN.backprop()

    print(NN.output)
    #eotest
    return 0

if __name__ == "__main__":
    main()


# https://github.com/PacktPublishing/Neural-Network-Projects-with-Python/blob/master/Chapter01/train_neural_network_from_scratch.py
