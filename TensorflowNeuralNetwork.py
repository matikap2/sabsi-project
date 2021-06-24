import tensorflow as tf
import numpy as np
from DatasetLoader import DatasetLoader

if __name__ == "__main__":
    DATASET_FOLDER = 'sabsi-project/datasets/'
    DATASET_COLORS = 'colors.csv'
    TEST_IMAGE = 'test.jpg'
    DATASET_COLORS_LABELS = ('Red', 'Green', 'Blue', 
                            'Yellow', 'Orange', 'Pink', 
                            'Purple', 'Brown', 'Grey', 
                            'Black', 'White') 

    data = DatasetLoader(DATASET_FOLDER + DATASET_COLORS)
    X_in, y_out = data.load_data()

    X_train = np.array(X_in[:8000])
    y_train = np.array(y_out[:8000])
    X_val = np.array(X_in[8000:10000])
    y_val = np.array(y_out[8000:10000])
    y_train = tf.keras.utils.to_categorical(y_train, len(DATASET_COLORS_LABELS))
    y_val = tf.keras.utils.to_categorical(y_val, len(DATASET_COLORS_LABELS))
    # X_test = np.array(X_in[10000:])
    # y_test = np.array(y_out[10000:])

    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=[3]),
    tf.keras.layers.Dense(len(DATASET_COLORS_LABELS), activation ="softmax"),
    ])

    model.compile(optimizer='sgd',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

    print(model.summary())

    model.fit(X_train, y_train, epochs=10)
    model.evaluate(X_val, y_val, verbose=2)

