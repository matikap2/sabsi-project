import tensorflow as tf
import numpy as np
import datetime
from DatasetLoader import DatasetLoader
from ImageLoader import ImageLoader

if __name__ == "__main__":
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
    y_image = tf.keras.utils.to_categorical(np.array(imported_colour), len(DATASET_COLORS_LABELS))

    X_train = np.array(X_in[:9000])
    y_train = np.array(y_out[:9000])
    X_val = np.array(X_in[9000:])
    y_val = np.array(y_out[9000:])
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

    log_dir = "sabsi-project/logs-tensorflow/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(X_train, 
        y_train, 
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard_callback])

    model.evaluate(X_image, y_image, verbose=2)
