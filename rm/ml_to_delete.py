import tensorflow as tf 
import numpy as np 
from tensorflow import keras
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt

os.system("clear")
print("Aj Em Elajw!")

# Load csv with data #
dataset = pd.read_csv('Projekt/colors.csv', delimiter=';')
print("csv loaded.")

#  Red, Green, Blue, Yellow, Orange, Pink, Purple, Brown, Grey, Black, and White #
dataset = pd.get_dummies(dataset, columns=['label'])
dataset.head()

# create datasets #
train_dataset = dataset#dataset.sample(frac=0.9, random_state=8)
#test_dataset = dataset.drop(train_dataset.index)
print("datasets created.")

# prepare for keras #
train_labels = pd.DataFrame([train_dataset.pop(x) for x in ['label_Red', 'label_Green', 'label_Blue', 'label_Yellow', 'label_Orange', 'label_Pink', 'label_Purple', 'label_Brown', 'label_Grey', 'label_Black', 'label_White']]).T
#test_labels = pd.DataFrame([test_dataset.pop(x) for x in ['label_Red', 'label_Green', 'label_Blue', 'label_Yellow', 'label_Orange', 'label_Pink', 'label_Purple', 'label_Brown', 'label_Grey', 'label_Black', 'label_White']]).T

##train_dataset = pd.DataFrame([train_dataset.pop(x) for x in ['red', 'green', 'blue']]).T
#print(train_dataset)
#print(train_labels)

print(train_dataset)
print(train_labels)


model = tf.keras.Sequential([
keras.layers.Dense(3, activation="relu", input_shape=[3]),
#keras.layers.Dropout(0.1),
keras.layers.Dense(90, activation="relu"),
keras.layers.Dropout(0.2),
keras.layers.Dense(90, activation="relu"),
keras.layers.Dropout(0.1),
keras.layers.Dense(40, activation="relu"),
keras.layers.Dropout(0.1),
keras.layers.Dense(24, activation="relu"),
keras.layers.Dense(11, activation ="softmax")
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['accuracy'])

model.summary()

history = model.fit(x=train_dataset, y=train_labels, 
                    validation_split=0.4,
                    epochs=500, 
                    #batch_size=2048, 
                    verbose=1,
                    #shuffle=True,   
                    #validation_data = (test_dataset, test_labels)
                    )


#model.evaluate(test_dataset, test_labels)


def draw_curves(history, key1='accuracy', ylim1=(0.8, 1.00), 
                key2='loss', ylim2=(0.0, 1.0)):
    plt.figure(figsize=(12,4))
     
    plt.subplot(1, 2, 1)
    plt.plot(history.history[key1], "r.-")
    plt.plot(history.history['val_' + key1], "g.-")
    plt.ylabel(key1)
    plt.xlabel('Epoch')
    #plt.ylim(ylim1)
    plt.legend(['train', 'test'], loc='best')
 
    plt.subplot(1, 2, 2)
    plt.plot(history.history[key2], "r.-")
    plt.plot(history.history['val_' + key2], "g.-")
    plt.ylabel(key2)
    plt.xlabel('Epoch')
    #plt.ylim(ylim2)
    plt.legend(['train', 'test'], loc='best')
    #plt.axis("auto y")
     
    plt.show()
     
draw_curves(history, key1='accuracy', ylim1=(0.7, 0.95), 
            key2='loss', ylim2=(0.0, 0.8))