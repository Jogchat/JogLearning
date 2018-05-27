import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam

WIDTH, HEIGHT = 28, 28


def alexNet():
    model = Sequential()
    model.add(Conv2D(8, (11, 11), padding="same", input_shape=(HEIGHT, WIDTH, 1), activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = alexNet()
    model.summary()
    model.save_weights("model.h5")
    
    ## Loading data set
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_train, n_test = len(x_train), len(x_test)
    n_cls = 10
    y_train = np_utils.to_categorical(y_train, n_cls)
    y_test = np_utils.to_categorical(y_test, n_cls)

    x_train = x_train.reshape(n_train, HEIGHT, WIDTH, 1).astype('float32')/255.
    x_test = x_test.reshape(n_test, HEIGHT, WIDTH, 1).astype('float32')/255.
    print(x_train.shape)
    print(y_train.shape)

