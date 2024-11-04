import logging
import time

import keras
from keras.api.datasets import mnist
from keras.api.models import Model
from keras.api.layers import Input, Dense, Dropout
from keras.api.utils import to_categorical

from utils.Timer import Timer

batch_size = 128
num_epochs = 50
hidden_size = 512


num_train_samples = 60000
num_test_samples = 10000

height = 28
width = 28
depth = 1
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(num_train_samples, height * width * depth)
x_test = x_test.reshape(num_test_samples, height * width * depth)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Нормализация
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def create_model() -> Model:
    model = keras.Sequential()
    model.add(keras.Input(shape=(width * height,)))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_model(model: Model, x_train, y_train, batch_size, epochs, x_test, y_test, name: str) -> Model:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1
    )
    model.save_weights(f'{len(model.layers)}-{name}-{epochs}-{hidden_size}.weights.h5')
    print("Выводим результат обучения...")
    model.evaluate(x_test, y_test, verbose=1)
    return model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    model = create_model()
    with Timer():
        train_model(model, x_train, y_train, batch_size, num_epochs, x_test, y_test, name='mlp')