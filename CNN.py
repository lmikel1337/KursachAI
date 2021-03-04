import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, LeakyReLU
from keras.layers.noise import AlphaDropout
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def preprocess_mnist(x_train, y_train, x_test, y_test):
    # Normalizing all images of 28x28 pixels
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # Float values for division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalizing the RGB codes by dividing it to the max RGB value
    x_train /= 255
    x_test /= 255

    # Categorical y values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test, input_shape


x_train, y_train, x_test, y_test, input_shape = preprocess_mnist(x_train, y_train, x_test, y_test)


def build_cnn(activation,
              dropout_rate,
              optimizer):
    model = Sequential()

    if (activation == 'selu'):
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation=activation,
                         input_shape=input_shape,
                         kernel_initializer='lecun_normal'))
        model.add(Conv2D(64, (3, 3), activation=activation,
                         kernel_initializer='lecun_normal'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(AlphaDropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation=activation,
                        kernel_initializer='lecun_normal'))
        model.add(AlphaDropout(0.5))
        model.add(Dense(10, activation='softmax'))
    else:
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation=activation,
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


act_func = ['leaky-relu', 'relu', 'sigmoid']
result = []

for activation in act_func:
    print('\nTraining with -->{0}<-- activation function\n'.format(activation))

    model = build_cnn(activation=activation,
                      dropout_rate=0.2,
                      optimizer=Adam())

    history = model.fit(x_train, y_train,
                        validation_split=0.20,
                        batch_size=128,  # 128 is faster, but less accurate. 16/32 recommended
                        epochs=25,
                        verbose=1,
                        validation_data=(x_test, y_test))

    result.append(history)

    K.clear_session()
    del model

print(result)

new_act_arr = act_func[1:]
new_results = result[1:]


def plot_act_func_results(results, activation_functions=[]):
    plt.figure(figsize=(10, 10))
    plt.style.use('fast')

    # Plot validation accuracy values
    for act_func in results:
        plt.plot(act_func.history['val_accuracy'])

    plt.title('Model accuracy')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Epoch')
    plt.legend(activation_functions)
    plt.show()

    # Plot validation loss values
    plt.figure(figsize=(10, 10))

    for act_func in results:
        plt.plot(act_func.history['val_loss'])

    plt.title('Model loss')
    plt.ylabel('Test Loss')
    plt.xlabel('Epoch')
    plt.legend(activation_functions)
    plt.show()


plot_act_func_results(new_results, new_act_arr)






