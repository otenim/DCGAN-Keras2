from keras.models import Model
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Reshape, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D

def create_generater():
    input = Input(shape=(100,))
    x = Dense(1024, activation='relu')(input)
    x = BatchNormalization()(x)
    x = Dense(128*7*7, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Reshape((7,7,128))(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (5,5), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(1, (5,5), activation='tanh', padding='same')(x)
    model = Model(input, x)
    return model

def create_discriminater():
    input = Input(shape=(28,28,1))
    x = Conv2D(64, (5,5), strides=(2,2), padding='same')(input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input, x)
    return model
