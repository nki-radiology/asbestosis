# -*- coding: utf-8 -*-
"""
@author: Kevin Groot Lipman
"""

from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, LeakyReLU, Lambda, Dense, Flatten, Reshape
from keras.models import Model
from keras_contrib.layers.convolutional.subpixelupscaling import SubPixelUpscaling
from keras import backend as K
from keras.losses import mse
from keras.optimizers import Adam
import keras


def sampling(arg):
    x_mean, log_var = arg
    epsilon = K.random_normal(shape=(K.shape(x_mean)), mean=0., stddev=1.0)
    return x_mean + K.exp(0.5 * log_var) * epsilon


def conv_block(x, filters, strides, use_bias, l2, subpixel=False):
    x = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=use_bias,
               kernel_regularizer=keras.regularizers.l2(l2))(x)
    x = BatchNormalization()(x) if not use_bias else x
    x = LeakyReLU()(x)
    x = SubPixelUpscaling(scale_factor=2)(x) if subpixel else x
    return x


def vae(shape=(256, 256, 1), nr_blocks=12, l2=1e-6, latent='spatial', latent_dim=8, latent_fil=8,
        annealing_weight=K.variable(0.001),
        use_bias=False, mode='train'):
    # Encoder
    i = Input(shape=shape)
    filters = latent_fil * 2
    for i in range(nr_blocks):
        if i == 0:
            x = conv_block(i, filters=filters, strides=1, use_bias=use_bias, l2=l2)
        x = conv_block(x, filters=filters, strides=2, use_bias=use_bias, l2=l2)
        filters += 16

    if latent == 'spatial':
        x_mean = Conv2D(latent_fil, (1, 1), padding='same')(x)
        log_var = Conv2D(latent_fil, (1, 1), padding='same')(x)
        if mode == 'train':
            x = Lambda(sampling, output_shape=(int(shape[0] / (shape[0] / latent_dim)),
                                               int(shape[0] // (shape[0] / latent_dim)),
                                               latent_fil))([x_mean, log_var])
        else:
            x = x_mean

    else:
        x = Flatten()(x)
        x_mean = Dense(latent_fil * latent_dim * latent_dim)(x)
        log_var = Dense(latent_fil * latent_dim * latent_dim)(x)
        if mode == 'train':
            x = Lambda(sampling, output_shape=(int(shape[0] / (shape[0] / latent_dim)),
                                               int(shape[0] // (shape[0] / latent_dim)),
                                               latent_fil))([x_mean, log_var])
        else:
            x = x_mean
        x = Dense(1536 * 4, kernel_regularizer=keras.regularizers.l2(l2))(x)
        x = Reshape([8, 8, 96])(x)

    # Decoder

    for i in range(nr_blocks):
        x = conv_block(x, filters=filters, strides=1, use_bias=use_bias, l2=l2, subpixel=True)
        filters -= 16

    o = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_regularizer=keras.regularizers.l2(l2))(x)
    model = Model(inputs=i, outputs=o)

    def lossfunc(y_true, y_pred):
        # Define loss function within the model to include latent space
        mseloss = mse(K.flatten(y_true), K.flatten(y_pred)) * shape[0] * shape[1]
        kl_loss = - annealing_weight * K.sum(1 + log_var - K.square(x_mean) - K.exp(log_var), axis=-1)
        return K.mean(mseloss + kl_loss)

    model.compile(optimizer=Adam(lr=0.0015), loss=lossfunc)
    return model
