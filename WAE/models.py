import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from functools import partial
from config import cla

PARAMS = cla()
Act_func = PARAMS.act_function
denseblock_n = PARAMS.denseblock_n

np.random.seed(1008)

Act_param = 0.2
use_bias = True


if Act_func == "ReLU":
    activation_func = keras.layers.ReLU(negative_slope=Act_param)
elif Act_func == "ELU":
    activation_func = keras.activations.elu
elif Act_func == "tanh":
    activation_func = keras.layers.Activation("tanh")
elif Act_func == "sigmoid":
    activation_func = keras.layers.Activation("sigmoid")
elif Act_func == "sin":
    activation_func = tf.math.sin


def DenseBlock(input_X, normalization=None, reg_param=1.0e-7, n=3, input_Z=None):

    N, H, W, Nx = input_X.shape

    padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    X = input_X

    X_list = [X]

    for i in range(n):
        if i == 0:
            X = X_list[0]
        else:
            X = keras.layers.concatenate(X_list, axis=-1)
        if normalization == "ln":
            X = tf.keras.layers.LayerNormalization()(X)
        elif normalization == "bn":
            X = tf.keras.layers.BatchNormalization()(X)
        X = activation_func(X)
        X = tf.pad(X, padding, "REFLECT")
        X = keras.layers.Conv2D(
            filters=Nx,
            kernel_size=3,
            strides=1,
            padding="valid",
            kernel_regularizer=keras.regularizers.l2(reg_param),
            use_bias=use_bias,
        )(X)
        X_list.append(X)

    return X


def DownSample(
    input_X, k, downsample=True, activation=True, do_padding=True, reg_param=1.0e-7
):

    padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    if do_padding:
        X = tf.pad(input_X, padding, "REFLECT")
    else:
        X = input_X
    X = keras.layers.Conv2D(
        filters=k,
        kernel_size=3,
        strides=1,
        padding="valid",
        kernel_regularizer=keras.regularizers.l2(reg_param),
        use_bias=use_bias,
    )(X)
    if activation:
        X = activation_func(X)
    if downsample:
        X = keras.layers.AveragePooling2D(pool_size=2, strides=2)(X)

    return X


def UpSample(
    input_X, k, upsample=True, activation=True, do_padding=True, reg_param=1.0e-7
):

    padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    X = input_X
    if do_padding:
        X = tf.pad(X, padding, "REFLECT")
    X = keras.layers.Conv2D(
        filters=k,
        kernel_size=3,
        strides=1,
        padding="valid",
        kernel_regularizer=keras.regularizers.l2(reg_param),
        use_bias=use_bias,
    )(X)
    if activation:
        X = activation_func(X)

    if upsample:
        X = keras.layers.UpSampling2D(size=2)(X)

    return X


def encoder(
    Input_W=32,
    Input_H=32,
    Input_C=4,
    latent_dim=100,
    k0=36,
    reg_param=1.0e-7,
    norm_type="ln",
    denseblock_size=denseblock_n,
):

    # We are asuming that X and Y have the same dimensions and same number of chanels
    input_XY = keras.Input(shape=(Input_W, Input_H, Input_C))

    # Downsampling + ResBlock
    X0 = DownSample(
        input_X=input_XY, downsample=False, k=k0, do_padding=True, reg_param=reg_param
    )
    X0 = DenseBlock(
        input_X=X0, normalization=norm_type, reg_param=reg_param, n=denseblock_size
    )

    X1 = DownSample(input_X=X0, k=k0, do_padding=True, reg_param=reg_param)
    X1 = DenseBlock(
        input_X=X1, normalization=norm_type, reg_param=reg_param, n=denseblock_size
    )

    X2 = DownSample(input_X=X1, k=2 * k0, do_padding=True, reg_param=reg_param)
    X2 = DenseBlock(
        input_X=X2, normalization=norm_type, reg_param=reg_param, n=denseblock_size
    )

    X3 = DownSample(input_X=X2, k=4 * k0, do_padding=True, reg_param=reg_param)
    X3 = DenseBlock(
        input_X=X3, normalization=norm_type, reg_param=reg_param, n=denseblock_size
    )

    X4 = DownSample(input_X=X3, k=8 * k0, do_padding=True, reg_param=reg_param)
    X4 = DenseBlock(
        input_X=X4, normalization=norm_type, reg_param=reg_param, n=denseblock_size
    )

    X5 = keras.layers.Flatten()(X4)

    X5 = keras.layers.Dense(
        units=latent_dim * 3,
        activation=activation_func,
        kernel_regularizer=keras.regularizers.l2(reg_param),
        use_bias=use_bias,
    )(X5)

    X6 = keras.layers.Dense(
        units=latent_dim,
        activation=activation_func,
        kernel_regularizer=keras.regularizers.l2(reg_param),
        use_bias=use_bias,
    )(X5)
    X7 = keras.layers.BatchNormalization()(X6)
    X8 = keras.layers.Dense(
        units=latent_dim,
        activation=None,
        kernel_regularizer=keras.regularizers.l2(reg_param),
        use_bias=use_bias,
    )(X7)

    model = keras.Model(inputs=input_XY, outputs=X8)

    return model


def decoder(
    latent_dim=100,
    cnn_init_dim=(2, 2, 512),
    output_channels=4,
    reg_param=1.0e-7,
    norm_type="bn",
    denseblock_size=denseblock_n,
):

    input_z = keras.Input(shape=(latent_dim))

    units_init = cnn_init_dim[0] * cnn_init_dim[1] * cnn_init_dim[2]

    X1 = tf.keras.layers.Dense(units=units_init)(input_z)

    X2 = tf.keras.layers.Reshape(target_shape=cnn_init_dim)(X1)

    X3 = UpSample(
        input_X=X2,
        k=int(cnn_init_dim[2] / 2),
        upsample=True,
        activation=True,
        do_padding=True,
        reg_param=reg_param,
    )
    X3 = DenseBlock(
        input_X=X3, normalization=norm_type, reg_param=reg_param, n=denseblock_size
    )

    X4 = UpSample(
        input_X=X3,
        k=int(cnn_init_dim[2] / 4),
        upsample=True,
        activation=True,
        do_padding=True,
        reg_param=reg_param,
    )
    X4 = DenseBlock(
        input_X=X4, normalization=norm_type, reg_param=reg_param, n=denseblock_size
    )

    X5 = UpSample(
        input_X=X4,
        k=int(cnn_init_dim[2] / 8),
        upsample=True,
        activation=True,
        do_padding=True,
        reg_param=reg_param,
    )
    X5 = DenseBlock(
        input_X=X5, normalization=norm_type, reg_param=reg_param, n=denseblock_size
    )

    X6 = UpSample(
        input_X=X5,
        k=int(cnn_init_dim[2] / 16),
        upsample=True,
        activation=True,
        do_padding=True,
        reg_param=reg_param,
    )
    X6 = DenseBlock(
        input_X=X6, normalization=norm_type, reg_param=reg_param, n=denseblock_size
    )

    X7 = UpSample(
        input_X=X6,
        k=output_channels,
        upsample=False,
        activation=False,
        do_padding=True,
        reg_param=reg_param,
    )

    X7 = tf.keras.activations.sigmoid(X7)

    model = tf.keras.Model(inputs=input_z, outputs=X7)

    return model


class WAE(tf.keras.Model):
    """Convolutional wasserstein autoencoder."""

    def __init__(self, latent_dim):
        super(WAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder(latent_dim=latent_dim)
        self.decoder = decoder(latent_dim=latent_dim)

    @tf.function
    def sample(self, n_size=10, z=None):
        if z is None:
            z = tf.random.normal(shape=(n_size, self.latent_dim))
        return self.decode(z)

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction
