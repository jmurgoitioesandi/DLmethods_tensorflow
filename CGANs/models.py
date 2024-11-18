import os
import numpy as np
import importlib
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from functools import partial
from config import cla

PARAMS = cla()

# np.random.seed(1008)

use_bias = True


def gen_model(act_func):

    if act_func == "ELU":
        activation_func = keras.layers.ELU()
    elif act_func == "tanh":
        activation_func = keras.layers.Activation("tanh")

    input_X = keras.Input(shape=(1))
    input_Z = keras.Input(shape=(1))
    input = tf.keras.layers.concatenate([input_X, input_Z])

    X1 = keras.layers.Dense(
        units=128, activation=activation_func, kernel_regularizer=None, use_bias=True
    )(input)

    X2 = keras.layers.Dense(
        units=256, activation=activation_func, kernel_regularizer=None, use_bias=True
    )(X1)

    X3 = keras.layers.Dense(
        units=64, activation=activation_func, kernel_regularizer=None, use_bias=True
    )(X2)

    X4 = keras.layers.Dense(
        units=32, activation=activation_func, kernel_regularizer=None, use_bias=True
    )(X3)

    X10 = keras.layers.Dense(
        units=1, activation=None, kernel_regularizer=None, use_bias=None
    )(X4)

    X_out = X10

    model = keras.Model(inputs=[input_X, input_Z], outputs=X_out)

    return model


def critic_model(act_func):

    if act_func == "ELU":
        activation_func = keras.layers.ELU()
    elif act_func == "tanh":
        activation_func = keras.layers.Activation("tanh")

    input_X = keras.Input(shape=(2))

    X1 = keras.layers.Dense(
        units=128,
        activation=activation_func,
        kernel_regularizer=None,
        use_bias=use_bias,
    )(input_X)

    X2 = keras.layers.Dense(
        units=256,
        activation=activation_func,
        kernel_regularizer=None,
        use_bias=use_bias,
    )(X1)

    X3 = keras.layers.Dense(
        units=64, activation=activation_func, kernel_regularizer=None, use_bias=use_bias
    )(X2)

    X4 = keras.layers.Dense(
        units=32, activation=activation_func, kernel_regularizer=None, use_bias=use_bias
    )(X3)

    X6 = keras.layers.Dense(
        units=1, activation=None, kernel_regularizer=None, use_bias=use_bias
    )(X4)

    X_out = X6

    model = keras.Model(inputs=input_X, outputs=X_out)

    return model


@tf.function
def gradient_penalty_Adler(true_X, true_Y, fake_Y, model, p=2, obtain_LipCt=False):
    shape = tf.concat(
        (tf.shape(true_Y)[0:1], tf.tile([1], [true_Y.shape.ndims - 1])), axis=0
    )
    epsilon = tf.random.uniform(shape, 0.0, 1.0)
    y_hat = epsilon * true_Y + (1 - epsilon) * fake_Y
    total_input = tf.concat([true_X, y_hat], axis=1)
    with tf.GradientTape() as t:
        t.watch(y_hat)
        d_hat = model(tf.concat([true_X, y_hat], axis=1), training=True)
    gradients = t.gradient(d_hat, y_hat)
    ddx = tf.sqrt(1.0e-8 + tf.reduce_sum(tf.square(gradients), axis=tf.range(1, 2)))
    d_regularizer = tf.reduce_mean(tf.pow(ddx - 1.0, p))
    if obtain_LipCt:
        LipCt = calculate_LipschitzCt(model, ddx, total_input, gp_type="Adler")
    else:
        LipCt = 0
    return d_regularizer, LipCt


@tf.function
def gradient_penalty_Oberai(true_X, true_Y, fake_Y, model, p=2, obtain_LipCt=False):
    shape = tf.concat(
        (tf.shape(true_Y)[0:1], tf.tile([1], [true_Y.shape.ndims - 1])), axis=0
    )
    epsilon = tf.random.uniform(shape, 0.0, 1.0)
    y_hat = epsilon * true_Y + (1 - epsilon) * fake_Y
    total_input = tf.concat([true_X, y_hat], axis=1)
    with tf.GradientTape() as t1:
        t1.watch(total_input)
        d_total = model(total_input, training=True)
    gradients_total = t1.gradient(d_total, total_input)
    ddtot = tf.sqrt(
        1.0e-8 + tf.reduce_sum(tf.square(gradients_total), axis=tf.range(1, 2))
    )
    d_regularizer = tf.reduce_mean(tf.pow(ddtot - 1.0, p))
    if obtain_LipCt:
        LipCt = calculate_LipschitzCt(model, ddtot, total_input, gp_type="Oberai")
    else:
        LipCt = 0
    return d_regularizer, LipCt


def calculate_LipschitzCt_ins(model, N, r0, n, multiplier=1.0):
    radius = multiplier * 0.1**n
    new_r = np.random.uniform(0, radius, N)
    new_theta = np.linspace(0, 2 * np.pi, N)
    new_points = np.concatenate(
        [
            np.reshape(r0[0] + np.multiply(new_r, np.cos(new_theta)), newshape=(N, 1)),
            np.reshape(r0[1] + np.multiply(new_r, np.sin(new_theta)), newshape=(N, 1)),
        ],
        axis=1,
    )
    new_points = tf.constant(new_points)
    with tf.GradientTape() as t1:
        t1.watch(new_points)
        d_total = model(new_points, training=True)
    gradients_total = t1.gradient(d_total, new_points)
    ddtot = tf.sqrt(
        1.0e-8 + tf.reduce_sum(tf.square(gradients_total), axis=tf.range(1, 2))
    )
    max_index = tf.math.argmax(ddtot)
    return new_points[max_index], tf.math.reduce_max(ddtot)


def calculate_LipschitzCt(
    model, gradients, total_input, r0=np.array([0.0, 0.0]), gp_type="Adler"
):
    N = total_input.shape[0]
    if gp_type == "Oberai":
        max_index = tf.math.argmax(gradients)
        multiplier = 1.0
        r0 = total_input[max_index]
    else:
        r0 = r0
        multiplier = 3.5
    new_max_point, new_max_gradient = calculate_LipschitzCt_ins(
        model, 2 * N, r0, 0, multiplier
    )
    new_max_point, new_max_gradient = calculate_LipschitzCt_ins(
        model, 2 * N, new_max_point, 1
    )
    new_max_point, new_max_gradient = calculate_LipschitzCt_ins(
        model, 2 * N, new_max_point, 2
    )
    new_max_point, new_max_gradient = calculate_LipschitzCt_ins(
        model, 2 * N, new_max_point, 3
    )
    new_max_point, new_max_gradient = calculate_LipschitzCt_ins(
        model, 2 * N, new_max_point, 4
    )
    new_max_point, new_max_gradient = calculate_LipschitzCt_ins(
        model, 2 * N, new_max_point, 5
    )
    return tf.constant(new_max_gradient)


@tf.function
def D_train_step_ctransform(
    true_X, true_Y, z, G_model, D_model, D_optim, obtain_LipCt=False
):

    with tf.GradientTape() as tape:
        fake_Y = G_model([true_X, z], training=True)
        fake = tf.squeeze(tf.concat([true_X, fake_Y], axis=1))
        true = tf.squeeze(tf.concat([true_X, true_Y], axis=1))
        true_mat = tf.repeat(
            tf.reshape(true, shape=(len(true_X), 1, 2)), len(true_X), axis=1
        )
        fake_mat = tf.repeat(
            tf.reshape(fake, shape=(1, len(true_X), 2)), len(true_X), axis=0
        )
        dist_mat = tf.math.sqrt(
            1e-8 + tf.math.reduce_sum(tf.math.pow(true_mat - fake_mat, 2), axis=2)
        )
        true_XY_val = D_model(true, training=True)
        true_val_mat = tf.repeat(
            tf.reshape(true_XY_val, shape=(len(true_X), 1)), len(true_X), axis=1
        )
        fake_ctrans = tf.reshape(
            tf.math.reduce_min(dist_mat - true_val_mat, axis=0),
            shape=tf.shape(true_XY_val),
        )

        ctrans_loss = tf.reduce_mean(fake_ctrans)
        true_loss = tf.reduce_mean(true_XY_val)
        wd_loss = true_loss + ctrans_loss
        D_loss = -wd_loss

    D_gradient = tape.gradient(D_loss, D_model.trainable_variables)
    D_optim.apply_gradients(zip(D_gradient, D_model.trainable_variables))

    Lip_ct = 0
    if obtain_LipCt:
        Lip_ct = calculate_LipschitzCt(D_model, None, true_X, gp_type="ctransform")
    del tape

    return D_loss, wd_loss, Lip_ct, fake


@tf.function
def G_train_step_ctransform(true_X, true_Y, z, G_model, D_model, G_optim, mae):

    with tf.GradientTape() as tape:
        fake_Y = G_model([true_X, z], training=True)
        fake = tf.squeeze(tf.concat([true_X, fake_Y], axis=1))
        true = tf.squeeze(tf.concat([true_X, true_Y], axis=1))
        true_mat = tf.repeat(
            tf.reshape(true, shape=(len(true_X), 1, 2)), len(true_X), axis=1
        )
        fake_mat = tf.repeat(
            tf.reshape(fake, shape=(1, len(true_X), 2)), len(true_X), axis=0
        )
        dist_mat = tf.math.sqrt(
            1e-8 + tf.math.reduce_sum(tf.math.pow(true_mat - fake_mat, 2), axis=2)
        )
        true_XY_val = D_model(true, training=True)
        true_val_mat = tf.repeat(
            tf.reshape(true_XY_val, shape=(len(true_X), 1)), len(true_X), axis=1
        )
        fake_ctrans = tf.reshape(
            tf.math.reduce_min(dist_mat - true_val_mat, axis=0),
            shape=tf.shape(true_XY_val),
        )

        ctrans_loss = tf.reduce_mean(fake_ctrans)
        wd_loss = ctrans_loss
        G_loss = wd_loss

    L1_loss = mae(true_Y, fake_Y)
    gen_gradient = tape.gradient(G_loss, G_model.trainable_variables)
    G_optim.apply_gradients(zip(gen_gradient, G_model.trainable_variables))

    del tape

    return G_loss, L1_loss


@tf.function
def D_train_step(
    true_X, true_Y, z, G_model, D_model, D_optim, gp_coef, gp_type, obtain_LipCt=False
):

    with tf.GradientTape() as tape:
        fake_Y = G_model([true_X, z], training=True)
        fake = tf.squeeze(tf.concat([true_X, fake_Y], axis=1))
        true = tf.squeeze(tf.concat([true_X, true_Y], axis=1))
        fake_XY_val = D_model(fake, training=True)
        true_XY_val = D_model(true, training=True)

        if gp_type == "Oberai":
            gp, Lip_ct = gradient_penalty_Oberai(
                true_X, true_Y, fake_Y, D_model, p=2, obtain_LipCt=obtain_LipCt
            )
        elif gp_type == "Adler":
            gp, Lip_ct = gradient_penalty_Adler(
                true_X, true_Y, fake_Y, D_model, p=2, obtain_LipCt=obtain_LipCt
            )

        fake_loss = tf.reduce_mean(fake_XY_val)
        true_loss = tf.reduce_mean(true_XY_val)
        wd_loss = true_loss - fake_loss
        D_loss = -wd_loss + gp_coef * gp

        D_loss += sum(D_model.losses)

    D_gradient = tape.gradient(D_loss, D_model.trainable_variables)
    D_optim.apply_gradients(zip(D_gradient, D_model.trainable_variables))

    del tape

    return D_loss, wd_loss, gp, Lip_ct, fake


@tf.function
def G_train_step(true_X, true_Y, z, G_model, D_model, G_optim, mae):

    with tf.GradientTape() as tape:
        fake_Y = G_model([true_X, z], training=True)
        fake = tf.squeeze(tf.concat([true_X, fake_Y], axis=1))
        fake_XY_val = D_model(fake, training=True)

        G_loss = -tf.reduce_mean(fake_XY_val)

        G_loss += sum(G_model.losses)

    L1_loss = mae(true_Y, fake_Y)
    gen_gradient = tape.gradient(G_loss, G_model.trainable_variables)
    G_optim.apply_gradients(zip(gen_gradient, G_model.trainable_variables))

    del tape

    return G_loss, L1_loss
