import os
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import tensorflow as tf
from config import cla
from models import WAE
from utils import (
    save_loss,
    get_lat_var,
    read_train_images_numpy,
    make_plots_training,
)

PARAMS = cla()

np.random.seed(PARAMS.seed_no)

print("\n ============== LAUNCHING TRAINING SCRIPT =================\n")

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# ============== Parameters ======================
n_epoch = PARAMS.n_epoch
slide_num = PARAMS.slide_num
reg_param = PARAMS.reg_param
lambda_param = PARAMS.lambda_param
z_dim = PARAMS.z_dim
batch_size = PARAMS.batch_size
n_train = PARAMS.n_train
log_iter = np.int(np.floor(n_train / batch_size))
save_suffix = PARAMS.save_suffix
dataname = PARAMS.dataname
n_out = PARAMS.sample_plots
learning_rate = PARAMS.learn_rate
act_func = PARAMS.act_function
denseblock_n = PARAMS.denseblock_n
lr_sched = PARAMS.lr_sched

print("\n --- Loading data from files\n")
datadir = "/scratch1/murgoiti/Datasets/RCD"
train_data = read_train_images_numpy(
    img_dir=datadir + "/" + dataname, N_train=n_train, batch_size=batch_size
)

print("\n --- Creating network folder \n")
savefolder = (
    "WAE_slidenum_"
    + slide_num
    + f"_zdim_{z_dim}_batchsize_{batch_size}_lambda_{lambda_param}_LR_{learning_rate}_LRsched_{lr_sched}{save_suffix}"
)
savedir = "/home1/murgoiti/exps/RCD/WAE_training/" + savefolder

if not os.path.exists(savedir):
    os.makedirs(savedir)
else:
    print("\n     *** Folder already exists!\n")

print("\n --- Saving parameters to file \n")
param_file = savedir + "/parameters.txt"
with open(param_file, "w") as fid:
    for pname in vars(PARAMS):
        fid.write(f"{pname} = {vars(PARAMS)[pname]}\n")

print("\n --- Creating GAN models\n")

wae_model = WAE(latent_dim=z_dim)

wae_optim = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.9)

glv = partial(get_lat_var, z_dim=z_dim)

# === Setting up training step with tf.function ===


@tf.function
def mmd_penalty(sample_qz, sample_pz):
    sigma2_p = 1.0
    n = batch_size
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)

    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
    dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
    distances_pz = norms_pz + tf.transpose(norms_pz) - 2.0 * dotprods_pz

    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
    dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
    distances_qz = norms_qz + tf.transpose(norms_qz) - 2.0 * dotprods_qz

    dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
    distances = norms_qz + tf.transpose(norms_pz) - 2.0 * dotprods

    C = 2.0 * z_dim * sigma2_p
    stat = 0.0
    res1 = C / (C + distances_qz)
    res1 += C / (C + distances_pz)
    res1 = tf.multiply(res1, 1.0 - tf.eye(n))
    res1 = tf.reduce_sum(res1) / (nf * nf - nf)
    res2 = C / (C + distances)
    res2 = tf.reduce_sum(res2) * 2.0 / (nf * nf)
    stat += res1 - res2
    return stat


@tf.function
def compute_loss(model, x, lambda_param):
    z = model.encode(x)
    encoded_means_sum = tf.reduce_sum(tf.math.reduce_mean(z, axis=0))
    z_prior = glv(batch_size=batch_size)
    x_recons = model.decode(z)
    reconstruction_loss = tf.math.reduce_sum(
        tf.math.pow(x - x_recons, 2), axis=[1, 2, 3]
    )
    mmd_pen = mmd_penalty(z, z_prior)
    L1_norm_fake = tf.reduce_sum(x_recons)
    return (
        tf.reduce_mean(reconstruction_loss + lambda_param * mmd_pen),
        tf.reduce_mean(reconstruction_loss),
        tf.reduce_mean(mmd_pen),
        L1_norm_fake,
        encoded_means_sum,
    )


@tf.function
def train_step(model, x, optimizer, lambda_param):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss, reconstruction_loss, KL_div, L1_norm_fake, encoded_means_sum = (
            compute_loss(model, x, lambda_param)
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, reconstruction_loss, KL_div, L1_norm_fake, encoded_means_sum


# ============ Training ==================
print("\n --- Starting training \n")
n_iters = 1
loss_log = []
reconstruction_loss_log = []
mmd_pen_log = []
L1_norm_fake_log = []
L1_norm_true_log = []
encoded_means_sum_log = []

for i in range(n_epoch):

    for true in train_data:  # Every "true" is a batch in "train_data"

        loss, reconstruction_loss, mmd_pen, L1_norm_fake, encoded_means_sum = (
            train_step(wae_model, true, wae_optim, lambda_param)
        )

        loss_log.append(loss.numpy())
        reconstruction_loss_log.append(reconstruction_loss.numpy())
        mmd_pen_log.append(mmd_pen.numpy())

        L1_norm_true = tf.math.reduce_mean(
            tf.math.reduce_sum(true, axis=tf.range(1, true.shape.ndims))
        )

        L1_norm_fake_log.append(L1_norm_fake.numpy())
        L1_norm_true_log.append(L1_norm_true.numpy())
        encoded_means_sum_log.append(encoded_means_sum.numpy())

        if n_iters % 100 == 0:
            print(
                f"     *** iter:{n_iters} ---> reconstruction_loss:{reconstruction_loss.numpy():.4e}, mmd_pen:{mmd_pen.numpy():.4e}"
            )
        n_iters += 1

    if lr_sched:
        wae_optim.lr.assign(learning_rate * np.exp(-i / 20))

    if (i == 0) or ((i + 1) % PARAMS.savefig_freq == 0):
        print("     *** Saving plots and network checkpoint")
        z = glv(batch_size=n_out)
        x_generated = wae_model.decode(z).numpy()

        x_reconstructed = wae_model.decode(wae_model.encode(true[0:5, :, :, :]))

        make_plots_training(x_generated, n_out, savedir, i, type_im="synth_generative")
        make_plots_training(
            x_reconstructed, n_out, savedir, i, type_im="synth_reconstruction"
        )
        make_plots_training(true[0:5, :, :, :], n_out, savedir, i, type_im="true")

        wae_model.encoder.save(f"{savedir}/wae_encoder_epoch_{i+1}")
        wae_model.decoder.save(f"{savedir}/wae_decoder_epoch_{i+1}")

        save_loss(loss_log, "loss", savedir, n_epoch)
        save_loss(reconstruction_loss_log, "reconstruction_loss", savedir, n_epoch)
        save_loss(mmd_pen_log, "mmd_penalty", savedir, n_epoch)
        save_loss(L1_norm_true_log, "L1_norm_true", savedir, n_epoch)
        save_loss(L1_norm_fake_log, "L1_norm_fake", savedir, n_epoch)
        save_loss(encoded_means_sum_log, "encoded_means_sum", savedir, n_epoch)


print("\n ============== DONE =================\n")
