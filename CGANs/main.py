# Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from make_plots import save_plot_histogram2d
from models import (
    gen_model,
    critic_model,
    D_train_step,
    G_train_step,
    D_train_step_ctransform,
    G_train_step_ctransform,
)
from utils import (
    generate_data_Deep_bimodal_example,
    generate_data_MGAN_example_1,
    swissroll,
    save_loss,
    calculate_errors_jointdist,
    calculate_EMD,
    sample_mollified,
)
from config import cla

PARAMS = cla()

print("\n ============== LAUNCHING TRAINING SCRIPT ================\n")

tf.config.run_functions_eagerly(True)

# ============== Parameters ======================
n_epoch = PARAMS.max_epochs
repetition = PARAMS.repetition
loss_type = PARAMS.loss_type
n_total = 1000
batch_size = n_total
dataset = PARAMS.problem
freq_metrics = 2000
save_model = True
save_histograms = False
save_metrics = True

if dataset == "bimodal":
    train_data, valid_data = generate_data_Deep_bimodal_example(
        interval=(-3.0, 3.0), N_total=n_total
    )
    test_data, valid_data1 = generate_data_Deep_bimodal_example(
        interval=(-3.0, 3.0), N_total=1000000
    )
elif dataset == "example1":
    train_data, valid_data = generate_data_MGAN_example_1(
        interval=(-3.0, 3.0), N_total=n_total
    )
    test_data, valid_data1 = generate_data_MGAN_example_1(
        interval=(-3.0, 3.0), N_total=1000000
    )
elif dataset == "swissroll":
    train_data, valid_data = swissroll(N_total=n_total, noise=0.1)
    test_data, valid_data1 = swissroll(N_total=1000000, noise=0.1)

mae = keras.losses.MeanAbsoluteError()

# Defining hyperparameters to investigate

n_critic = 20
gp_coef = 1e-3
act_func = "tanh"
learning_rate = 1e-4
init_idx = 0
idx = 0
savefolders_log = []
loss_type = "Adler"

# ============ Training ==================
print("\n --- Starting training \n")

train_data = tf.constant(train_data, dtype=tf.float32)


if idx >= init_idx:
    print(idx)
    G_model = gen_model(act_func=act_func)

    D_model = critic_model(act_func=act_func)

    G_optim = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.9)
    D_optim = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.9)

    n_iters = 1
    G_loss_log = []
    D_loss_log = []
    wd_loss_log = []
    optimal_wd_loss_log = []
    normalized_optimal_wd_loss_log = []
    Lipschitz_ct_log = []
    L1_loss_log = []
    kl_div_log = []
    rel_L2_err_log = []
    emd_log = []
    obtain_LipCt = False
    b = 0

    print("\n --- Creating network folder \n")
    savedir = (
        f"/home1/murgoiti/exps/Lipshitz1_paper_results_FullBatch_ntotal_{n_total}/"
        + dataset
        + "_"
        + repetition
        + "/"
    )
    savefolder = (
        savedir
        + dataset
        + "_"
        + loss_type
        + "_gpcoef_"
        + str(gp_coef)
        + "_LR_"
        + str(learning_rate)
        + "_"
        + act_func
        + "_ncritic_"
        + str(n_critic)
        + "_rep_"
        + repetition
    )  # + "_changencritic_" + str(change_n_critic)

    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    else:
        print("\n     *** Folder already exists!\n")

    while n_iters <= n_epoch * (n_total // batch_size):

        if (n_iters) % (freq_metrics) == 0:
            obtain_LipCt = True
            print("n_iters")

        true_X = train_data[b * batch_size : (b + 1) * batch_size, 0:1]
        true_Y = train_data[b * batch_size : (b + 1) * batch_size, 1:2]
        b += 1
        if b > (n_total // batch_size - 1):
            b = 0
            # np.random.shuffle(train_data.numpy())
            train_data = tf.constant(train_data, dtype=tf.float32)

        z = np.random.normal(0, 1, batch_size)

        if loss_type == "ctransform":
            D_loss, wd_loss, Lip_ct, fake = D_train_step_ctransform(
                true_X,
                true_Y,
                z,
                G_model,
                D_model,
                D_optim,
                obtain_LipCt=obtain_LipCt,
            )
        else:
            D_loss, wd_loss, gp, Lip_ct, fake = D_train_step(
                true_X,
                true_Y,
                z,
                G_model,
                D_model,
                D_optim,
                gp_coef,
                loss_type,
                obtain_LipCt=obtain_LipCt,
            )

        D_loss_log.append(D_loss.numpy())
        wd_loss_log.append(wd_loss.numpy())

        if n_iters % freq_metrics == 0:
            true = np.concatenate([true_X[:5000], true_Y[:5000]], axis=1)
            emd = calculate_EMD(true, fake.numpy()[:5000])
            emd_log.append(emd)

            z_tmp = np.random.normal(0, 1, 1000000)
            true_X_tmp = test_data[:, 0:1]
            true_Y_tmp = test_data[:, 1:2]
            fake_tmp = sample_mollified(
                G_model,
                true_X_tmp,
                z_tmp,
                n_total=1000000,
                ball_radius=0,
            )
            rel_L2_err, kl_div = calculate_errors_jointdist(
                true_X_tmp, true_Y_tmp, fake_tmp
            )

            rel_L2_err_log.append(rel_L2_err)
            kl_div_log.append(kl_div)

            optimal_wd_loss_log.append(wd_loss.numpy())
            Lipschitz_ct_log.append(Lip_ct.numpy())
            normalized_optimal_wd_loss_log.append(wd_loss.numpy() / Lip_ct.numpy())

            obtain_LipCt = False

        if (n_iters) % (n_critic) == 0:
            if loss_type == "ctransform":
                G_loss, L1_loss = G_train_step_ctransform(
                    true_X,
                    true_Y,
                    z,
                    G_model,
                    D_model,
                    G_optim,
                    mae,
                )
            else:
                G_loss, L1_loss = G_train_step(
                    true_X,
                    true_Y,
                    z,
                    G_model,
                    D_model,
                    G_optim,
                    mae,
                )

            G_loss_log.append(G_loss.numpy())
            L1_loss_log.append(L1_loss.numpy())

        if n_iters % 200000 == 0 and save_model:
            gen_model_name = savefolder + "/generator/epochs={}".format(n_iters)
            G_model.save(gen_model_name)
            critic_model_name = savefolder + "/critic/epochs={}".format(n_iters)
            D_model.save(critic_model_name)

        if n_iters % 200000 == 0 and save_histograms:

            true_X = train_data[:, 0:1]
            true_Y = train_data[:, 1:2]

            save_plot_histogram2d(fake_tmp, true_X, true_Y, savefolder, n_iters)

        if n_iters % 10000 == 0 and save_metrics:

            save_loss(G_loss_log, "g_loss", savefolder, n_epoch)
            save_loss(D_loss_log, "d_loss", savefolder, n_epoch)
            save_loss(wd_loss_log, "wd_loss", savefolder, n_epoch)
            save_loss(L1_loss_log, "L1_loss", savefolder, n_epoch)
            save_loss(
                kl_div_log,
                "KL_divergence",
                savefolder,
                n_epoch,
                plot=False,
            )
            save_loss(
                rel_L2_err_log,
                "relative_L2_error",
                savefolder,
                n_epoch,
                plot=True,
            )
            save_loss(
                optimal_wd_loss_log,
                "optimal_wd_loss",
                savefolder,
                n_epoch,
            )
            save_loss(
                normalized_optimal_wd_loss_log,
                "normalized_optimal_wd_loss",
                savefolder,
                n_epoch,
            )
            save_loss(
                Lipschitz_ct_log,
                "Lipschitz_ct",
                savefolder,
                n_epoch,
            )
            save_loss(emd_log, "emd_log", savefolder, n_epoch)

        n_iters += 1
idx += 1


print("\n ============== DONE =================\n")
