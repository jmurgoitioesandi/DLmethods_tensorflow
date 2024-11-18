import os
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import h5py
from config import cla
from models import *
PARAMS = cla()
np.random.seed(PARAMS.seed_no)
#============== Parameters ======================

slide_num = PARAMS.slide_num


def read_train_images_numpy(img_dir,N_train,batch_size):


    train_images = np.load(img_dir)
    
    total_train  = len(train_images)
    
    # We are assuming that the data is saved with chanel >= 2 (1 chanel at least each for X and Y)
    width,height,chanels = train_images[0].shape

    print(f'     *** Datasets:')
    print(f'         ... training samples   = {N_train} of {total_train}')
    print(f'         ... sample dimension   = {width}X{height}X{chanels}')
    train_images = tf.constant(train_images,dtype='float32')

    train_images = tf.data.Dataset.from_tensor_slices(train_images).shuffle(total_train).batch(batch_size, drop_remainder=True)

    return train_images


def get_lat_var(batch_size,z_dim):

    z = tf.random.normal((batch_size,z_dim))    

    return z    

def save_loss(loss,loss_name,savedir,n_epoch):
    

    np.savetxt(f"{savedir}/{loss_name}.txt",loss)
    fig, ax1 = plt.subplots()
    ax1.plot(loss)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel(loss_name)
    ax1.set_xlim([1,len(loss)])


    ax2 = ax1.twiny()
    ax2.set_xlim([0,n_epoch])
    ax2.set_xlabel('Epochs')


    plt.tight_layout()
    plt.savefig(f"{savedir}/{loss_name}.png",dpi=200)    
    plt.close()  

##PLOTTING OOD DETECTION


def make_plots_GAN_training(synthetic_images, n_rows, savedir, i, type_im="synth"):

    fig1,axs1 = plt.subplots(5, 5, dpi=200, figsize=(20,22))
    max_value = np.amax(synthetic_images)
    for idx in range(n_rows):
        combined_channels = [synthetic_images[idx,:,:,1:2] + synthetic_images[idx,:,:,3:4],
                             synthetic_images[idx,:,:,2:3] + synthetic_images[idx,:,:,3:4],
                             synthetic_images[idx,:,:,0:1] + synthetic_images[idx,:,:,3:4]]
        synthetic_RGB_image = np.concatenate(combined_channels, axis=2)
        axs1[idx,0].imshow(synthetic_RGB_image/2, vmin=0, vmax=1)
        axs1[idx,1].imshow(synthetic_images[idx,:,:,0], cmap='gray', vmin=0, vmax=1)
        axs1[idx,2].imshow(synthetic_images[idx,:,:,1], cmap='gray', vmin=0, vmax=1)
        axs1[idx,3].imshow(synthetic_images[idx,:,:,2], cmap='gray', vmin=0, vmax=1)
        axs1[idx,4].imshow(synthetic_images[idx,:,:,3], cmap='gray', vmin=0, vmax=1)
        axs1[idx,0].set_xticks([])
        axs1[idx,0].set_yticks([])
        axs1[idx,1].set_xticks([])
        axs1[idx,1].set_yticks([])
        axs1[idx,2].set_xticks([])
        axs1[idx,2].set_yticks([])
        axs1[idx,3].set_xticks([])
        axs1[idx,3].set_yticks([])
        axs1[idx,4].set_xticks([])
        axs1[idx,4].set_yticks([])
        if idx == 0:
            axs1[idx,1].set_title('DAPI', fontsize=24)
            axs1[idx,2].set_title('TRITC', fontsize=24)
            axs1[idx,3].set_title('CY5', fontsize=24)
            axs1[idx,4].set_title('FITC', fontsize=24)

    fig1.tight_layout()
    fig1.subplots_adjust(wspace=0.0)
    fig1.savefig(f"{savedir}/{type_im}_samples_{i+1}.png")
    plt.close('all')
    
