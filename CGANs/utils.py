import os
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import ot

np.random.seed(1008)
#============== Parameters ======================

def swissroll(N_total, noise):

	t = 3*np.pi/2 * (1 + 2*np.random.rand(1,N_total))
	data = np.concatenate((0.1*t*np.cos(t),0.1*t*np.sin(t))) + noise*np.random.randn(2,N_total)	
 
	return np.transpose(data)[:N_total,:], np.transpose(data)[N_total:,:]

def generate_data_sin(interval_end=2*np.pi, z_scale=0.1, N_total = 100):

    dataset = np.ones((N_total,2))
    dataset[:,0] = np.random.uniform(0,interval_end, N_total)
    dataset[:,1] = np.sin(dataset[:,0]) + np.random.normal(0.0,z_scale,size=N_total)
    
    return dataset[:N_total,:], dataset[N_total:,:]

def generate_data_MGAN_example_1(interval=(-3.,3.), N_total = 100):

    dataset = np.ones((int(N_total*1.2),2))
    dataset[:,0] = np.random.uniform(interval[0], interval[1], int(N_total*1.2))
    dataset[:,1] = np.tanh(dataset[:,0]) + np.random.gamma(shape=1.,scale=.3,size=int(N_total*1.2))
    
    return dataset[0:N_total,:], dataset[N_total:,:]

def generate_data_MGAN_example_2(interval=(-3.,3.), N_total = 100):

    dataset = np.ones((int(N_total*1.2),2))
    dataset[:,0] = np.random.uniform(interval[0], interval[1], int(N_total*1.2))
    dataset[:,1] = np.tanh(dataset[:,0] + np.random.normal(0,.05,size=int(N_total*1.2)))
    
    return dataset[0:N_total,:], dataset[N_total:,:]

def generate_data_MGAN_example_3(interval=(-3.,3.), N_total = 100):

    dataset = np.ones((int(N_total*1.2),2))
    dataset[:,0] = np.random.uniform(interval[0], interval[1], int(N_total*1.2))
    dataset[:,1] = np.tanh(dataset[:,0]) * np.random.gamma(shape=1.,scale=.3,size=int(N_total*1.2))
    
    return dataset[0:N_total,:], dataset[N_total:,:]

def generate_data_Deep_bimodal_example(interval=(-3.,3.), N_total = 100):

    dataset = np.ones((int(N_total*1.2),2))
    dataset[:,0] = np.random.normal(0, 1.0, int(N_total*1.2))
    dataset[:,1] = np.cbrt(dataset[:,0] + np.random.normal(0,1,size=int(N_total*1.2)))
    
    return dataset[0:N_total,:], dataset[N_total:,:]
       

def get_lat_var(batch_size,g_type,x_W,x_H,z_dim):
    if g_type == 'type1':
        z = tf.random.normal((batch_size,1,1,z_dim))
    elif g_type == 'type2':
        z = tf.random.normal((batch_size,x_W,x_H,1))    

    return z    


def save_loss(loss,loss_name,savedir,n_epoch, plot=True):
    

    np.savetxt(f"{savedir}/{loss_name}.txt",loss)
    fig, ax1 = plt.subplots()
    ax1.plot(loss)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel(loss_name)
    ax1.set_xlim([1,len(loss)])


    ax2 = ax1.twiny()
    ax2.set_xlim([0,n_epoch])
    ax2.set_xlabel('Epochs')

    if plot:
        plt.tight_layout()
        plt.savefig(f"{savedir}/{loss_name}.png",dpi=200)    
        plt.close()  

def get_main_dir(dir_path):

    n = len(dir_path)-1
    found = False
    while n > - 1 and not found:
        if dir_path[n] != '/':
            n-=1
        else:
            found = True
            n+=1

    return dir_path[n::]    

def calculate_KL_divergence(hist1, hist2):
    kl_div = 0
    for i in range(50):
        for j in range(50):
            if hist2[i,j] > 0 and hist1[i,j] > 0:
                kl_div += hist1[i,j]*np.log(hist1[i,j]/hist2[i,j])
    return kl_div

def calculate_errors_jointdist(true_X, true_Y, fake):

    hist2d_fake, e1, e2 = np.histogram2d(fake[:,0], fake[:,1], bins = [75,50], range=np.array([[-3,3],[-2,2]]), density=True)
    hist2d_fake = hist2d_fake/np.sum(hist2d_fake)
    hist2d_true, e1, e2 = np.histogram2d(true_X[:,0], true_Y[:,0], bins = [75,50], range=np.array([[-3,3],[-2,2]]), density=True)
    hist2d_true = hist2d_true/np.sum(hist2d_true)
    relative_L2_error = tf.norm(hist2d_fake-hist2d_true, ord='euclidean')/tf.norm(hist2d_fake, ord='euclidean')

    #kl_divergence = calculate_KL_divergence(hist2d_fake, hist2d_true)

    return relative_L2_error , 0#kl_divergence

def calculate_EMD(train_data, generated_data):
  
  M = ot.dist(train_data, generated_data, 'euclidean')
  ab = np.ones(train_data.shape[0]) / train_data.shape[0]
  loss = ot.emd2(ab, ab, M)

  return loss

def sample_mollified(model, true_X, z, n_total, ball_radius = 0.01):

    epsilon = np.reshape(np.random.normal(0,ball_radius, n_total), newshape=true_X.shape)
    fake_Y = model([true_X + epsilon, z]).numpy()
    fake = np.concatenate([true_X, fake_Y], axis=1)
    np.random.shuffle(fake)
    return fake
