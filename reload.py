from __future__ import division, print_function, absolute_import

from network import *
import numpy as np
import tensorflow as tf2
import scipy.stats
import scipy.io as sio
# Number of connections at input and output
conn_dim = 3655 #(upper-triangle of Connectiivty matrix)
data_path='MSconnect_SC.mat' #path to your test data
model_path='directory/model_gam1_lam4.ckpt-20000' #path to your saved network"
meta_file=model_path + '.meta'
save_path='output_ckpt-20000_fs86_MSconnect_gam02_lam001.mat' #path for saving results
batch_size = 1

#Xavier initializer
#initializer = tf.contrib.layers.xavier_initializer()
initializer = tf2.initializers.GlorotUniform()


with tf.device('//device:GPU:6'):
    ################ Build Network############################
    # Network Inputs
    sc_input = tf.placeholder(tf.float32, shape=[None, conn_dim], name='SC')
    fc_output = tf.placeholder(tf.float32, shape=[None, conn_dim], name='FC')
    keep_prob = tf.placeholder(tf.float32, name="dropout")
    fc_generated = predictor(sc_input,keep_prob)


    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    saver = tf.train.Saver()


# Create sesion
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

    #load trained model/network
    new_saver = tf.train.import_meta_graph(meta_file)
    new_saver.restore(sess,model_path)

    #load test data
    input_data = sio.loadmat(data_path)['sc']
    output_data = sio.loadmat(data_path)['fc']

    # initialize outputs to zero
    output = np.zeros(np.shape(output_data))
    input = np.zeros(np.shape(input_data))
    estimated = np.zeros(np.shape(output_data))
    total = -1

    for iters in range(0, np.shape(input_data)[0], batch_size):
        batch_in = input_data[iters:iters + batch_size, :]
        actual_output = output_data[iters:iters + batch_size, :]

        pred = sess.run([fc_generated], feed_dict={sc_input: batch_in, keep_prob: 1})

        total=total+1
        input[total,:] = batch_in
        output[total,:] =actual_output
        estimated[total,:]=np.squeeze(pred, axis=0)

    sio.savemat(save_path, {'in': input, 'out': output, 'predicted': estimated})