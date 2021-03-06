from __future__ import division, print_function, absolute_import

from network import *
import numpy as np
import tensorflow as tf2
import scipy.stats
import scipy.io as sio
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Training Parameters
epochs = 20000
nepoch=10
batch_size = 10
learning_rate = 1e-5
drop_out=0.5
reg_param=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] #gamma, inter-eFC correlation
reg_constant=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1] #lambda

# Number of connections at input and output
conn_dim = 3655 #(upper-triangle of Connectiivty matrix), 2278 originally
data_path='SCsdstream_FChpf_train340.mat'

for k in range(len(reg_param)):
    for j in range(len(reg_constant)):
        
        #path to save model and training results
        save_dir='directory/model_gam%d'%k + '_lam%d'%j + '.ckpt'
        save_files='directory/training_output.mat'

        #Xavier initializer
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf2.initializers.GlorotUniform()

        with tf.device('//device:GPU:0'):
            ################ Build Network############################
            # Network Inputs
            sc_input = tf.placeholder(tf.float32, shape=[None, conn_dim], name='SC')
            fc_output = tf.placeholder(tf.float32, shape=[None, conn_dim], name='FC')
            keep_prob = tf.placeholder(tf.float32, name="dropout")


            fc_generated = predictor(sc_input,keep_prob)
            fc_gen = predictor(sc_input,1) #for computing correlation without dropout
            reg = compute_corr_loss(fc_gen,batch_size) #regularization parameter, phi, inter-pFC correlation
            loss =  tf.losses.mean_squared_error(fc_output,fc_generated) + reg_constant[j] * tf.abs(reg - reg_param[k])


            # Build Optimizers
            optimizer_gen = tf.train.AdamOptimizer(learning_rate)


            # Training Variables for each optimizer
            train_gen = optimizer_gen.minimize(loss)

            # Initialize the variables (i.e. assign their default value)
            init = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            saver = tf.train.Saver()

        # Create session for training
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

            # Run the initializer
            sess.run(init)
            sess.run(init_l)

            #initialze varaibles for recording correlations and loss
            corr = np.zeros((epochs))
            nnloss = np.zeros((epochs))

            for i in range(epochs):

                #for averaging the results over  entire dataset for each epoch
                loss_temp = 0
                ci = 0

                #load data
                input_data = sio.loadmat(data_path)['sc']
                output_data = sio.loadmat(data_path)['fc']

                # Train
                for iters in range(0, np.shape(input_data)[0], batch_size):
                    batch_out = output_data[iters:iters + batch_size, :]
                    batch_in = input_data[iters:iters + batch_size, :]

                    #for training (dropout included)
                    feed_dict = {fc_output: batch_out, sc_input: batch_in, keep_prob: drop_out}
                    _, lt= sess.run([ train_gen,  loss], feed_dict=feed_dict)

                    #calculate the loss without dropout
                    feed_dict = {fc_output: batch_out, sc_input: batch_in, keep_prob: 1}
                    generated = sess.run(fc_generated, feed_dict=feed_dict)
                    corr_intra=compute_corr(generated)#compute intra-pFC

                    loss_temp += lt
                    ci += corr_intra


                #Averaging resuults for the all the iterations
                nnloss[i] = loss_temp / (np.shape(input_data)[0] / batch_size)
                corr[i] = ci / (np.shape(input_data)[0] / batch_size)
                print('Epoch %d: G loss: %f Corr: %f' % (i, nnloss[i], corr[i]))

            save_path = saver.save(sess, save_dir,global_step=i+1)
            print("Model saved in file: %s" % save_path)
            sio.savemat(save_files, {'loss': nnloss, 'corr':corr})

