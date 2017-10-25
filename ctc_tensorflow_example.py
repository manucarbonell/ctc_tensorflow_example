#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
import cv2
import os

from six.moves import xrange as range
from IAM_input import IAM_input
try:
    from python_speech_features import mfcc
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError

from utils import maybe_download as maybe_download
from utils import sparse_tuple_from as sparse_tuple_from

iam_train=IAM_input()


# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
################################################### CHANGE NUMBER OF FEATURES TO IMAGE HEIGHT
num_features=int(iam_train.im_height)
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 200
num_hidden = 120
num_layers = 1
batch_size = iam_train.batch_size
n_channels = 1
initial_learning_rate = 1e-3
momentum = 0.9

num_examples = iam_train.total_examples
num_batches_per_epoch = int(num_examples/batch_size)

checkpoint_path="./checkpoints"

# THE MAIN CODE!

graph = tf.Graph()

with graph.as_default():
    # Input tensor has size [batch_size,num_features,max_stepsize, n_channels], but the
    # batch_size and max_stepsize can vary along each step

    inputs = tf.placeholder(tf.float32, [batch_size,num_features, None, n_channels])
    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)
    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    ############ CONVOLUTION
    w_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
    b_conv1 = tf.Variable(tf.constant(0., shape=[32]))

    conv1 = tf.nn.conv2d(inputs, w_conv1, strides=[1, 1, 1, 1], padding='SAME')

    conv1 = tf.nn.bias_add(conv1, b_conv1)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.reshape(conv1,[batch_size,-1,num_features*32])
    ############



    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, conv1, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    global_step = tf.Variable(0, trainable=False)



    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               100, 0.96, staircase=True)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,0.9).minimize(cost,global_step=global_step)

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run(  )



    saver = tf.train.Saver(tf.global_variables())

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        print("Model restored.")

    else:
        print("No checkpoint found, start training from beginning.")



    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        for batch in range(num_batches_per_epoch):
        #for batch in range(10):
            X, Y = iam_train.get_batch()

            train_seq_len = [x.shape[1] for x in X]
            print("EPOCH",curr_epoch,"STEP",batch)

            train_targets = sparse_tuple_from(Y)
            print ("TARGETS",train_targets)
            print("inputs",X.shape)
            feed = {inputs: X,
                    targets: train_targets,
                    seq_len: train_seq_len}

            batch_cost, _ = session.run([cost, optimizer], feed)
            #if batch % 10 == 0:
                #decod = session.run(decoded,feed)

                #for j in range(batch_size):
                    #print(decod,(decod),len(decod[0]))
                    #print("DECODED:", iam_train.id_to_char(decod[j][1]))
                    #print("Y:", iam_train.id_to_char(Y[j]))
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples

        print("Saving model...")
        saver.save(session,os.path.join(checkpoint_path,'model.ckpt'))
        print ("Finished.")

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f} time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         time.time() - start))


    # Decoding
    d = session.run(decoded[0], feed_dict=feed)
    str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

    print('Decoded:\n%s' % str_decoded)
