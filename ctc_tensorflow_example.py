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
from utils import sparse_tuple_from as sparse_tuple_from

class HTR:
    def __init__(self):
        self.dataset=IAM_input()
        self.num_features = int(self.dataset.im_height)
        self.batch_size = self.dataset.batch_size
        self.n_channels = self.dataset.im_depth
        self.num_classes = 28
        self.initial_learning_rate=1e-3
        self.num_epochs=200
        self.num_examples=self.dataset.total_examples
        self.num_batches_per_epoch = int(self.num_examples / self.batch_size)
        self.checkpoint_path='./checkpoints'
        self.hidden_neurons=254

    def model(self,inputs,seq_len):
        w_conv1 = tf.Variable(tf.random_normal([3, 3, 1, 28]))
        b_conv1 = tf.Variable(tf.constant(0., shape=[28]))

        conv1 = tf.nn.conv2d(inputs, w_conv1, strides=[1, 1, 1, 1], padding='SAME')

        conv1 = tf.nn.bias_add(conv1, b_conv1)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.reshape(conv1, [self.batch_size, -1, self.num_features * 28])

        cell = tf.contrib.rnn.LSTMCell(self.hidden_neurons, state_is_tuple=True)

        # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([cell] * 1,
                                            state_is_tuple=True)

        # The second output is the last state and we will no use that
        outputs, _ = tf.nn.dynamic_rnn(stack, conv1, seq_len, dtype=tf.float32)

        shape = tf.shape(inputs)
        batch_s, max_timesteps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, self.hidden_neurons])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        W = tf.Variable(tf.truncated_normal([self.hidden_neurons,
                                             self.num_classes],
                                            stddev=0.1))
        # Zero initialization
        # Tip: Is tf.zeros_initializer the same?
        b = tf.Variable(tf.constant(0., shape=[self.num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, self.num_classes])

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))

        return logits

    def train(self,session):

        inputs = tf.placeholder(tf.float32, [self.batch_size, self.num_features, None, self.dataset.im_depth])

        targets = tf.sparse_placeholder(tf.int32)

        seq_len = tf.placeholder(tf.int32, [None])

        logits=self.model(inputs,seq_len)

        loss = tf.nn.ctc_loss(targets, logits, seq_len)

        cost = tf.reduce_mean(loss)

        global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step,
                                                   8000, 0.98, staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost, global_step=global_step)

        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))

        tf.global_variables_initializer().run(session=sess)

        saver = tf.train.Saver(tf.global_variables())

        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            print("Model restored.")

        else:
            print("No checkpoint found, start training from beginning.")

        for curr_epoch in range(self.num_epochs):
            train_cost = train_ler = 0
            start = time.time()

            X, Y = self.dataset.get_batch()

            for batch in range(self.num_batches_per_epoch):


                train_seq_len = [x.shape[1] for x in X]

                print("EPOCH", curr_epoch, "PROGRESS", self.dataset.index_in_epoch, self.dataset.total_examples)

                train_targets = sparse_tuple_from(Y)

                feed = {inputs: X,
                        targets: train_targets,
                        seq_len: train_seq_len}

                batch_cost, _ = session.run([cost, optimizer], feed)

                train_cost += batch_cost * self.batch_size
                train_ler += session.run(ler, feed_dict=feed) * self.batch_size

                #VERBOSE
                if batch % 2 == 0:
                    decod = session.run(decoded, feed)

                    for j in range(self.batch_size):
                        # print("Y:", j, iam_train.id_to_char(Y[j]))
                        print("DECODED BATCH OUTPUT:", self.dataset.id_to_char(decod[0][1]))



transcriptor=HTR()



with tf.Session() as sess:

    transcriptor.train(sess)

