#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import pdb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0 = All, 1 = INFO, 2 = WARNING, 3 = ERROR

import datetime
import math
import pandas
import shutil
import six
import tempfile
import time
import traceback

import numpy as np
import tensorflow as tf

from itertools import cycle
from utils.audio import compute_mfcc
from utils.preprocess import preprocess
from utils.text import Alphabet


# Importer
# ========
tf.app.flags.DEFINE_string ('train_files',      '',     'comma separated list of files specifying the dataset used for training.')
tf.app.flags.DEFINE_string ('dev_files',        '',     'comma separated list of files specifying the dataset used for validation.')
tf.app.flags.DEFINE_string ('test_files',       '',     'comma separated list of files specifying the dataset used for testing.')

# Global Constants
# ================
tf.app.flags.DEFINE_boolean ('train',         True,     'whether to train the network')
tf.app.flags.DEFINE_boolean ('test',         False,     'whether to test the network')
tf.app.flags.DEFINE_integer ('epoch',           75,     'target epoch to train - if negative, the absolute number of additional epochs will be trained')

# Batch sizes
tf.app.flags.DEFINE_integer ('train_batch_size', 1,     'number of elements in a training batch')
tf.app.flags.DEFINE_integer ('dev_batch_size',   1,     'number of elements in a validation batch')
tf.app.flags.DEFINE_integer ('test_batch_size',  1,     'number of elements in a test batch')

# Decoder
tf.app.flags.DEFINE_string ('alphabet_config_path', 'utils/alphabet.txt',  'path to the configuration file specifying the phone used by the network.')

FLAGS = tf.app.flags.FLAGS


def initialize_globals():
    global session_config
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    global alphabet
    alphabet = Alphabet(os.path.abspath(FLAGS.alphabet_config_path))

    # Geometric Constants
    # ===================
    # The number of MFCC features
    global n_input
    n_input = 13 # TODO: Determine this programatically from the sample rate

    # The number of frames in the context
    global n_context
    n_context = 1 # TODO: Determine the optimal value using a validation data set

    # The dimension of LSTM cell in the encoder and the decoder
    global n_cell_dim
    n_cell_dim = 256

    # The number of characters in the target language plus twos entence start and end labels
    global n_character
    n_character = alphabet.size()

    # The number of epochs when training models
    global n_epoch
    n_epoch = FLAGS.epoch

    # Using Seq2Seq library (contrib)
    global seq2seq
    seq2seq = tf.contrib.seq2seq


# Execution
# =========
class SimpleASR:
    def __init__(self, s_len, s_indices, t_len, t_input_indices, t_output_indices, n_of_classes, t_dict):
        with tf.variable_scope('input_layer'):
            # s : source, t: target
            self._s_len = s_len
            self._s_indices = s_indices
            self._t_len = t_len
            self._t_input_indices = t_input_indices
            self._t_output_indices = t_output_indices
            self._n_class = n_of_classes
            self._t_dict = t_dict

        with tf.variable_scope('encoder'):
            enc_fw_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_cell_dim, activation = tf.nn.tanh)
            enc_bw_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_cell_dim, activation = tf.nn.tanh)
            _, enc_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_fw_cell, cell_bw = enc_bw_cell,
                                                             inputs = self._s_indices, sequence_length = self._s_len, dtype = tf.float32)

        with tf.variable_scope('pipe'):
            self.concatenated_enc_state = tf.concat(values = [enc_states[0], enc_states[1]], axis = 1)
            dec_init_state = tf.layers.dense(inputs = self.concatenated_enc_state ,
                                             units = n_cell_dim, activation=None)

            t_embeddings = tf.eye(num_rows = self._n_class)
            t_embeddings = tf.get_variable(name = 'embeddings',
                                           initializer = t_embeddings,
                                           trainable = False)
            self.t_batch = tf.nn.embedding_lookup(params = t_embeddings, ids = self._t_input_indices)

            # t_batch shape : (batch_size, step_size, n_chars)
            self.batch_size = tf.shape(self.t_batch)[0]
            self.t_max_len = tf.shape(self.t_batch)[1]
            self.tr_tokens = tf.tile(input = [self.t_max_len], multiples = [self.batch_size])

        with tf.variable_scope('decoder'):
            dec_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_cell_dim, activation = tf.nn.tanh)
            output_layer = tf.layers.Dense(units = n_of_classes,
                                           kernel_initializer= tf.contrib.layers.xavier_initializer(uniform = False))

            with tf.variable_scope('training'):
                tr_helper = seq2seq.TrainingHelper(inputs = self.t_batch,
                                                   sequence_length = self.tr_tokens)
                tr_decoder = seq2seq.BasicDecoder(cell = dec_cell, helper = tr_helper,
                                                  initial_state = dec_init_state,
                                                  output_layer = output_layer)
                self._tr_outputs, _, _ = seq2seq.dynamic_decode(decoder = tr_decoder,
                                                                impute_finished = True,
                                                                maximum_iterations = self.t_max_len)

        with tf.variable_scope('seq2seq_loss'):
            masking = tf.sequence_mask(lengths = self._t_len,
                                       maxlen = self.t_max_len, dtype = tf.float32)
            self.__seq2seq_loss = seq2seq.sequence_loss(logits = self._tr_outputs.rnn_output,
                                                        targets = self._t_output_indices,
                                                        weights = masking)

    @property
    def loss(self):
        return self.__seq2seq_loss

class AttASR:
    def __init__(self, s_len, s_indices, t_len, t_input_indices, t_output_indices, n_of_classes, t_dict):
        with tf.variable_scope('input_layer'):
            # s : source, t: target
            self._s_len = s_len
            self._s_indices = s_indices
            self._t_len = t_len
            self._t_input_indices = t_input_indices
            self._t_output_indices = t_output_indices
            self._n_class = n_of_classes
            self._t_dict = t_dict

        with tf.variable_scope('encoder'):
            enc_fw_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_cell_dim, activation = tf.nn.tanh)
            enc_bw_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_cell_dim, activation = tf.nn.tanh)
            enc_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_fw_cell, cell_bw = enc_bw_cell,
                                                             inputs = self._s_indices, sequence_length = self._s_len, dtype = tf.float32)
            self.enc_outputs = tf.concat(values = [enc_outputs[0], enc_outputs[1]], axis = 2)

        with tf.variable_scope('pipe'):
            t_embeddings = tf.eye(num_rows = self._n_class)
            t_embeddings = tf.get_variable(name = 'embeddings',
                                           initializer = t_embeddings,
                                           trainable = False)
            self.t_batch = tf.nn.embedding_lookup(params = t_embeddings, ids = self._t_input_indices)

            # t_batch shape : (batch_size, step_size, n_chars)
            self.batch_size = tf.shape(self.t_batch)[0]
            self.t_max_len = tf.shape(self.t_batch)[1]
            self.tr_tokens = tf.tile(input = [self.t_max_len], multiples = [self.batch_size])

        with tf.variable_scope('decoder'):
            dec_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_cell_dim, activation = tf.nn.tanh)

            # Applying attention-mechanism
            attn = seq2seq.LuongAttention(num_units = n_cell_dim,
                                         memory = self.enc_outputs,
                                         memory_sequence_length = self._s_len, dtype = tf.float32)
            self.attn_cell = seq2seq.AttentionWrapper(cell = dec_cell, attention_mechanism = attn)
            dec_initial_state = self.attn_cell.zero_state(batch_size = self.batch_size, dtype = tf.float32)
            output_layer = tf.layers.Dense(units = n_of_classes,
                                           kernel_initializer= tf.contrib.layers.xavier_initializer(uniform = False))

            with tf.variable_scope('training'):
                tr_helper = seq2seq.TrainingHelper(inputs = self.t_batch,
                                                   sequence_length = self.tr_tokens)
                tr_decoder = seq2seq.BasicDecoder(cell = self.attn_cell, helper = tr_helper,
                                                  initial_state = dec_initial_state,
                                                  output_layer = output_layer)
                self._tr_outputs, _, _ = seq2seq.dynamic_decode(decoder = tr_decoder,
                                                                impute_finished = True,
                                                                maximum_iterations = self.t_max_len)

        with tf.variable_scope('seq2seq_loss'):
            masking = tf.sequence_mask(lengths = self._t_len,
                                       maxlen = self.t_max_len, dtype = tf.float32)
            self.__seq2seq_loss = seq2seq.sequence_loss(logits = self._tr_outputs.rnn_output,
                                                        targets = self._t_output_indices,
                                                        weights = masking)

    @property
    def loss(self):
        return self.__seq2seq_loss


class LargeSimpleASR:
    def __init__(self, s_len, s_indices, t_len, t_input_indices, t_output_indices, n_of_classes, t_dict):
        with tf.variable_scope('input_layer'):
            # s : source, t: target
            self._s_len = s_len
            self._s_indices = s_indices
            self._t_len = t_len
            self._t_input_indices = t_input_indices
            self._t_output_indices = t_output_indices
            self._n_class = n_of_classes
            self._t_dict = t_dict
            # FIXME: need to declare the global variable of the 'n_enc_layers' and the 'n_dec_layers'.
            self.n_enc_layers = 3
            self.n_dec_layers = 1  # FIXME: do not work the stacked LSTM cells in the decoder.

        with tf.variable_scope('encoder'):
            enc_fw_cell = tf.contrib.rnn.BasicRNNCell
            enc_bw_cell = tf.contrib.rnn.BasicRNNCell
            enc_fw_cells = [enc_fw_cell(num_units = n_cell_dim, activation = tf.nn.tanh) for _ in range(self.n_enc_layers)]
            enc_bw_cells = [enc_bw_cell(num_units = n_cell_dim, activation = tf.nn.tanh) for _ in range(self.n_enc_layers)]

            _, self.enc_states_fw, self.enc_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw = enc_fw_cells, cells_bw = enc_bw_cells,
                inputs = self._s_indices, sequence_length = self._s_len, dtype = tf.float32)

        with tf.variable_scope('pipe'):
            self.concatenated_enc_state = tf.concat(values = [self.enc_states_fw[self.n_enc_layers-1], self.enc_states_bw[self.n_enc_layers-1]], axis = 1)
            dec_init_state = tf.layers.dense(inputs = self.concatenated_enc_state,
                                             units = n_cell_dim, activation = None)

            t_embeddings = tf.eye(num_rows = self._n_class)
            t_embeddings = tf.get_variable(name = 'embeddings',
                                           initializer = t_embeddings,
                                           trainable = False)
            self.t_batch = tf.nn.embedding_lookup(params = t_embeddings, ids = self._t_input_indices)

            # t_batch shape : (batch_size, step_size, n_chars)
            self.batch_size = tf.shape(self.t_batch)[0]
            self.t_max_len = tf.shape(self.t_batch)[1]
            self.tr_tokens = tf.tile(input = [self.t_max_len], multiples = [self.batch_size])

        with tf.variable_scope('decoder'):
            dec_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_cell_dim, activation = tf.nn.tanh)
            output_layer = tf.layers.Dense(units = n_of_classes,
                                           activation = tf.nn.softmax,
                                           kernel_initializer= tf.contrib.layers.xavier_initializer(uniform = False))

            with tf.variable_scope('training'):
                tr_helper = seq2seq.TrainingHelper(inputs = self.t_batch,
                                                   sequence_length = self.tr_tokens)
                tr_decoder = seq2seq.BasicDecoder(cell = dec_cell, helper = tr_helper,
                                                  initial_state = dec_init_state,
                                                  output_layer = output_layer)
                self._tr_outputs, _, _ = seq2seq.dynamic_decode(decoder = tr_decoder,
                                                                impute_finished = True,
                                                                maximum_iterations = self.t_max_len)

        with tf.variable_scope('seq2seq_loss'):
            masking = tf.sequence_mask(lengths = self._t_len,
                                       maxlen = self.t_max_len, dtype = tf.float32)
            self.__seq2seq_loss = seq2seq.sequence_loss(logits = self._tr_outputs.rnn_output,
                                                        targets = self._t_output_indices,
                                                        weights = masking)

    @property
    def loss(self):
        return self.__seq2seq_loss


class LargeAttASR:
    def __init__(self, s_len, s_indices, t_len, t_input_indices, t_output_indices, n_of_classes, t_dict):
        with tf.variable_scope('input_layer'):
            # s : source, t: target
            self._s_len = s_len
            self._s_indices = s_indices
            self._t_len = t_len
            self._t_input_indices = t_input_indices
            self._t_output_indices = t_output_indices
            self._n_class = n_of_classes
            self._t_dict = t_dict
            # FIXME: need to declare the global variable of the 'n_enc_layers' and the 'n_dec_layers'.
            self.n_enc_layers = 2
            self.n_dec_layers = 1  # FIXME: do not work the stacked LSTM cells in the decoder.

        with tf.variable_scope('encoder'):
            enc_fw_cell = tf.contrib.rnn.BasicRNNCell
            enc_bw_cell = tf.contrib.rnn.BasicRNNCell
            enc_fw_cells = [enc_fw_cell(num_units = n_cell_dim, activation = tf.nn.tanh) for _ in range(self.n_enc_layers)]
            enc_bw_cells = [enc_bw_cell(num_units = n_cell_dim, activation = tf.nn.tanh) for _ in range(self.n_enc_layers)]

            self.enc_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw = enc_fw_cells, cells_bw = enc_bw_cells,
                inputs = self._s_indices, sequence_length = self._s_len, dtype = tf.float32)

        with tf.variable_scope('pipe'):
            t_embeddings = tf.eye(num_rows = self._n_class)
            t_embeddings = tf.get_variable(name = 'embeddings',
                                           initializer = t_embeddings,
                                           trainable = False)
            self.t_batch = tf.nn.embedding_lookup(params = t_embeddings, ids = self._t_input_indices)

            # t_batch shape : (batch_size, step_size, n_chars)
            self.batch_size = tf.shape(self.t_batch)[0]
            self.t_max_len = tf.shape(self.t_batch)[1]
            self.tr_tokens = tf.tile(input = [self.t_max_len], multiples = [self.batch_size])

        with tf.variable_scope('decoder'):
            dec_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_cell_dim, activation = tf.nn.tanh)

            # Applying attention-mechanism
            attn = seq2seq.LuongAttention(num_units = n_cell_dim,
                                          memory = self.enc_outputs,
                                          memory_sequence_length = self._s_len, dtype = tf.float32)
            self.attn_cell = seq2seq.AttentionWrapper(cell = dec_cell, attention_mechanism = attn)
            dec_initial_state = self.attn_cell.zero_state(batch_size = self.batch_size, dtype = tf.float32)
            output_layer = tf.layers.Dense(units = n_of_classes,
                                           activation = tf.nn.softmax,
                                           kernel_initializer= tf.contrib.layers.xavier_initializer(uniform = False))

            with tf.variable_scope('training'):
                tr_helper = seq2seq.TrainingHelper(inputs = self.t_batch,
                                                   sequence_length = self.tr_tokens)
                tr_decoder = seq2seq.BasicDecoder(cell = self.attn_cell, helper = tr_helper,
                                                  initial_state = dec_initial_state,
                                                  output_layer = output_layer)
                self._tr_outputs, _, _ = seq2seq.dynamic_decode(decoder = tr_decoder,
                                                                impute_finished = True,
                                                                maximum_iterations = self.t_max_len)

        with tf.variable_scope('seq2seq_loss'):
            masking = tf.sequence_mask(lengths = self._t_len,
                                       maxlen = self.t_max_len, dtype = tf.float32)
            self.__seq2seq_loss = seq2seq.sequence_loss(logits = self._tr_outputs.rnn_output,
                                                        targets = self._t_output_indices,
                                                        weights = masking)

    @property
    def loss(self):
        return self.__seq2seq_loss


def model_feeder(features, batch_size=1, feeder_name=''):
    # Get samples for model feeding
    data_zip = cycle(zip(
        features['X_length'], features['X_indices'],
        features['Y_length'], features['Y_input_indices'], features['Y_target_indices']))

    # Types and shapes of sample 
    data_types = (tf.int32, tf.float32, tf.int32, tf.int32, tf.int32)
    data_shapes = ([], [None, n_input + (2 * n_input * n_context)], [], [None], [None])

    # Create data pipline with tf.data
    dataset = tf.data.Dataset.from_generator(lambda:data_zip, output_types=data_types, output_shapes=data_shapes)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([],[None, n_input + (2 * n_input * n_context)],[],[None],[None])).repeat()
    iters = dataset.make_initializable_iterator()

    return iters


def train(server=None):
    # Reading training set
    train_data = preprocess(FLAGS.train_files.split(','),
                            n_input,
                            n_context,
                            alphabet)

    # Reading validation set
    #dev_data = preprocess(FLAGS.dev_files.split(','),
    #                      n_input,
    #                      n_context,
    #                      alphabet)

    # Generate feeding data in the training set
    train_iters = model_feeder(train_data,
                               batch_size=FLAGS.train_batch_size,
                               feeder_name='train_batch')

    # Generate feeding data in the validation set
    # dev_iters = model_feeder(dev_data, \
    #                         batch_size=FLAGS.dev_batch_size, \
    #                         feeder_name='dev_batch')

    X_length_mb, X_indices_mb, Y_length_mb, Y_input_indices_mb, Y_target_indices_mb = train_iters.get_next()
    sim_asr = SimpleASR(s_len = X_length_mb, s_indices = X_indices_mb,
                             t_len = Y_length_mb, t_input_indices = Y_input_indices_mb, t_output_indices = Y_target_indices_mb,
                             n_of_classes = n_character, t_dict = alphabet)

    opt = tf.train.AdamOptimizer(learning_rate = 0.001)
    training_op = opt.minimize(loss = sim_asr.loss)

    sess = tf.Session(config = session_config)
    sess.run(tf.global_variables_initializer())

    print('Start training')
    print('==> total epochs : {}'.format(n_epoch))
    for epoch in range(n_epoch):
        avg_tr_loss = 0
        tr_step = 0

        sess.run(train_iters.initializer)
        #FIXME: need to automatically select the batch size from the #epoch and #data.
        #for step in range(n_batch):
        for step in range(164):
            # for debugging the AttASR or the LargeAttASR class
            #_, tr_loss, dec_out, enc_out, dec_in, dec_toks, dec_target_idx = sess.run(fetches = [training_op, sim_asr.loss, sim_asr._tr_outputs, sim_asr.enc_outputs, sim_asr.t_batch, sim_asr.tr_tokens, sim_asr._t_output_indices])

            # for debugging the SimpleASR or the LargeSimpleASR class
            _, tr_loss, dec_out, enc_out, dec_in, dec_toks, dec_target_idx = sess.run(fetches=[training_op, sim_asr.loss, sim_asr._tr_outputs, sim_asr.concatenated_enc_state, sim_asr.t_batch, sim_asr.tr_tokens, sim_asr._t_output_indices])

            #_, tr_loss = sess.run(fetches=[training_op, sim_asr.loss])
            avg_tr_loss += tr_loss
            tr_step += 1
            print('==> epoch : {}, step : {}, tr_loss : {:.3f}'.format(epoch+1, tr_step, tr_loss))

        #pdb.set_trace()
        avg_tr_loss /= tr_step
        print('epoch : {}, avg_tr_loss : {:.3f}'.format(epoch+1, avg_tr_loss))

    pdb.set_trace()
    print('Finished')


def main(_):

    initialize_globals()

    if FLAGS.train:
        # Only one local task
        train()
        print('Session closed.')


if __name__ == "__main__":
    tf.app.run()













