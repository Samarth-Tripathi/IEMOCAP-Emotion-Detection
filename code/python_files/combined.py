import numpy as np
import os
import sys

import wave
import copy
import math
from aup import BasicConfig, print_result

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Input, Flatten, Merge, Embedding, Convolution1D,Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import label_binarize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import argparse

from features import *
from helper import *

import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent, _time_distributed_dense
from keras.engine import InputSpec

tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

class AttentionDecoder(Recurrent):

    def __init__(self, units, output_dim,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space

        references:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences

    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s

        """
            Matrices for creating the context vector
        """

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_r = self.add_weight(shape=(self.units, ),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_z = self.add_weight(shape=(self.units, ),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_p = self.add_weight(shape=(self.units, ),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.output_dim),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   name='W_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = x

        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)

        return super(AttentionDecoder, self).call(x)

    def get_initial_state(self, inputs):
        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def step(self, x, states):

        ytm, stm = states

        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(stm, self.timesteps)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),
                   K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state
        # first calculate the "r" gate:

        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1-zt)*stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(stm, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def train():
    code_path = os.path.dirname(os.path.realpath(os.getcwd()))
    emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])
    data_path = code_path + "/../data/"
    #sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    framerate = 16000

    import pickle
    with open(data_path + '/'+'data_collected.pickle', 'rb') as handle:
        data2 = pickle.load(handle)

    text = []

    for ses_mod in data2:
        text.append(ses_mod['transcription'])

    MAX_SEQUENCE_LENGTH = 50

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    token_tr_X = tokenizer.texts_to_sequences(text)
    x_train_text = []

    x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

    import codecs
    EMBEDDING_DIM = 300

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    file_loc = data_path + '/glove.42B.300d.txt'

    print (file_loc)

    gembeddings_index = {}
    with codecs.open(file_loc, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            gembedding = np.asarray(values[1:], dtype='float32')
            gembeddings_index[word] = gembedding
    #
    f.close()
    print('G Word embeddings:', len(gembeddings_index))

    nb_words = len(word_index) +1
    g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        gembedding_vector = gembeddings_index.get(word)
        if gembedding_vector is not None:
            g_word_embedding_matrix[i] = gembedding_vector

    print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))

    def calculate_features(frames, freq, options):
        window_sec = 0.2
        window_n = int(freq * window_sec)

        st_f = stFeatureExtraction(frames, freq, window_n, window_n / 2)

        if st_f.shape[1] > 2:
            i0 = 1
            i1 = st_f.shape[1] - 1
            if i1 - i0 < 1:
                i1 = i0 + 1

            deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
            for i in range(i0, i1):
                i_left = i - 1
                i_right = i + 1
                deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
            return deriv_st_f
        elif st_f.shape[1] == 2:
            deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
            deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
            return deriv_st_f
        else:
            deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
            deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
            return deriv_st_f

    x_train_speech = []

    counter = 0
    for ses_mod in data2:
        x_head = ses_mod['signal']
        st_features = calculate_features(x_head, framerate, None)
        st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
        x_train_speech.append( st_features.T )
        counter+=1
        if(counter%1000==0):
            print(counter)

    x_train_speech = np.array(x_train_speech)
    x_train_speech.shape

    x_train_mocap = []
    counter = 0
    for ses_mod in data2:
        x_head = ses_mod['mocap_head']
        if(x_head.shape != (200,18)):
            x_head = np.zeros((200,18))   
        x_head[np.isnan(x_head)]=0
        x_hand = ses_mod['mocap_hand']
        if(x_hand.shape != (200,6)):
            x_hand = np.zeros((200,6))   
        x_hand[np.isnan(x_hand)]=0
        x_rot = ses_mod['mocap_rot']
        if(x_rot.shape != (200,165)):
            x_rot = np.zeros((200,165))   
        x_rot[np.isnan(x_rot)]=0
        x_mocap = np.concatenate((x_head, x_hand), axis=1)
        x_mocap = np.concatenate((x_mocap, x_rot), axis=1)
        x_train_mocap.append( x_mocap )

    x_train_mocap = np.array(x_train_mocap)
    x_train_mocap = x_train_mocap.reshape(-1,200,189,1)
    #x_train_mocap.shape

    Y=[]
    for ses_mod in data2:
        Y.append(ses_mod['emotion'])

    Y = label_binarize(Y,emotions_used)

    #Y.shape

    counter = 0
    for ses_mod in data2:
        if (ses_mod['id'][:5]=="Ses05"):
            break
        counter+=1
    #counter

    xtrain_sp = x_train_speech[:3838]
    xtest_sp = x_train_speech[3838:]
    xtrain_tx = x_train_text[:3838]
    xtest_tx = x_train_text[3838:]
    ytrain_sp = Y[:3838]
    ytest_sp = Y[3838:]

    x_train_mocap2 = x_train_mocap.reshape(-1,200,189)
    xtrain_mo = x_train_mocap2[:3838]
    xtest_mo = x_train_mocap2[3838:]
    xtrain_mo = x_train_mocap[:3838]
    xtest_mo = x_train_mocap[3838:]

    

    model_text = Sequential()
    #model.add(Embedding(2737, 128, input_length=MAX_SEQUENCE_LENGTH))
    model_text.add(Embedding(nb_words,
                        EMBEDDING_DIM,
                        weights = [g_word_embedding_matrix],
                        input_length = MAX_SEQUENCE_LENGTH,
                        trainable = True))

    model_text.add(LSTM(FLAGS.textlstm, return_sequences=True, recurrent_dropout = 0.2))
    model_text.add(Dropout(FLAGS.dropout))
    model_text.add(LSTM(FLAGS.textlstm, return_sequences=False, recurrent_dropout = 0.2))
    model_text.add(Dropout(FLAGS.dropout))
    model_text.add(Dense(256))


    model_speech = Sequential()
    model_speech.add(LSTM(FLAGS.speechlstm, return_sequences=True, input_shape=(100, 34), recurrent_dropout = 0.2))
    model_speech.add(Dropout(FLAGS.dropout))
    model_speech.add(AttentionDecoder(FLAGS.speechlstm,FLAGS.speechlstm))
    model_speech.add(Dropout(FLAGS.dropout))
    model_speech.add(Flatten())
    model_speech.add(Dense(256))

    model_mocap = Sequential()
    model_mocap.add(Conv2D(32, 3, strides=(2, 2), border_mode='same', input_shape=(200, 189, 1)))
    model_mocap.add(Dropout(FLAGS.dropout))
    model_mocap.add(Activation('relu'))
    model_mocap.add(Conv2D(64, 3, strides=(2, 2), border_mode='same'))
    model_mocap.add(Dropout(FLAGS.dropout))
    model_mocap.add(Activation('relu'))
    model_mocap.add(Conv2D(64, 3, strides=(2, 2), border_mode='same'))
    model_mocap.add(Dropout(FLAGS.dropout))
    model_mocap.add(Activation('relu'))
    model_mocap.add(Conv2D(128, 3, strides=(2, 2), border_mode='same'))
    model_mocap.add(Dropout(FLAGS.dropout))
    model_mocap.add(Activation('relu'))
    model_mocap.add(Conv2D(128, 3, strides=(2, 2), border_mode='same'))
    model_mocap.add(Dropout(FLAGS.dropout))
    model_mocap.add(Activation('relu'))
    model_mocap.add(Flatten())
    model_mocap.add(Dense(256))

    model_combined = Sequential()
    model_combined.add(Merge([model_text, model_speech, model_mocap], mode='concat'))

    model_combined.add(Activation('relu'))

    model_combined.add(Dense(FLAGS.finalfc))
    model_combined.add(Activation('relu'))

    model_combined.add(Dense(4))
    model_combined.add(Activation('softmax'))

    #sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    cadam = Adam(lr=FLAGS.adamlr)
    model_combined.compile(loss='categorical_crossentropy',optimizer=cadam ,metrics=['acc'])

    ## compille it here according to instructions

    #model.compile()
    model_speech.summary()
    model_text.summary()
    model_mocap.summary()
    model_combined.summary()

    print("Model1 Built")

    hist = model_combined.fit([xtrain_tx,xtrain_sp,xtrain_mo], ytrain_sp, 
                     batch_size=64, nb_epoch=30, verbose=1, 
                     validation_data=([xtest_tx,xtest_sp,xtest_mo], ytest_sp))
    
    return max(hist.history['val_acc'])

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("config file required")
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='drop probability for training dropout.')
    parser.add_argument('--textlstm', type=int, default=256,
                        help='text LSTM layer size.')
    parser.add_argument('--speechlstm', type=int, default=128,
                        help='speech layer size.')
    parser.add_argument('--finalfc', type=int, default=256,
                        help='final FC layer size.')
    parser.add_argument('--adamlr', type=float, default=0.001,
                        help='adam lr')

    FLAGS, unparsed = parser.parse_known_args()

    config = BasicConfig(**FLAGS.__dict__).load(sys.argv[1])
    FLAGS.__dict__.update(config)

    val = train()
    print(str(val))
    print_result(val)
    
    
    
