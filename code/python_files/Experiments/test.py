import numpy as np
import os
import sys
import wave
import copy
import math

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
# from keras.layers import LSTM, Input, Flatten, Merge, Bidirectional
from keras.layers import LSTM, Input, Flatten, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import label_binarize

from features import *
from helper import *


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

code_path = os.path.dirname(os.path.realpath(os.getcwd()))
emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])
data_path = code_path + "/../data/sessions/"
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000

import pickle5

with open('/cap/data/data_collected.pickle', 'rb') as handle:
    data2 = pickle5.load(handle)

x_train_speech = []

counter = 0
for ses_mod in data2:
    x_head = ses_mod['signal']
    st_features = calculate_features(x_head, framerate, None)
    st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
    x_train_speech.append(st_features.T)
    counter += 1
    # if (counter % 100 == 0):
    #     print(counter)

x_train_speech = np.array(x_train_speech)
print(x_train_speech.shape)


def lstm_model(optimizer='Adadelta'):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(100, 34)))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dense(512))
    model.add(Activation('relu'))

    # classification
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

model = lstm_model()
print(model.summary())

Y = []
for ses_mod in data2:
    Y.append(ses_mod['emotion'])

Y = label_binarize(Y, emotions_used)

Y.shape

import time

start = time.time()
hist = model.fit(x_train_speech, Y,
                 batch_size=100, epochs=60, verbose=1, shuffle = True,
                 validation_split=0.2)

print("total time = " + str(time.time() - start))