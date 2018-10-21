from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import sklearn
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder

import keras
from keras import initializers, regularizers, constraints
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
from keras.engine import InputSpec
from keras.models import load_model
import sys


def getOnePredict(source):
    result = []
    for i in source:
        result_item = np.zeros(i.size)
        result_item[i.argmax()] = 1
        result.append(result_item)
    return np.array(result)


def onehot2index(source):
    result = np.zeros(source.size)
    for i in range(len(source)):
        result[i] = source[i].argmax()
    return result


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

class Attention_layer(Layer): 
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
 
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
 
        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
 
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
 
        super(Attention_layer, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
 
    def call(self, x, mask=None):
        uit = K.dot(x, self.W)
 
        if self.bias:
            uit += self.b
 
        uit = K.tanh(uit)
 
        a = K.exp(uit)
 
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
 
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        #print(a)
        # a = K.expand_dims(a)
        #print(x)
        weighted_input = x * a
        #print weighted_input
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def getUploadData(mode):
    data = []

    with open('test_a.txt') as f:
        isFirst = True
        while True:
            line = f.readline()

            if isFirst:
                isFirst = False
                continue

            if not line:
                break

            l = line.split('\t')

            title = l[2].split(',')  # e.g. ['w12','w23']
            content = []
            if mode == 1:
                content = l[4].split(',')

            if mode == 0:
                new_result = np.zeros(36)
            else:
                new_result = np.zeros(120)

            # format to delete 'w'
            for i in range(len(title)):
                if i >= 36:
                    break
                new_result[i] = title[i][1:]

            if mode == 1:
                for i in range(len(title), len(content) + len(title)):
                    if i >= 120:
                        break
                    new_result[i] = content[i - len(title)][1:]

            data.append(new_result)

    return np.array(data)



data_36 = getUploadData(0)
data_120 = getUploadData(1)


name = sys.argv[1]
mode = sys.argv[2]

print(name)
print(mode)

model = None


model = keras.models.load_model(name, custom_objects={'Attention_layer':Attention_layer,'f1': f1})
if mode == '0':
    result = model.predict(data_36)
else:
    result = model.predict(data_120)

np.save(name.split('.')[0], result)