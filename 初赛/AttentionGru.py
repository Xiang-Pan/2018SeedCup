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
#X_train, X_test, y_train, y_test = train_test_split(title, label, test_size=0.1, random_state=42)

import keras
from keras import initializers, regularizers, constraints
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
from keras.engine import InputSpec
MAX_LENGTH = 120

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
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#get word level data
def getData():
    label = []
    data = []
    #firstLabel = []
    
    with open('train_a_c.txt') as f:
        isFirst = True
        count = 1
        while True:
            line = f.readline()

            if isFirst:
                isFirst = False
                continue            

            if not line:
                break

            l = line.split('\t')
            
            #firstLabel.append(int(l[5]))
            label.append(int(l[7]))
            title = l[2].split(',')#e.g. ['w12','w23']
            content = l[4].split(',')
            new_result = np.zeros(MAX_LENGTH)
            
            #format to delete 'w'
            for i in range(len(title)):
                if i >= MAX_LENGTH:
                    break
                new_result[i] = title[i][1:]

            for i in range(len(title),len(content)+len(title)):
                if i >= MAX_LENGTH:
                    break
                new_result[i] = content[i-len(title)][1:]
                        
            data.append(new_result)
            if count%10000 == 0:
                print(count)
            count+=1
    
    #change lable to one hot
    label = np.array(label)
    ohe = OneHotEncoder()
    example = []
    for i in range(10+64+1,10+64+125+1):
        example.append([i])
    ohe.fit(example)
    label = ohe.transform(label.reshape(label.size,1)).toarray()
    
    return (np.array(data),label)

	
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
		
		
(data,label) = getData()

from sklearn.cross_validation import train_test_split
import keras
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.08, random_state=42)

class KMaxPooling(Layer):
    """
    k-max-pooling for 3 dimention
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)



main_input = Input(shape=(MAX_LENGTH,), dtype='float64')
embed = Embedding(353717+1, 378, input_length=MAX_LENGTH)(main_input)
#gru = Bidirectional(GRU(100, dropout=0.2, return_sequences=True))(embed)
gru = Bidirectional(keras.layers.CuDNNGRU(100, return_sequences=True))(embed)
attention = Attention_layer()(gru)
main_output = Dense(125, activation='softmax')(attention)
model = Model(inputs = main_input, outputs = main_output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[f1])

history = model.fit(x=x_train, y=y_train,
          batch_size = 32,epochs=1, 
         validation_data=(x_test,y_test))

name = 'atten_' + str(history.history['f1'][-1])[2:]+'.h5'
model.save(name)
		 
#auto train
count = 1
f1Max = 0
stop = 3

while True:
    
    history = model.fit(x=x_train, y=y_train,
          batch_size = 32,epochs=1, 
         validation_data=(x_test,y_test))
    
    name = 'atten_' + str(history.history['f1'][-1])[2:]+'.h5'
    model.save(name)
    
    if history.history['f1'][0] > f1Max:
        f1Max = history.history['f1'][0]
        stop = 3
    else:
        stop-=1
        
    if stop == 0:
        break
    
    count+=1








