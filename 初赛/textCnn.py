from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
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
from sklearn.preprocessing import OneHotEncoder
import keras

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


def getOnePredict(source):
    result = []
    for i in source:
        result_item = np.zeros(i.size)
        result_item[i.argmax()] = 1
        result.append(result_item)
    return np.array(result)

def onehot2index(source):
    result = np.zeros(source.shape[0])
    for i in range(source.shape[0]):
        result[i] = source[i].argmax()
    return result

from keras import backend as K

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

            label.append(int(l[5]))
            title = l[2].split(',')#e.g. ['w12','w23']
            '''content = l[4].split(',')'''
            new_result = np.zeros(36)
            
            #format to delete 'w'
            for i in range(len(title)):
                if i >= 36:
                    break
                new_result[i] = title[i][1:]

            '''for i in range(len(title),len(content)+len(title)):
                if i >= 120:
                    break
                new_result[i] = content[i-len(title)][1:]    '''        
            
            data.append(new_result)
            if count%10000 == 0:
                print(count)
            count+=1
    
    from sklearn.preprocessing import OneHotEncoder
    train_label = np.array(label)
    ohe = OneHotEncoder()
    ohe.fit([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
    train_label = ohe.transform(train_label.reshape(140561,1)).toarray()
    
    return (np.array(data),train_label)

(data,label) = getData()

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.05, random_state=42)

from keras.layers import BatchNormalization

main_input = Input(shape=(36,), dtype='float64')
embedder = Embedding(353717 + 1, 256, input_length = 36)
embed = embedder(main_input)

# cnn1模块，kernel_size = 3
conv1_1 = Convolution1D(256, 3, padding='same')(embed)
bn1_1 = BatchNormalization()(conv1_1)
relu1_1 = Activation('relu')(bn1_1)
conv1_2 = Convolution1D(128, 3, padding='same')(relu1_1)
bn1_2 = BatchNormalization()(conv1_2)
relu1_2 = Activation('relu')(bn1_2)
cnn1 = MaxPool1D(pool_size=4)(relu1_2)

# cnn2模块，kernel_size = 4
conv2_1 = Convolution1D(256, 4, padding='same')(embed)
bn2_1 = BatchNormalization()(conv2_1)
relu2_1 = Activation('relu')(bn2_1)
conv2_2 = Convolution1D(128, 4, padding='same')(relu2_1)
bn2_2 = BatchNormalization()(conv2_2)
relu2_2 = Activation('relu')(bn2_2)
cnn2 = MaxPool1D(pool_size=4)(relu2_2)

# cnn3模块，kernel_size = 5
conv3_1 = Convolution1D(256, 5, padding='same')(embed)
bn3_1 = BatchNormalization()(conv3_1)
relu3_1 = Activation('relu')(bn3_1)
conv3_2 = Convolution1D(128, 5, padding='same')(relu3_1)
bn3_2 = BatchNormalization()(conv3_2)
relu3_2 = Activation('relu')(bn3_2)
cnn3 = MaxPool1D(pool_size=4)(relu3_2)

# 拼接三个模块
cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)
flat = Flatten()(cnn)

# 此处concantenate上强监督

drop = Dropout(0.5)(flat)
fc = Dense(512)(drop)
bn = BatchNormalization()(fc)
main_output = Dense(10, activation='softmax')(bn)

model = Model(inputs = main_input, outputs = main_output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[f1])

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1,
                    validation_data=(x_test, y_test))

name = 'first_' + str(history.history['f1'][-1])[2:]+'.h5'
model.save(name)

count = 1
f1Max = 0
stop = 3

while True:
    
    history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1,
                    validation_data=(x_test, y_test))
    
    if count % 2==0:
        name = 'first_' + str(history.history['f1'][-1])[2:]+'.h5'
        model.save(name)
    
    if history.history['f1'][0] > f1Max:
        f1Max = history.history['f1'][0]
        stop = 3
    else:
        stop-=1
        
    if stop == 0:
        break
    
    count+=1