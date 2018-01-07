# Deep-Learning
# RNN
# Modulation Recognition


import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
import scipy.io
import sys, os
import keras
from keras.utils.np_utils import to_categorical
import json
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM, Flatten
from keras.layers import RepeatVector, Lambda, concatenate
from keras.layers import Merge, merge, Permute
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras import optimizers
from keras.regularizers import l2
from keras.regularizers import l1
from keras.layers.normalization import BatchNormalization
from keras.constraints import nonneg
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution3D, MaxPooling3D
import keras.backend as K


def lstm_layer(input_x,config,length,dim):
    stack = config['stack']
    bidir = config['bidir']
    temp_att = config['temp_att']
    lstm_dim = config['lstm_dim']
    stack_num = config['stack_num']

    iq = input_x
    if bidir and stack:
        xq = iq
        for i in xrange(stack_num-1):
            xq = Bidirectional(LSTM(lstm_dim, return_sequences=True))(xq)
        if temp_att:
            activations = Bidirectional(LSTM(lstm_dim, return_sequences=True))(xq)
            attention = TimeDistributed(Dense(1, activation='softmax'))(activations)
            attention = Flatten()(attention)
            attention = RepeatVector(lstm_dim*2)(attention)
            attention = Permute([2, 1])(attention)
            xq = merge([activations, attention], mode='mul')
            xq = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(lstm_dim*2,))(xq)
        else:
            xq = Bidirectional(LSTM(lstm_dim))(xq)
    elif bidir and not stack:
        if temp_att:
            activations = Bidirectional(LSTM(lstm_dim, return_sequences=True))(iq)
            attention = TimeDistributed(Dense(1, activation='softmax'))(activations)
            attention = Flatten()(attention)
            attention = RepeatVector(lstm_dim*2)(attention)
            attention = Permute([2, 1])(attention)
            xq = merge([activations, attention], mode='mul')
            xq = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(lstm_dim*2,))(xq)
        else:
            xq = Bidirectional(LSTM(lstm_dim))(iq)
    elif not bidir and stack:
        xq = iq
        for i in xrange(stack_num-1):
            xq = LSTM(lstm_dim, return_sequences=True)(xq)
        if temp_att:
            activations = LSTM(lstm_dim, return_sequences=True)(xq)
            attention = TimeDistributed(Dense(1, activation='softmax'))(activations)
            attention = Flatten()(attention)
            attention = RepeatVector(lstm_dim)(attention)
            attention = Permute([2, 1])(attention)
            xq = merge([activations, attention], mode='mul')
            xq = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(lstm_dim,))(xq)
        else:
            xq = LSTM(lstm_dim)(xq)

    elif not bidir and not stack:
        if temp_att:
            activations = LSTM(lstm_dim, return_sequences=True)(iq)
            attention = TimeDistributed(Dense(1, activation='softmax'))(activations)
            attention = Flatten()(attention)
            attention = RepeatVector(lstm_dim)(attention)
            attention = Permute([2, 1])(attention)
            xq = merge([activations, attention], mode='mul')
            xq = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(lstm_dim,))(xq)
        else:
            xq = LSTM(lstm_dim)(iq)
    return xq

# load our dataset
sss = scipy.io.loadmat('Data_modulation_recognation_0dB_50000.mat')
print sss
X = numpy.array(sss['Data_modulation_recognation_0dB_50000'])
print X.shape
n = 50000
d = 100
lstm = True
y = X[:n,100]
X = X[:n,:d]
y = y - 1
print y
print list(y).count(0)
print list(y).count(1)
print list(y).count(2)
print list(y).count(3)
print list(y).count(4)

#print numpy.mean(X[10000]),y[10000]
#print numpy.mean(X[40000]),y[40000]

y = keras.utils.np_utils.to_categorical(y,5)
if lstm:
	Xreal = (X.real).reshape([n,d,1])
	Ximag = (X.imag).reshape([n,d,1])
	X = numpy.concatenate([Xreal,Ximag],axis=2)
else:
	Xreal = (X.real).reshape([n,d])
	Ximag = (X.imag).reshape([n,d])
	X = numpy.concatenate([Xreal,Ximag],axis=1)
print Xreal.shape
print Ximag.shape

rand = numpy.arange(len(X))
numpy.random.shuffle(rand)
X = X[rand]
y = y[rand]
split = 3000
Xv = X[:split]
yv = y[:split]
X = X[split:]
y = y[split:]
print X.shape
print y.shape
print Xv.shape
print yv.shape
#assert False

config = dict()
stack = config['stack'] = True
bidir = config['bidir'] = True
temp_att = config['temp_att'] = True

lstm_dim = config['lstm_dim'] = 12
stack_num = config['stack_num'] = 2
dense_dim = config['dense_dim'] = 64

# create model
drop = 0.0
reg = 0.0
if lstm:
	it = Input(shape=(d,2,))
        xt = lstm_layer(it,config,d,2)
	xt = Dropout(drop)(xt)
	xt = Dense(16, activation='relu', W_regularizer=l2(reg))(xt)
	xt = Dense(16, activation='relu', W_regularizer=l2(reg))(xt)
        out = Dense(5, activation='softmax', W_regularizer=l2(reg))(xt)
	model = Model(it,out)
else:
	model = Sequential()
	model.add(Dense(128, input_dim=d*2, activation='relu', W_regularizer=l2(reg)))
	model.add(Dropout(drop))
	model.add(Dense(64, activation='relu', W_regularizer=l2(reg)))
	model.add(Dropout(drop))
	model.add(Dense(32, activation='relu', W_regularizer=l2(reg)))
	model.add(Dropout(drop))
	model.add(Dense(5, activation='softmax', W_regularizer=l2(reg)))

# Compile model
adam = keras.optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Fit the model
model.fit(X, y, validation_data=(Xv,yv), epochs=1000, batch_size=128, verbose=2)
# evaluate the model
scores = model.evaluate(Xv, yv)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
