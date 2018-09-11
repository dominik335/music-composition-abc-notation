# import libraries
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import GRU
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.models import load_model
import numpy as np
import random
import sys
import os
import tensorflow as tf
from keras.callbacks import CSVLogger
from methods import *


exec(settings)

if gpu_restrict:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.Session(config=config)

dataset = open(data_file).read()
dataset = convert(dataset)

chars = sorted(list(set(dataset)))
total_chars = len(dataset)
vocabulary = len(chars)

print("Total number of characters: ", total_chars)
print("Vocabulary size: ", vocabulary)

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

index = int((total_chars - timesteps) / batch)
index = batch * index
dataset = dataset[:index + timesteps]
total_chars = len(dataset)

dataX = []
dataY = []

for i in range(0, total_chars - timesteps):
    dataX.append(dataset[i:i + timesteps])
    dataY.append(dataset[i + timesteps])


total_patterns = len(dataX)
print("\nTotal number of learning sequences: ", total_patterns)
index = int((0.9*len(dataX)) / batch)
index = index * batch
trainPortion = int(index)

# One Hot Encoding...
X = np.zeros((total_patterns, timesteps, vocabulary), dtype=np.bool)
Y = np.zeros((total_patterns, vocabulary), dtype=np.bool)

for pattern in range(total_patterns):
    for seq_pos in range(timesteps):
        vocab_index = char_to_int[dataX[pattern][seq_pos]]
        X[pattern, seq_pos, vocab_index] = 1
    vocab_index = char_to_int[dataY[pattern]]
    Y[pattern, vocab_index] = 1


Xtr = X[:trainPortion, :]
Xval = X[trainPortion:, :]

Ytr = Y[:trainPortion, :]
Yval = Y[trainPortion:, :]


if use_previous_model == 0:
    print('\nBuilding model...')

    model = Sequential()
    if hidden_layers == 1:
        model.add(GRU(neurons[0], batch_input_shape=(batch, timesteps, vocabulary), stateful=True))
    else:
        model.add(GRU(neurons[0], batch_input_shape=(batch, timesteps, vocabulary), stateful=True, return_sequences=True))
    model.add(Dropout(dropout_rate))
    for i in range(1, hidden_layers):
        if i == (hidden_layers - 1):
            model.add(GRU(neurons[i], activation='relu', stateful=True))
        else:
            model.add(GRU(neurons[i], stateful=True, return_sequences=True))
        model.add(Dropout(dropout_rate))
    model.add(Dense(vocabulary))
    model.add(Activation('softmax'))

    my_optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=my_optimizer,metrics=['acc'])


else:
    print('\nLoading model...')
    try:
        model = load_model(model_filename)
        print("Model loaded!")
    except:
        print("\nCouldn't load model! Exiting...")
        sys.exit(1)

model.summary()

# define the checkpoint
checkpoint = ModelCheckpoint(model_filename, monitor='loss', verbose=1, save_best_only=True, mode='min')
checkpoint2 = ModelCheckpoint(model_filename+"val.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#define logger
csv_logger = CSVLogger('log.csv', append=True, separator=';')

for iteration in range(1, 10):
        print()
        print('Iteration: ', iteration)
        print()
        model.fit(Xtr, Ytr, batch_size=batch, epochs= epochs, validation_data=(Xval,Yval), shuffle=False, callbacks=[checkpoint,checkpoint2, csv_logger])
        model.save('mymodel.h5')
