import numpy as np
import pretty_midi
from pandas import DataFrame
from pandas import concat
import keras.backend.tensorflow_backend as tfb
import tensorflow as tf


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def convert(path):
    timeres = 20
    midi_data = pretty_midi.PrettyMIDI(path)
    roll = midi_data.get_piano_roll(fs=timeres)
    roll= np.where(roll>0 ,1,0)
    while(np.all(roll[:,0] == 0)): #drop leading "0" columns
        roll = np.delete(roll,0,1)
    return np.transpose(roll[40:100]) #pitch LtR, time UtD


def convert_back(roll,path):
    midi_out_path = path.split('.')[0] + "-enriched.midi"
    timeres = 20
    roll = np.transpose(roll)
    roll= np.where(roll>0.5 ,127,0)
    leading_zeros=np.zeros([40,roll.shape[1]])
    roll = np.vstack((leading_zeros, roll))
    bck = piano_roll_to_pretty_midi(roll, fs = timeres)
    bck.write(midi_out_path)

