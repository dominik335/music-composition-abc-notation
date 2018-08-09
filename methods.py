import numpy as np
import pretty_midi
from pandas import DataFrame
from pandas import concat
import keras.backend.tensorflow_backend as tfb
import tensorflow as tf

sett = '''
gpu_restrict = True
# gpu_restrict = False
use_previous_model = 0
timesteps = seq_len = 20
batch = 4000
dropout_rate = 0.3
epochs = 3
select_size = 0

hidden_layers = 3
learning_rate = 0.01  # float(raw_input("Learning Rate: "))
neurons = [1500, 1000, 500]
neurons = [15, 10, 5]
temperature = 0.1
data_file = "cleaned"
sample_len = 30
model_filename = "BestGRU.h5" 

'''




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

