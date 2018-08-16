import numpy as np
import pretty_midi
from pandas import DataFrame
from pandas import concat
import keras.backend.tensorflow_backend as tfb
import tensorflow as tf

sett = '''
sample_len = 100
gpu_restrict = True
# gpu_restrict = False
use_previous_model = 0
timesteps = seq_len = 60
batch = 3000
dropout_rate = 0.3
epochs = 3
select_size = 0

hidden_layers = 3
learning_rate = 0.01  # float(raw_input("Learning Rate: "))
neurons = [100,200, 100]
#neurons = [15, 10]
temperature = 0.3
data_file = "cleaned"
sample_len = 30
model_filename = "BestGRU.h5" 

'''

def removelineswith(seed,pattern):
    out = ""
    for i in seed.split('\n'):
        if not i.startswith(pattern):
            out = out+ i
            out = out + '\n'
    out = out[:out.rfind('\n')]
    return out

def convert(seed):
    banned_list= ["T:", "%", "W:", "w:", "K:", "H:", "D:" "Z:", "R:"]
    for item in banned_list:
        seed = removelineswith(seed,item)
    return seed

