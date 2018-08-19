import keras.backend.tensorflow_backend as tfb
import tensorflow as tf

sett = '''
sample_len = 100
gpu_restrict = True
# gpu_restrict = False
use_previous_model = 0
timesteps = seq_len = 60
batch = 2500
dropout_rate = 0.5
epochs = 3
select_size = 0

hidden_layers = 2
learning_rate = 0.01 
neurons = [150,200]
#neurons = [15, 10]
temperature = 0.6
#data_file = "cleaned"
data_file = "jazz"

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

