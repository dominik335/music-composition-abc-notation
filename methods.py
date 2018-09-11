settings = '''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
sample_len = 100
gpu_restrict = True
use_previous_model = 0
timesteps = 60
batch = 600
dropout_rate = 0.5
epochs = 3
select_size = 0

hidden_layers = 2
learning_rate = 0.01 
neurons = [10,20]
temperature = 0.1
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

