from __future__ import print_function
import sys, os

if len(sys.argv) < 3:
    print("\nYou need to pass input file and number of steps as arguments!\n")
    exit(1)

from keras.models import load_model
from methods import *
import numpy as np

exec(settings)


def sample(seed):
    generated = ""
    for i in range(sample_len):
        x = np.zeros((batch, seq_len, vocabulary))
        for seq_pos in range(seq_len):
            vocab_index = char_to_int[seed[seq_pos]]
            x[0, seq_pos, vocab_index] = 1
        prediction = model.predict(x, batch_size=batch, verbose=0)
        prediction = prediction[0]
        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / temperature
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        RNG_int = np.random.choice(range(vocabulary), p=prediction.ravel())

        next_char = int_to_char[RNG_int]
        generated = generated+next_char
        seed = seed[1:] + next_char
    return (generated)


inputf = sys.argv[1]
outputf = inputf + "enriched.abc"

sample_len = int(sys.argv[2])
model_filename = "bestmodel.h5"
model = load_model(model_filename)

dataset = open(data_file).read()
chars = sorted(list(set(dataset)))
total_chars = len(dataset)
vocabulary = len(chars)
seed = open(inputf).read()
loadedcontent = seed

print ("Model and input data loaded successfully!")
seed = convert(seed)
seed = seed[-seq_len:]

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
generated = ""

result = sample(seed)

print ("\nGenerated sequence: "+ result)

outfile = open(outputf, 'w')
outfile.write(loadedcontent+result)
outfile.close()