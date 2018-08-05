# Multilayered (or singlelayered) GRU based RNN for text generation using Keras libraries

# import libraries
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import GRU
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import random
import sys
import os
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)

# load Dataset
path = "hello"  # raw_input("Enter file name (example: Wittgenstein.txt) for training and testing data (make sure it's in the same directory):\n ")
dataset = open(path).read()

# store the list of all unique characters in dataset
chars = sorted(list(set(dataset)))

total_chars = len(dataset)
vocabulary = len(chars)

print("Total Characters: ", total_chars)
print("Vocabulary: ", vocabulary)

# Creating dictionary or map in order to map all characters to an integer and vice versa
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

mapvar = "y"  # raw_input("Do you want to see the character to integer map? (y/n): ")

if mapvar == "y" or mapvar == "Y":
    # Show the map from Char to Int
    print('\nGenerated Map: ')
    for char in chars:
        print(char, ' is mapped to ', char_to_int[char], ' and vice versa.')

# Asking the important questions
sample_len = 30  # int(raw_input("\nLength of sample text: "))
temperature = 1  # float(raw_input("Temperature: "))
print("\nChoose:")
print("Enter 0 to create a model and train it from the beginning.")
print("Enter 1 to generate texts from saved model and weights.")
print("Enter 2 to resume training using last saved model and weights.")
Answer = 0  # int(raw_input('Enter: '))

if Answer == 0:
    hidden_layers = 3  # int(input("\nNumber of Hidden Layers (Minimum 1): "))
    neurons = []
    if hidden_layers == 0:
        hidden_layers = 1;
    for i in range(0, hidden_layers):
        pass  # neurons.append(int(input("Number of Neurons in Hidden Layer "+str(i+1)+": ")))
    seq_len = 20  # int(input("Time Steps: "))
    learning_rate = 0.01  # float(raw_input("Learning Rate: "))
    dropout_rate = 0.2  # float(raw_input("Dropout Rate: "))
    batch = 2000  # int(raw_input("Training Batch Size: "))
neurons = [1500, 1000, 500]

if Answer != 0:
    try:
        f = open('GRUModelInfo', "r")
        lines = f.readlines()
        for i in lines:
            thisline = i.split(" ")
        seq_len = int(thisline[0])
        batch = int(thisline[1])
        f.close()
    except:
        print("\nUh Oh! Caught some exceptions! May be you are missing the file having time step information")
        seq_len = 20  # int(input("Time Steps (I hope, you remember what it was): "))
        batch = 500  # int(input("Training batch size(I hope, you remember what it was): "))
        f = open('GRUModelInfo', 'w+')
        f.write(str(seq_len) + " " + str(batch))
        f.close()

if Answer == 0 or Answer == 2:

    # Doing some maths so that the total patterns in future become DIVISIBLE by batch size
    # total no. of patterns need to divisible by batch size because each batch must be of the same size..
    # ...so that the RNN layer can be 'Stateful'

    index = int((total_chars - seq_len) / batch)
    index = batch * index
    dataset = dataset[:index + seq_len]
    total_chars = len(dataset)

    # prepare input data and output(target) data
    # (X signified Inputs and Y signifies Output(targeted-output in this case)
    dataX = []
    dataY = []

    for i in range(0, total_chars - seq_len):  # Example of an extract of dataset: Language
        dataX.append(dataset[i:i + seq_len])  # Example Input Data: Languag
        dataY.append(dataset[i + seq_len])  # Example of corresponding Target Output Data: e

    total_patterns = len(dataX)
    print("\nTotal Patterns: ", total_patterns)

    # One Hot Encoding...
    X = np.zeros((total_patterns, seq_len, vocabulary), dtype=np.bool)
    Y = np.zeros((total_patterns, vocabulary), dtype=np.bool)

    for pattern in range(total_patterns):
        for seq_pos in range(seq_len):
            vocab_index = char_to_int[dataX[pattern][seq_pos]]
            X[pattern, seq_pos, vocab_index] = 1
        vocab_index = char_to_int[dataY[pattern]]
        Y[pattern, vocab_index] = 1

if Answer == 0:
    # build the model: a multi(or single depending on user input)-layered GRU based RNN
    print('\nBuilding model...')

    model = Sequential()
    if hidden_layers == 1:
        model.add(GRU(neurons[0], batch_input_shape=(batch, seq_len, vocabulary), stateful=True))
    else:
        model.add(GRU(neurons[0], batch_input_shape=(batch, seq_len, vocabulary), stateful=True, return_sequences=True))
    model.add(Dropout(dropout_rate))
    for i in range(1, hidden_layers):
        if i == (hidden_layers - 1):
            model.add(GRU(neurons[i], stateful=True))
        else:
            model.add(GRU(neurons[i], stateful=True, return_sequences=True))
        model.add(Dropout(dropout_rate))

    model.add(Dense(vocabulary))
    model.add(Activation('softmax'))
    my_optimizer = RMSprop(lr=learning_rate)
    my_optimizer = Adam()
    model.compile(loss='binary_crossentropy', optimizer=my_optimizer)

    # save model information
    model.save('GRUModel.h5')
    f = open('GRUModelInfo', 'w+')
    f.write(str(seq_len) + " " + str(batch))
    f.close()

else:
    print('\nLoading model...')
    try:
        model = load_model('GRUModel.h5')
    except:
        print("\nUh Oh! Caught some exceptions! May be you don't have any trained and saved model to load.")
        print("Solution: May be create and train the model anew ?")
        sys.exit(0)

model.summary()

# define the checkpoint
filepath = "BestGRUWeights.h5"  # Best weights for sampling will be saved here.
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True,
                             mode='min')


# Function for creating a sample text from a random seed (an extract from the dataset).
# The seed acts as the input for the GRU RNN and after feed forwarding through the network it produces the output
# (the output can be considered to be the prediction for the next character)

def sample(seed):
    for i in range(sample_len):
        # One hot encoding the input seed
        x = np.zeros((batch, seq_len, vocabulary))
        for seq_pos in range(seq_len):
            vocab_index = char_to_int[seed[seq_pos]]
            x[0, seq_pos, vocab_index] = 1
        # procuring the output (or prediction) from the network
        prediction = model.predict(x, batch_size=batch, verbose=0)
        prediction = prediction[0]

        # The prediction is an array of probabilities for each unique characters.

        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / temperature  # Scaling prediction values with 'temperature'
        # to manipulate diversities.
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)

        # Randomly an integer(mapped to a character) is chosen based on its likelihood
        # as described in prediction list

        RNG_int = np.random.choice(range(vocabulary), p=prediction.ravel())

        # The next character (to generate) is mapped to the randomly chosen integer
        # Procuring the next character from the dictionary by putting in the chosen integer
        next_char = int_to_char[RNG_int]

        # Display the chosen character
        sys.stdout.write(next_char)
        sys.stdout.flush()
        # modifying seed for the next iteration for finding the next character
        seed = seed[1:] + next_char

    print()


if Answer == 0 or Answer == 2:
    if Answer == 2:
        filename = "GRUWeights.h5"
        try:
            model.load_weights(filename)
        except:
            print("\nUh Oh! Caught some exceptions! May be you don't have any trained and saved weights to load.")
            print("Solution: May be create and train the model anew ?")
            sys.exit(0)
    # Train Model and print sample text at each epoch.
    for iteration in range(1, 60):
        print()
        print('Iteration: ', iteration)
        print()

        # Train model. If you have forgotten: X = input, Y = targeted outputs
        model.fit(X, Y, batch_size=batch, nb_epoch=15, shuffle=False, callbacks=[checkpoint])
        model.save_weights(
            'GRUWeights.h5')  # Saving current model state so that even after terminating the program; training
        # can be resumed for last state in the next run.
        print()

        # Randomly choosing a sequence from dataset to serve as a seed for sampling
        start_index = random.randint(0, total_chars - seq_len - 1)
        seed = dataset[start_index: start_index + seq_len]

        sample(seed)
else:
    # load the network weights
    filename = "BestGRUWeights.h5"
    try:
        model.load_weights(filename)
    except:
        print("\nUh Oh! Caught some exceptions! May be you don't have any trained and saved weights to load.")
        print("Solution: May be create and train the model anew ?")
        sys.exit(0)
    Answer2 = "y"
    while Answer2 == "y" or Answer2 == "Y":
        print("\nGenerating Text:\n")
        # Randomly choosing a sequence from dataset to serve as a seed for sampling
        start_index = random.randint(0, total_chars - seq_len - 1)
        seed = dataset[start_index: start_index + seq_len]
        sample(seed)
        print()
        Answer2 = input("Generate another sample Text? (y/n): ")