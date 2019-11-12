# Filename: 10_train.py
# Functie: script te train the network
# Remark: Initial source based on Tensorflow v1 usage

# To run tensorflow for cpu and Ubuntu Linux platform
# you can use prebuild tensorflow package.
# You have to manually build the tensorflow package.
# Prebuild tenosrflow package on Linux 
# gives a runtime warning about CPU extentions AVX2, FMA.
# See article https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
#
# Build a TensorFlow pip package from source and install it on Ubuntu Linux and macOS
# See https://www.tensorflow.org/install/source

# See https://docs.bazel.build/versions/master/install.html
# See https://docs.bazel.build/versions/master/install-ubuntu.html#install-with-installer-ubuntu
# See http://bazel.build/docs/getting-started.html to start a new project!

# used commands to install:
# download bazel install script # see https://docs.bazel.build/versions/master/install-ubuntu.html#install-with-installer-ubuntu
# install bazel install script with
# chmod +x bazel-<version>-installer-linux-x86_64.sh
# ./bazel-<version>-installer-linux-x86_64.sh --user
# cd <project directory> 
# without a git respository
# git clone https://github.com/tensorflow/tensorflow.git
# with a git repository
# git submodule add https://github.com/tensorflow/tensorflow.git
# cd tensorflow
# for release 2.0 tensorflow
# git checkout r2.0
# ./configure
# build tensorflow package. Beware to use the correct command depending on with version
# bazel build //tensorflow/tools/pip_package:build_pip_package
# create wheel file in /tmp/tensorflow_pkg
# bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# Install the package tensorflow-2.0.0-cp36-cp36m-linux_x86_64.whl
# pip install /tmp/tensorflow_pkg/tensorflow-2.0.0-cp36-cp36m-linux_x86_64.whl

# Info after successful build of release r2.0
# INFO: Elapsed time: 11160.249s, Critical Path: 239.71s
# INFO: 17804 processes: 17804 local.
# INFO: Build completed successfully, 18538 total actions

""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle

import numpy

from music21 import chord, converter, instrument, note
import tensorflow as tf
# from keras.callbacks import ModelCheckpoint # tensorflow v1
# from keras.layers import LSTM, Activation, Dense, Dropout # tensorflow v1
# from keras.models import Sequential  # tensorflow v1
# from keras.utils import utils # tensorflow v1. this does not work use tf.keras.utils.<function> as call

# homeDir when this script is used from my Laptop
homeDir = '/home/claude/Documents/sources/python/python3/python3_Muziek_Generator/MLMG/'
# homeDir when this script is used from my Virtualbox Linux VM
#homeDir = '/home/test/Documents/sources/python/python3/cx1964ReposMusic21MLMG/'

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()
    
    # get amount of pitch names
    n_vocab = len(set(notes))
    
    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob(homeDir+"midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open(homeDir+'data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes # return een list of notes. Offset informatie (=tijd) gaat verloren

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100 # voorspel de volgende noot obv 100 voorgaande noten

    # get all unique pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    #        for number, note in enumerate(pitchnames) genereert een reeks met elementen inde vorm <rangnummer>, <pitchname>
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        # hier worden list met notes mapped naar een list met integers mbv gebruikt van de note_to_int dict
        # mapping is nodig omdat neural netwerk met integers werkt om de gewichten uit te kunnen rekenen.
        network_input.append([note_to_int[char] for char in sequence_in])
        
        # hier worden list met notes mapped naar een list met integers mbv gebruikt van de note_to_int dict
        # mapping is nodig omdat neural netwerk met integers werkt om de gewichten uit te kunnen rekenen.
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
 
    # normalize input
    network_input = network_input / float(n_vocab)

    print("network_output:",network_output )
    # Converts a class vector (integers) to binary class matrix
    # See https://www.tensorflow.org/api_docs/python/tf/keras/utils
    #network_output = utils.to_categorical(network_output) # tbv tf 1
    network_output = tf.keras.utils.to_categorical(network_output) # tf v2

    return (network_input, network_output) # return input and output list with mapped notes


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    # Zie orginele definitie create_network
    # https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
    # https://github.com/Skuldur/Classical-Piano-Composer

    # zie alternatief https://adgefficiency.com/tf2-lstm-hidden/
 
    # ToDo: uitzoeken betekenis argumenten tf.keras.layers.LSTM call
    #       aanleiding melding tijdens trainen invalid argument
    #       kan ook bij code zitten die training op start (zie verder)

    # Uit artikel mbt input_shape in onderstaande tf.keras.layers.LSTM(
    # For the first layer we have to provide a 
    # unique parameter called input_shape. The purpose of the parameter is
    # to inform the network of the shape of the data it will be training.
    # Geldt ook voor tensorflow v2 ????

    # model = Sequential() # tensorflow v1
    model = tf.keras.models.Sequential([  # tensorflow v2
      tf.keras.layers.LSTM(
        # 512, orgineel tf v1
        512, # aantal nodes in layer uit artikel v1; Geldit ook voor v2?
        input_shape=(network_input.shape[1], network_input.shape[2]), ## zie artikel. Geldt dit ik ook voor v2?
        return_sequences=True # also tensorflow v2 LSTM argument
      ),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.LSTM(512, return_sequences=True),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.LSTM(512),
      tf.keras.layers.Dense(256),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(n_vocab),
      tf.keras.layers.Activation('softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
 
    # show used model
    model.summary()

    return model

def train(model, network_input, network_output):
    """ train the neural network """

    # Zie artikel Jason Brownlee 2 methoden mbt leerproces tav veiligstellen van de
    # gewichten in een hdf5 file obv ModelCheckpoint.
    # zie Deep Learning With Python
    #     Develop Deep Learning Models on
    #     Theano and TensorFlow using Keras
    #     Jason Brownlee
    # Zie pagina 96 van pdf

    # methode1: creeer meerdere hdf5 files
    # filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    # methode2: creeer een hdf5 file
    # Zie pagina 95 - 96 van pdf (14.3 Checkpoint best Neural Network Model only)
    # De file waarin tgv het checkpoint process tijdens de leerfase de weights in worden weggeschreven.
    filepath = "weights-best.hdf5"

    # Zie ook paragraaf 14.2 en 14.3
    # mbt gebruik parameters (monitor='val_acc' and  mode='max') pagina 94 - 96
    # Zie https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    # checkpoint = ModelCheckpoint( # tensorfow v1
    checkpoint = tf.keras.callbacks.ModelCheckpoint(  # tensorflow v2    
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='max'
    )
    callbacks_list = [checkpoint]
    # het leer proces
    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    print("Tensorflow version: ", tf.__version__)
    print("Eager execution running: ", tf.executing_eagerly())
    print("keras version: ", tf.keras.__version__)
    train_network()
