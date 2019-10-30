# Filename: 10_train.py
# Functie: script te train the network
# Remark: Intial source based on Tensorflow v1 usage

""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord

from tensorflow.keras.models import Sequential # tensorflow v2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint

homeDir = '/home/claude/Documents/sources/python/python3/python3_Muziek_Generator/MLMG/'

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()
    
    # get amount of pitch names
    n_vocab = len(set(notes))
    
    # ToDo: hieronder gaat iets mis
    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
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

    # ToDo: Hierondre gaat iets het mis
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output) # return input en output list met mapped notes


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

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

    # methode1: creeer meerder hdf5 files
    # filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    # methode2: creeer een hdf5 file
    # Zie pagina 95 - 96 van pdf (14.3 Checkpoint best Neural Network Model only)
    filepath = "weights-best.hdf5"

    # Zie ook paragraaf 14.2 en 14.3
    # mbt gebruik parameters (monitor='val_acc' and  mode='max') pagina 94 - 96
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='max'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
