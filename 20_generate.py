""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

homeDir = '/home/claude/Documents/sources/python/python3/python3_Muziek_Generator/MLMG/'

def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open(homeDir+'data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """

    # Dit statement genereert warning
    # WARNING:tensorflow:From /home/claude/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:74:
    # The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead
    model = Sequential()
    
    # Dit statement genereert warning:
    # WARNING:tensorflow:From /home/claude/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    #
    # WARNING:tensorflow:From /home/claude/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    #
    # OMP: Info #212: KMP_AFFINITY: decoding x2APIC ids.
    # OMP: Info #210: KMP_AFFINITY: Affinity capable, using global cpuid leaf 11 info
    # OMP: Info #154: KMP_AFFINITY: Initial OS proc set respected: 0-7
    # OMP: Info #156: KMP_AFFINITY: 8 available OS procs
    # OMP: Info #157: KMP_AFFINITY: Uniform topology
    # OMP: Info #179: KMP_AFFINITY: 1 packages x 4 cores/pkg x 2 threads/core (4 total cores)
    # OMP: Info #214: KMP_AFFINITY: OS proc to physical thread map:
    # OMP: Info #171: KMP_AFFINITY: OS proc 0 maps to package 0 core 0 thread 0 
    # OMP: Info #171: KMP_AFFINITY: OS proc 4 maps to package 0 core 0 thread 1 
    # OMP: Info #171: KMP_AFFINITY: OS proc 1 maps to package 0 core 1 thread 0 
    # OMP: Info #171: KMP_AFFINITY: OS proc 5 maps to package 0 core 1 thread 1 
    # OMP: Info #171: KMP_AFFINITY: OS proc 2 maps to package 0 core 2 thread 0 
    # OMP: Info #171: KMP_AFFINITY: OS proc 6 maps to package 0 core 2 thread 1 
    # OMP: Info #171: KMP_AFFINITY: OS proc 3 maps to package 0 core 3 thread 0 
    # OMP: Info #171: KMP_AFFINITY: OS proc 7 maps to package 0 core 3 thread 1 
    # OMP: Info #250: KMP_AFFINITY: pid 18881 tid 18881 thread 0 bound to OS proc set 0
    # OMP: Info #250: KMP_AFFINITY: pid 18881 tid 18887 thread 1 bound to OS proc set 1
    # OMP: Info #250: KMP_AFFINITY: pid 18881 tid 18888 thread 2 bound to OS proc set 2
    # OMP: Info #250: KMP_AFFINITY: pid 18881 tid 18889 thread 3 bound to OS proc set 3

    # ### ToDo ###
    # Nog zoeken naar command binnen deze functie in onderstaande code die ValueError veroorzaakt
    # Fout melding
    # ValueError: Dimension 1 in both shapes must be equal, but are 58 and 359.
    # Shapes are [256,58] and [256,359]. for 'Assign_11' (op: 'Assign') with input shapes: [256,58], [256,359].
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

    # Load the weights to each node
    model.load_weights(homeDir+'weights.hdf5')
    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=homeDir+'test_output.mid')

    # create sheetmusic
    sheetmusic = stream.Stream(output_notes)
    sheetmusic.show()

if __name__ == '__main__':
    generate()
