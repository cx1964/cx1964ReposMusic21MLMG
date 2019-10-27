# Filename: 20_generate.py
# Functie: script to generate the notes
# Remark: Intial source based on Tensorflow v1 usage

""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord

from tensorflow.keras.models import Sequential # tensorflow v2
# from keras.models import Sequential # tensorflow v1
# When using from keras.models import Sequential with tensorflow
# create message: AttributeError: module 'tensorflow' has no attribute 'get_default_graph'
# see: https://stackoverflow.com/questions/55496289/how-to-fix-attributeerror-module-tensorflow-has-no-attribute-get-default-gr

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils.vis_utils import plot_model

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
    # todo: tgv overgang tensorflow 1 ->2 gaat in create_network() wat mis
    model = create_network(normalized_input, n_vocab)
    '''
    todo: aanzetten
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)
    '''

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

    #  see keras tensorflow v2 documentation https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    print("OK1 for create_network() Sequential()")
    # create_network
    model = Sequential() # this also works for tensorflow v2
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
    
    # Visualize Model
    print(model.summary())
    plot_model(model, to_file=homeDir+'model_plot.png', show_shapes=True, show_layer_names=True)
    print("See: model_plot.png")

    print("Tot hier OK1 en geen fout melding")
    # Load the weights to each node

    #model.load_weights(homeDir+'weights.hdf5') # Deze regel creeert een fout
    # Zie artikel Jason Brownlee mbt 2 methode mbt leerproces tav veiligstellen van de gewichten in een hdf5 file
    model.load_weights(homeDir+'weights-improvement-01-3.7268-bigger.hdf5')
    print("OK final: create_network()")
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
    #sheetmusic.show()

if __name__ == '__main__':
    generate()
