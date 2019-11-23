# Filename: 20_generate.py
# Functie: script to generate the notes
# Remark: Intial source based on Tensorflow v1 usage
#         script converted tf 1.x to 2.x

""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord

import tensorflow as tf
# from keras.models import Sequential # tensorflow v1

# Because of pylint issues in Visual code use
# Use <cntrl> <shift> p lint, to change lint to bandit
# use pip install bandit in Visual code to install bandit

# v1
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Activation
# error no module named tensorflow.keras.utils.vis_utils
# from tensorflow.keras.utils.vis_utils import plot_model
# from tensorflow.keras.utils import plot_model

assert str(tf.version.VERSION)[:1] == '2', "this script requires tensorflow 2.x"
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

    #  see keras tensorflow v2 documentation
    #  https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    print("OK1 for create_network() Sequential()")
    # create_network

    # issues "Skipping optimization due to error while loading function libraries: Invalid argument: Functions"
    #        see https://github.com/tensorflow/tensorflow/issues/30263 
    #        work arround in all tf.keras.layers.LSTM calls
    #        change param activation from tf.nn.tanh to None
    print("This script has Still an issue: ")
    print("Skipping optimization due to error while loading function libraries: Invalid argument: Functions")
    print("https://github.com/tensorflow/tensorflow/issues/30263") 
    model = tf.keras.models.Sequential([  # tensorflow v2
       tf.keras.layers.LSTM(
         # 512, orgineel tf v1
          512 # aantal nodes in layer uit artikel v1; Geldit ook voor v2?
         ,input_shape=(network_input.shape[1], network_input.shape[2]) # zie artikel. Geldt dit ik ook voor v2?
                                                                       # first layer need this parameter
         ,return_sequences=True # also tensorflow v2 LSTM argument
         ,activation=None # see issue. This is a workarround 
       )
      ,tf.keras.layers.Dropout(0.3)
      ,tf.keras.layers.LSTM( 512
                           ,return_sequences=True
                           ,activation=None) # see issue. This is a workarround 
      ,tf.keras.layers.Dropout(0.3)
      ,tf.keras.layers.LSTM( 512
                            ,activation=None # see issue. This is a workarround
                           )
      ,tf.keras.layers.Dense(256) # For tf 2.0
                                  # activation: Activation function to use.
                                  # If you don't specify anything,
                                  # no activation is applied (ie. "linear" activation: a(x) = x).
                                  # check if this also valid voor tf 1.0
      ,tf.keras.layers.Dropout(0.3)
      ,tf.keras.layers.Dense( n_vocab
                             ,activation=tf.nn.softmax
                            )
      #tf.keras.layers.Activation('softmax') # This is move to previous line
    ])
    
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.compile( optimizer=tf.keras.optimizers.RMSprop()  # Optimizer
                  ,loss=tf.keras.losses.CategoricalCrossentropy() # Loss function to minimize
                  ,metrics=['accuracy'] # added
                 )

    # Visualize Model
    print(model.summary())
    #plot_model(model, to_file=homeDir+'model_plot.png', show_shapes=True, show_layer_names=True)
    #print("See: model_plot.png")

    print("Tot hier OK1 en geen fout melding")
    # Load the weights to each node

    # Zie artikel Jason Brownlee mbt 2 methode mbt leerproces tav
    # veiligstellen van de gewichten in een hdf5 file
    # Zie hoofdstuk 14.
    # zie Deep Learning With Python
    #     Develop Deep Learning Models on
    #     Theano and TensorFlow using Keras
    #     Jason Brownlee
    #
    # Zie pagina 96 van pdf
    # model.load_weights(homeDir+'weights-improvement-01-3.7268-bigger.hdf5')
    model.load_weights(homeDir+'weights-best.hdf5')
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
    sheetmusic.show()
    print("20_generate.py execute succesfully")

if __name__ == '__main__':
    print("tf.version.VERSION: ", tf.version.VERSION)
    print("tf.version.GIT_VERSION: ", tf.version.GIT_VERSION)    
    print("Eager execution running: ", tf.executing_eagerly())
    print("keras version: ", tf.keras.__version__)
    generate()
