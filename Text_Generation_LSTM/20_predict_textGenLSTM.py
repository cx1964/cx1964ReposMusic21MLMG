# Filename: 20_predict_textGenLSTM.py
# Function: Text Generation With LSTM Recurrent Neural Networks
#           in Python with Keras
# Remark: This code comes from article
# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/   
# 
# This code has similarities with learning code of article
# https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
# see also 10_learn.py
#

# ### Generating Text with an LSTM Network ###
# 
# Generating text using the trained LSTM network is relatively straightforward.
#
# Firstly, we load the data and define the network in exactly the same way,
# except the network weights are loaded from a checkpoint file and the
# network does not need to be trained.
#
#
# # *** load the network weights ***
# filename = "weights-improvement-19-1.9435.hdf5"
# model.load_weights(filename)
# model.compile(  loss='categorical_crossentropy'
#                ,optimizer='adam'
#              )
#
# Also, when preparing the mapping of unique characters to integers, we must
# also create a reverse mapping that we can use to convert the integers back
# to characters so that we can understand the predictions.
#
# int_to_char = dict((i, c) for i, c in enumerate(chars))
#
# Finally, we need to actually make predictions.
#
# The simplest way to use the Keras LSTM model to make predictions is to first
# start off with a seed sequence as input, generate the next character then
# update the seed sequence to add the generated character on the end and trim
# off the first character. This process is repeated for as long as we want to
# predict new characters (e.g. a sequence of 1,000 characters in length).
#
# We can pick a random input pattern as our seed sequence, then print generated
# characters as we generate them.
# pick a random seed
# start = numpy.random.randint(0, len(dataX)-1)
# pattern = dataX[start]
# print "Seed:"
# print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# # generate characters
# for i in range(1000):
#	  x = numpy.reshape(pattern, (1, len(pattern), 1))
#	  x = x / float(n_vocab)
#	  prediction = model.predict(x, verbose=0)
#	  index = numpy.argmax(prediction)
#	  result = int_to_char[index]
#	  seq_in = [int_to_char[value] for value in pattern]
#	  sys.stdout.write(result)
#	  pattern.append(index)
#	  pattern = pattern[1:len(pattern)]
#print "\nDone."
#
# The full code example for generating text using the loaded LSTM model is
# listed below for completeness.
'''
text afmaken
'''

# The full code example for generating text using the loaded LSTM model is listed
# below for completeness.
#
# # ### Load LSTM network and generate text ###
# import sys
# import numpy
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.callbacks import ModelCheckpoint
# from keras.utils import np_utils
# # load ascii text and covert to lowercase
# filename = "wonderland.txt"
# raw_text = open(filename, 'r', encoding='utf-8').read()
# raw_text = raw_text.lower()
# # create mapping of unique chars to integers, and a reverse mapping
# chars = sorted(list(set(raw_text)))
# char_to_int = dict((c, i) for i, c in enumerate(chars))
# int_to_char = dict((i, c) for i, c in enumerate(chars))
# # summarize the loaded data
# n_chars = len(raw_text)
# n_vocab = len(chars)
# print ("Total Characters: ", n_chars)
# print ("Total Vocab: ", n_vocab)
# # prepare the dataset of input to output pairs encoded as integers
# seq_length = 100
# dataX = []
# dataY = []
# for i in range(0, n_chars - seq_length, 1):
# 	  seq_in = raw_text[i:i + seq_length]
# 	  seq_out = raw_text[i + seq_length]
#	  dataX.append([char_to_int[char] for char in seq_in])
#	  dataY.append(char_to_int[seq_out])
# n_patterns = len(dataX)
# print ("Total Patterns: ", n_patterns)
# # reshape X to be [samples, time steps, features]
# X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# # normalize
# X = X / float(n_vocab)
# # one hot encode the output variable
# y = np_utils.to_categorical(dataY)
# # define the LSTM model
# model = Sequential()
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
# # load the network weights
# # filename = "weights-improvement-19-1.9435.hdf5"
# filename = "weights-best.hdf5" 
# model.load_weights(filename)
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# # pick a random seed
# start = numpy.random.randint(0, len(dataX)-1)
# pattern = dataX[start]
# print ("Seed:")
# print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# # generate characters
# for i in range(1000):
#	  x = numpy.reshape(pattern, (1, len(pattern), 1))
#	  x = x / float(n_vocab)
#	  prediction = model.predict(x, verbose=0)
#	  index = numpy.argmax(prediction)
#	  result = int_to_char[index]
#	  seq_in = [int_to_char[value] for value in pattern]
#	  sys.stdout.write(result)
#	  pattern.append(index)
#	  pattern = pattern[1:len(pattern)]
# print ("\nDone.")
#
# Running this example first outputs the selected random seed, then each
# character as it is generated.
#
# For example, below are the results from one run of this text generator.
# The random seed was:
#
# be no mistake about it: it was neither more nor less than a pig, and she
# felt that it would be quit
#
# The generated text with the random seed (cleaned up for presentation) was:
#
# be no mistake about it: it was neither more nor less than a pig, and she
# felt that it would be quit e aelin that she was a little want oe toiet
# ano a grtpersent to the tas a little war th tee the tase oa teettee
# the had been tinhgtt a little toiee at the cadl in a long tuiee aedun
# thet sheer was a little tare gereen to be a gentle of the tabdit  soenee
# the gad  ouw ie the tay a tirt of toiet at the was a little 
# anonersen, and thiu had been woite io a lott of tueh a tiie  and taede
# bot her aeain  she cere thth the bene tith the tere bane to tee
# toaete to tee the harter was a little tire the same oare cade an anl ano
# the garee and the was so seat the was a little gareen and the sabdit,
# and the white rabbit wese tilel an the caoe and the sabbit se teeteer,
# and the white rabbit wese tilel an the cade in a lonk tfne the sabdi
# ano aroing to tea the was sf teet whitg the was a little tane oo thete
# the sabeit  she was a little tartig to the tar tf tee the tame of the
# cagd, and the white rabbit was a little toiee to be anle tite thete ofs
# and the tabdit was the wiite rabbit, and
#
# We can note some observations about the generated text.
#
# - It generally conforms to the line format observed in the original text of
#   less than 80 characters before a new line.
# - The characters are separated into word-like groups and most groups are
#   actual English words (e.g. “the”, “little” and “was”), but many do not
#   (e.g. “lott”, “tiie” and “taede”).
# - Some of the words in sequence make sense(e.g. “and the white rabbit“),
#   but many do not (e.g. “wese tilel“).
#
# The fact that this character based model of the book produces output like this
# is very impressive. It gives you a sense of the learning capabilities of LSTM
# networks.
#
# The results are not perfect. In the next section we look at improving the
# quality of results by developing a much larger LSTM network.
#
#
#
# ### Larger LSTM Recurrent Neural Network ###
# We got results, but not excellent results in the previous section. Now, we can try to improve the quality of the generated text by creating a much larger network.
#
# We will keep the number of memory units the same at 256, but add a second layer.
#
# model = Sequential()
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(256))
# model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')
#
# We will also change the filename of the checkpointed weights so that we can
# tell the difference between weights for this network and the previous (by
# appending the word “bigger” in the filename).
# 
# filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
#
# Finally, we will increase the number of training epochs from 20 to 50 and
# decrease the batch size from 128 to 64 to give the network more of an
# opportunity to be updated and learn.
#
#

# The full code listing is presented below for completeness.
'''
# Below code is only to create a weights-best.hdf5 weights file
print("if below code is finish comment out")
print("create a weights-best.hdf5 weights file")
# *** Larger LSTM Network to Generate Text for Alice in Wonderland ***
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
#print("Total Vocab: ", n_vocab)
print("Total Vocab (= distinct characters): ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# *** reshape X to be [samples, time steps, features] ***
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# *** normalize ***
X = X / float(n_vocab)
# *** one hot encode the output variable ***
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add( LSTM(256
          ,input_shape=(X.shape[1], X.shape[2])
          ,return_sequences=True)
         )
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense( y.shape[1]
                ,activation='softmax'
               )
         )
model.compile( loss='categorical_crossentropy'
              ,optimizer='adam')
# *** define the checkpoint ***
#filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
filepath = "weights-best.hdf5"
checkpoint = ModelCheckpoint( filepath
                             ,monitor='loss'
                             ,verbose=1
                             ,save_best_only=True
                             ,mode='min'
                             )
callbacks_list = [checkpoint]
# fit the model
model.fit( X
          ,y
          ,epochs=50
          ,batch_size=64
          ,callbacks=callbacks_list
         )
print("Comment out, above code")
print("Save weights file weights-best.hdf5")
print("Code below with weights-best.hdf5")	
'''	   
#		 
# Running this example takes some time, at least 700 seconds per epoch.
#
# After running this example you may achieved a loss of about 1.2. For example
# the best result I achieved from running this model was stored in a checkpoint
# file with the name:
#
# weights-improvement-47-1.2219-bigger.hdf5
#
# Achieving a loss of 1.2219 at epoch 47.
#
# As in the previous section, we can use this best model from the run to
# generate text.
#
# The only change we need to make to the text generation script from the
# previous section is in the specification of the network topology and from
# which file to seed the network weights.
#
#
#
# The full code listing is provided below for completeness.
#
# ### Load Larger LSTM network and generate text ###
# Attention: weights-best.hdf5 file is created with code block above
# and NOT below !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
#print "Total Vocab: ", n_vocab
print("Total Vocab (= distinct characters): ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM( 256
               ,input_shape=(X.shape[1], X.shape[2])
			   ,return_sequences=True
			  )
		 )
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense( y.shape[1]
                ,activation='softmax'
			   )
		 )
# load the network weights
#filename = "weights-improvement-47-1.2219-bigger.hdf5"
filename = "weights-best.hdf5"
model.load_weights(filename)
model.compile(  loss='categorical_crossentropy'
               ,optimizer='adam'
			 )
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\nDone.")

# One example of running this text generation script produces the output below.
#
# The randomly chosen seed text was:
#
# d herself lying on the bank, with her
# head in the lap of her sister, who was gently brushing away s
#
# The generated text with the seed (cleaned up for presentation) was :
#
# herself lying on the bank, with her
# head in the lap of her sister, who was gently brushing away
# so siee, and she sabbit said to herself and the sabbit said to herself and the sood
# way of the was a little that she was a little lad good to the garden,
# and the sood of the mock turtle said to herself, 'it was a little that
# the mock turtle said to see it said to sea it said to sea it say it
# the marge hard sat hn a little that she was so sereated to herself, and
# she sabbit said to herself, 'it was a little little shated of the sooe
# of the coomouse it was a little lad good to the little gooder head. and
# said to herself, 'it was a little little shated of the mouse of the
# good of the courte, and it was a little little shated in a little that
# the was a little little shated of the thmee said to see it was a little
# book of the was a little that she was so sereated to hare a little the
# began sitee of the was of the was a little that she was so seally and
# the sabbit was a little lad good to the little gooder head of the gad
# seared to see it was a little lad good to the little good
#
#
# We can see that generally there are fewer spelling mistakes and the text
# looks more realistic, but is still quite nonsensical.
#
# For example the same phrases get repeated again and again like
# “said to herself” and “little“. Quotes are opened but not closed.
#
# These are better results but there is still a lot of room for improvement.
#
# ### 10 Extension Ideas to Improve the Model ###
# Below are 10 ideas that may further improve the model that you could
# experiment with are:
#
#  1. Predict fewer than 1,000 characters as output for a given seed.
#  2. Remove all punctuation from the source text, and therefore from the
#     models’ vocabulary.
#  3. Try a one hot encoded for the input sequences.
#  4. Train the model on padded sentences rather than random sequences of
#     characters.
#  5. Increase the number of training epochs to 100 or many hundreds.
#  6. Add dropout to the visible input layer and consider tuning the dropout
#     percentage.
#  7. Tune the batch size, try a batch size of 1 as a (very slow) baseline and
#     larger sizes from there.
#  8. Add more memory units to the layers and/or more layers.
#  9. Experiment with scale factors (temperature) when interpreting the
#     prediction probabilities.
# 10. Change the LSTM layers to be “stateful” to maintain state across batches.
#
# Did you try any of these extensions? Share your results in the comments.
#
# ### Resources ###
#
# This character text model is a popular way for generating text using
# recurrent neural networks.
#
# Below are some more resources and tutorials on the topic if you are
# interested in going deeper. Perhaps the most popular is the tutorial
# by Andrej Karpathy titled “The Unreasonable Effectiveness of Recurrent
# Neural Networks“.
# see http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# 
# - Generating Text with Recurrent Neural Networks [pdf], 2011
#   http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf
# - Keras code example of LSTM for text generation.
#   https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
# - Lasagne code example of LSTM for text generation.
#   https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
# - MXNet tutorial for using an LSTM for text generation.
#   http://mxnetjl.readthedocs.io/en/latest/tutorial/char-lstm.html
# - Auto-Generating Clickbait With Recurrent Neural Networks.
#   https://larseidnes.com/2015/10/13/auto-generating-clickbait-with-recurrent-neural-networks/
#	
# ### Summary ###
#
# In this post you discovered how you can develop an LSTM recurrent neural network for text generation in Python with the Keras deep learning library.
#
# After reading this post you know:
#
# 1. Where to download the ASCII text for classical books for free that you can
#    use for training.
# 2. How to train an LSTM network on text sequences and how to use the trained
#    network to generate new sequences.
# 3. How to develop stacked LSTM networks and lift the performance of the model.
#
# Do you have any questions about text generation with LSTM networks or about
# this post? Ask your questions in the comments below and I will do my best to
# answer them.
