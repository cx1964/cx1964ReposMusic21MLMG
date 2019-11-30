# Filename: 10_learn_textGenLSTM.py
# Function: Text Generation With LSTM Recurrent Neural Networks
#           in Python with Keras
# Remark: This code comes from article
# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/   
# 
# This code has similarities with learning code of article
# https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
# see also 10_learn.py
#
# The code of the article "Text Generation With LSTM Recurrent Neural Networks
# in Python with Keras" processes a sequences of characters.
# The code of the article "How to Generate Music using a LSTM Neural Network in Keras"
# processes a sequences of list of notes where notes are mapped to numbers.

# Develop a Small LSTM Recurrent Neural Network
# In this section we will develop a simple LSTM network to learn
# sequences of characters from Alice in Wonderland. In the next section
# we will use this model to generate new sequences of characters.
#
# ### The problem is really a single character classification problem with 47 classes ###
#
# Let’s start off by importing the classes and functions we intend to use
# to train our model.

# *** import relevant libraries ***
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# Next, we need to load the ASCII text for the book into memory and
# convert all of the characters to lowercase to reduce the vocabulary
# that the network must learn.

# *** load ascii text and covert to lowercase ***
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# Now that the book is loaded, we must prepare the data for modeling by the neural network.
# We cannot model the characters directly, instead we must convert the characters to integers.
#
# We can do this easily by first creating a set of all of the distinct characters in the book,
# then creating a map of each character to a unique integer.

# *** create mapping of unique chars to integers ***
chars = sorted(list(set(raw_text))) # see also 10_train.py
char_to_int = dict((c, i) for i, c in enumerate(chars)) # see also 10_train.py

# For example, the list of unique sorted lowercase characters in the book is as follows:
# ['\n', '\r', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', ':', ';', '?', '[', ']', '_',
#  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
#  's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\xbb', '\xbf', '\xef']

# You can see that there may be some characters that we could remove to further clean up the
# dataset that will reduce the vocabulary and may improve the modeling process.
#
# Now that the book has been loaded and the mapping prepared, we can summarize the dataset.

# *** summarize the loaded data ***
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab (= distinct characters): ", n_vocab)

# Running the code to this point produces the following output.
# Total Characters:  147674
# Total Vocab (= distinct characters):  47

# We can see that the book has just under 150,000 characters and that when converted to lowercase
# that there are only 47 distinct characters in the vocabulary for the network to learn. Much more
# than the 26 in the alphabet.
#
# We now need to define the training data for the network. There is a lot of flexibility in how you
# choose to break up the text and expose it to the network during training.
#
# In this tutorial we will split the book text up into subsequences with a fixed length of
# 100 characters, an arbitrary length. We could just as easily split the data up by sentences and
# pad the shorter sequences and truncate the longer ones.
#
# Each training pattern of the network is comprised of 100 time steps of one character (X) followed
# by one character output (y). When creating these sequences, we slide this window along the whole
# book one character at a time, allowing each character a chance to be learned from the 100 characters that preceded it (except the first 100 characters of course).
#
# For example, if the sequence length is 5 (for simplicity) then the first two training patterns
# would be as follows:

# CHAPT -> E
# HAPTE -> R

# As we split up the book into these sequences, we convert the characters to integers using our
# lookup table we prepared earlier.

# *** prepare the dataset of input to output pairs encoded as integers ***
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

# Running the code to this point shows us that when we split up the dataset into training data
# for the network to learn that we have just under 150,000 training pattens. This makes sense as
# excluding the first 100 characters, we have one training pattern to predict each of the remaining
# characters.

# Total Patterns:  147574

# Now that we have prepared our training data we need to transform it so that it is suitable for
# use with Keras.
#
# First we must transform the list of input sequences into the form [samples, time steps, features]
# expected by an LSTM network.
#
# Next we need to rescale the integers to the range 0-to-1 to make the patterns easier to learn by
# the LSTM network that uses the sigmoid activation function by default.
#
# Finally, we need to convert the output patterns (single characters converted to integers) into
# a one hot encoding. This is so that we can configure the network to predict the probability of
# each of the 47 different characters in the vocabulary (an easier representation) rather than
# trying to force it to predict precisely the next character. Each y value is converted into a
# sparse vector with a length of 47, full of zeros except with a 1 in the column for the
# letter (integer) that the pattern represents.
#
# For example, when “n” (integer value 31) is one hot encoded it looks as follows:

# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.]

# We can implement these steps as below.

# *** reshape X to be [samples, time steps, features] ***
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# We can now define our LSTM model. Here we define a single hidden LSTM layer with
# 256 memory units. The network uses dropout with a probability of 20. The output layer
# is a Dense layer using the softmax activation function to output a probability prediction
# for each of the 47 characters between 0 and 1.
# 
# The problem is really a single character classification problem with 47 classes and as such
# is defined as optimizing the log loss (cross entropy), here using the ADAM optimization algorithm
# for speed.

# *** define the LSTM model ***
model = Sequential()
model.add(LSTM( 256
               ,input_shape=(X.shape[1], X.shape[2])
              )
         )
model.add(Dropout(0.2))
model.add(Dense( y.shape[1]
                ,activation='softmax')
         )
model.compile( loss='categorical_crossentropy'
              ,optimizer='adam'
             )
# show used model
model.summary()

# There is no test dataset. We are modeling the entire training dataset to learn the probability
# of each character in a sequence.
# 
# We are not interested in the most accurate (classification accuracy) model of the training
# dataset. This would be a model that predicts each character in the training dataset perfectly.
# Instead we are interested in a generalization of the dataset that minimizes the chosen loss
# function. We are seeking a balance between generalization and overfitting but short of
# memorization.
#
# The network is slow to train (about 300 seconds per epoch on an Nvidia K520 GPU). Because of the
# slowness and because of our optimization requirements, we will use model checkpointing to record
# all of the network weights to file each time an improvement in loss is observed at the end of the
# epoch. We will use the best set of weights (lowest loss) to instantiate our generative model in
# the next section.

# *** define the checkpoint ***
# filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
filepath="weights-best.hdf5"
checkpoint = ModelCheckpoint( filepath
                              ,monitor='loss'
                              ,verbose=1
                              ,save_best_only=True
                              ,mode='min'
                            )
callbacks_list = [checkpoint]

# We can now fit our model to the data. Here we use a modest number of 20 epochs and a large
# batch size of 128 patterns.

# *** fit the model ***
model.fit( X
          ,y
          ,epochs=20
          ,batch_size=128
          ,callbacks=callbacks_list
         )

# You will see different results because of the stochastic nature of the model, and because it is
# hard to fix the random seed for LSTM models to get 100% reproducible results. This is not a
# concern for this generative model.
# 
# After running the example, you should have a number of weight checkpoint files in the local
# directory.
#
# You can delete them all except the one with the smallest loss value. For example, when I ran this
# example, below was the checkpoint with the smallest loss that I achieved.

# weights-improvement-19-1.9435.hdf5

# The network loss decreased almost every epoch and I expect the network could benefit from
# training for many more epochs.
# 
# In the next section we will look at using this model to generate new text sequences.
