# ML Music Generator
This project .
The initial source was made by Sigurður Skúli by is respository Classical-Piano-Composer
(https://github.com/Skuldur/Classical-Piano-Composer)




## Requirements

* Python 3.x
* Installing the following packages using pip:
	* Music21
	* Keras
	* Tensorflow
	* h5py

## Training

To train the network you run **10_train.py**.

E.g.

```
python 10_train.py
```

The network will use every midi file in ./midi_songs to train the network. The midi files should only contain a single instrument to get the most out of the training.

**NOTE**: You can stop the process at any point in time and the weights from the latest completed epoch will be available for text generation purposes.

## Generating music

Once you have trained the network you can generate text using **20_generate.py**

E.g.

```
python 20_generate.py
```

You can run the prediction file right away using the **weights.hdf5** file


Documentation
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
		

Toegevoegd door Johntestgit