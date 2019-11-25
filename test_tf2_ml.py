# File: test_tf2_ml.py
# Function: working tf 2.0 example
#           regel1
#           regel2
# see https://towardsdatascience.com/building-your-first-neural-network-in-tensorflow-2-tensorflow-for-hackers-part-i-e1e2f1dfe7a0

import tensorflow as tf
from tensorflow import keras
import numpy as np

(x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()

def preprocess(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)

  return x, y

def create_dataset(xs, ys, n_classes=10):
  ys = tf.one_hot(ys, depth=n_classes)
  return tf.data.Dataset.from_tensor_slices((xs, ys)) \
    .map(preprocess) \
    .shuffle(len(ys)) \
    .batch(128)

train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_val, y_val)

train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_val, y_val)

model = keras.Sequential([
    keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dense(units=192, activation='relu'),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_dataset.repeat(), 
    epochs=10, 
    steps_per_epoch=500,
    validation_data=val_dataset.repeat(), 
    validation_steps=2
)

predictions = model.predict(val_dataset)


# Our model outputs a probability distribution about how likely
# each clothing type is shown on an image. To make a decision,
# we can get the one with the highest probability:
print(np.argmax(predictions[0]))

# different types of clothing:
#| Label | Description |
#|-------|-------------|
#| 0     | T-shirt/top |
#| 1     | Trouser     |
#| 2     | Pullover    |
#| 3     | Dress       |
#| 4     | Coat        |
#| 5     | Sandal      |
#| 6     | Shirt       |
#| 7     | Sneaker     |
#| 8     | Bag         |
#| 9     | Ankle boot  |