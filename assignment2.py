import numpy as np
import matplotlib.pyplot as plt
import gzip
import pandas as pd

from timeit import default_timer as timer

import tensorflow as tf
import tensorflow_datasets as tfds
(ds_train, ds_test), ds_info = tfds.load(
    'mnist', 
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)
def normalize_img(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) 
    return image, label
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()  
ds_train = ds_train.shuffle(buffer_size=ds_info.splits['train'].num_examples)  
ds_train = ds_train.batch(32)  
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)  
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.cache()  
ds_test = ds_test.batch(32)  
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(10)  
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
history = model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.yscale('log')
plt.grid(True)
plt.xlabel('Epoch #')
plt.show()