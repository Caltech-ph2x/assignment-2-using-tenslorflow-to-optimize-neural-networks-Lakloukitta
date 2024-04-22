import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt

(ds_train, ds_test), ds_info = tfds.load(
    'fashion_mnist', 
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(buffer_size=ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(32)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dense(32)  
])


model.compile(
    optimizer=tf.keras.optimizers.Nadam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
history = model.fit(
    ds_train,
    epochs=64,
    validation_data=ds_test,
)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.yscale('log')
plt.grid(True)
plt.xlabel('Epoch #')
plt.show()