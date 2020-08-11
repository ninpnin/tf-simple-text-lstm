import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model import init_model

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()


encoder = tfds.features.text.SubwordTextEncoder.load_from_file("encoder.tf")

def load_data(filepath):
    data = np.load(filepath)
    print(data[:100])
    return data


SEED_LEN = 15

def split_data(data):
    print("Split data into input slices and output indices...")
    inputs, outputs = [], []

    for ix, wd in enumerate(data):
        if ix >= SEED_LEN:
            current_input = data[ix-SEED_LEN:ix]
            #print(current_input)
            inputs.append(current_input)
            outputs.append(wd)

    print("Done.")
    print(inputs[:20], outputs[:20])
    return inputs, outputs

def create_dataset(data):
    inputs, outputs = split_data(data)
    print("Create a tf.data.Dataset...")
    train_dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    print("Done.")
    return train_dataset

model = init_model(encoder.vocab_size)
model.load_weights("weights/weights")
print(model.summary())

data = load_data("data/d.npy")
train_dataset = create_dataset(data)

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)

#test_dataset = test_dataset.padded_batch(BATCH_SIZE)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=1)

#model.save('model')
model.save_weights("weights/weights")

