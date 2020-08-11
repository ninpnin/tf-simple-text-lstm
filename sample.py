import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model import init_model

SEED_LEN = 15
SATURATION = 0.5

def load():
    encoder = tfds.features.text.SubwordTextEncoder.load_from_file("encoder.tf")
    model = init_model(encoder.vocab_size)
    model.load_weights("weights/weights")

    print(encoder)
    print(model)
    return encoder, model

def initial_seed(encoder):
    seed = input("Seed text:\n")
    seed = encoder.encode(seed)

    if len(seed) < SEED_LEN:
        diff = SEED_LEN - len(seed)
        seed_zeros = np.zeros(SEED_LEN, dtype=np.int32)
        for ix, wd in enumerate(seed):
            seed_zeros[diff + ix] = wd

        seed = list(seed_zeros)

    return seed[:SEED_LEN]

def draw_sample(model, seed):
    prediction = model.predict(seed)
    #print(prediction.shape)

    new_probs = prediction[-1]
    new_probs = new_probs * SATURATION
    new_probs = np.exp(new_probs)
    new_probs = new_probs / np.sum(new_probs)

    new_entry = np.random.choice(len(new_probs), p=new_probs)

    return new_entry

def main():
    encoder, model = load()
    model.save_weights("weights/weights")
    seed = initial_seed(encoder)
    seed = np.array(seed)
    for i in range(25):
        #print(seed)
        new_entry = draw_sample(model, seed[-SEED_LEN:])

        #wd = encoder.decode([new_entry])
        #print(wd)

        seed = np.append(seed, new_entry)

    print(encoder.decode(seed[SEED_LEN:]))

if __name__ == '__main__':
    main()