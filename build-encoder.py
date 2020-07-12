#import tensorflow as tf
import tensorflow_datasets as tfds

def get_data():
    data_path = "data/fi-novels1.txt"
    data = open(data_path).read().lower()
    return data

def split_data(data):
    splitchar = '.'
    return list(data.split(splitchar))

corpus = get_data()
print(corpus)

# Build encoder
print("Create subword encoder based on data...")
V = 2**14
encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    split_data(corpus), target_vocab_size=V)

encoder_path = "encoder.tf"
encoder.save_to_file(encoder_path)
print(encoder.subwords)
print("Done.")