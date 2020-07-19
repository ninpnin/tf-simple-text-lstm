import tensorflow_datasets as tfds
import re
import numpy as np

DFOLDER = "data/"
def load_data(fpath):
    s = open(fpath, "r").read().lower()
    s = re.sub(r'([^\s\w]|_)+', '', s)
    return s

def tokenize(data, encoder):
    return encoder.encode(data)

def tokenize_decode(data, encoder):
    encoded = encoder.encode(data)
    decoded = [encoder.decode([ix]) for ix in encoded]
    return encoded, decoded


def main():
    encoder = tfds.features.text.SubwordTextEncoder.load_from_file("encoder.tf")

    testdata = "pajari pajari pajari pajari jos ei pää huumeidenkäyttöä kestä niin kestääkö se sitten yhtään mitään"
    data = load_data(DFOLDER + "fi-novels1.txt")

    print("Encode data...")
    encoded, decoded = tokenize_decode(data, encoder)
    print("Done.")

    for wd in decoded[1495:1500]:
        print(wd)

    encoded = np.array(encoded)

    print("Save encoded data...")
    np.save(DFOLDER + "d.npy", encoded)
    print("Done.")

if __name__ == '__main__':
    main()