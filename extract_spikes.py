import re
from absl import flags
from absl import app
import numpy as np

FLAGS = flags.FLAGS

def main(argv):
    with open(FLAGS.file_name, 'r') as f:
        text = f.read()

    # extract spike_sums:
    spikes = []
    for line in text.split('\n'):
        if line.startswith('[') and line.endswith(']'):
            spikes.append(int(float(line[1:-1])))  # to convert scientific notation as well 

    average_n_spikes = np.sum(spikes) / FLAGS.n_neurons / FLAGS.n_images
    print(f'Average number of spikes: {average_n_spikes}')

    pass


if __name__ == '__main__':
    flags.DEFINE_string('file_name', '', 'Name of the output file')
    flags.DEFINE_integer('n_neurons', 1, 'Number of neurons in the model')
    flags.DEFINE_integer('n_images', 1, 'Number of images')
    app.run(main)
