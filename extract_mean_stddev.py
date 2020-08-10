import re
from absl import flags
from absl import app
import numpy as np

FLAGS = flags.FLAGS

def main(argv):
    with open(FLAGS.file_name, 'r') as f:
        text = f.read()

    means = []
    stddevs = []
    search_str = '\[(-?\d+\.\d+)\]\[(\d+\.\d+)\]'
    matches = re.findall(search_str, text)
    for match in matches:
        mean, stddev = match
        mean, stddev = float(mean), float(stddev)
        means.append(mean)
        stddevs.append(stddev)

    means = np.mean(means)
    stddevs = np.mean(stddevs)

    print(f'Average mean of Input: {means}\nAverage stddev of Input: {stddevs}')



if __name__ == '__main__':
    flags.DEFINE_string('file_name', '', 'Name of the output file')
    app.run(main)
