import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python3 count_spikes.py log_file_name.txt")
    quit()

with open(sys.argv[1], "r") as file:
    log = file.read()

log = log.split("\n")
a = []
for i, line in enumerate(log):
    if line.startswith("["):
        a.append(eval(line)[0])
print(np.sum(a) / 10000)

