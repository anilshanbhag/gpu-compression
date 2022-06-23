import numpy as np
import sys

l = 1<<20
max_val = l - 1

alpha = float(sys.argv[1])
w = np.random.zipf(alpha, l).astype(int)

for i in range(0, l):
    if w[i] > max_val:
        w[i] = max_val

np.savetxt('datad3_' + str(int(alpha)), w, fmt='%i')

