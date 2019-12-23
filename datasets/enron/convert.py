#!/usr/bin/python

import sys
import struct
import numpy as np
import h5py

matrix = []
with h5py.File('enron.hdf5', 'r') as inf:
    with open('enron.dat', 'wb') as ouf:
        dataset = inf['dataset']
        counter = 0
        for row in dataset:
            ouf.write(struct.pack('i', len(row)))
            ouf.write(struct.pack('%sf' % len(row), *row))
            counter += 1
            matrix.append(np.array(row, dtype=np.float32))
            if counter % 10000 == 0:
                sys.stdout.write('%d points processed...\n' % counter)
np.save('enron.dat', np.array(matrix))