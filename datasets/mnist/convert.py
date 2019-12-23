#!/usr/bin/python3

import sys
import struct
import numpy as np
import h5py

matrix = []
with h5py.File('MNIST.hdf5', 'r') as inf:
    with open('mnist.dat', 'wb') as ouf:
        dataset = inf['dataset']
        
        counter = 0
        for row in dataset:
            ouf.write(struct.pack('i', len(row)))
            ouf.write(struct.pack('%sf' % len(row), *row))
            counter += 1
            matrix.append(np.array(row, dtype=np.float32))
            if counter % 10000 == 0:
                sys.stdout.write('%d points processed...\n' % counter)
        queries = inf['query']
        for row in queries:
            ouf.write(struct.pack('i', len(row)))
            ouf.write(struct.pack('%sf' % len(row), *row))
            counter += 1
            matrix.append(np.array(row, dtype=np.float32))
            if counter % 10000 == 0:
                sys.stdout.write('%d points processed...\n' % counter)                
np.save('mnist.dat', np.array(matrix))