#!/usr/bin/python

import sys
import struct
import numpy as np
import sys
if sys.version_info[0] != 2:
    raise Exception("Must be using Python 2")

matrix = []
with open('glove.twitter.27B.100d.txt', 'r') as inf:
    with open('glove.twitter.27B.100d.dat', 'wb') as ouf:
        counter = 0
        for line in inf:
            # tmp = line.split()[0]
            # try:
            #     x = float(tmp)
            #     print(f"{tmp} {counter + 1}")
            #     exit(0)
            # except:
            #     pass
            row = [float(x) for x in line.split()[1:]]
            # if len(row) != 100:
            #     print(f"{line} {len(row)}")
            assert len(row) == 100
            ouf.write(struct.pack('i', len(row)))
            ouf.write(struct.pack('%sf' % len(row), *row))
            counter += 1
            matrix.append(np.array(row, dtype=np.float32))
            if counter % 10000 == 0:
                sys.stdout.write('%d points processed...\n' % counter)
np.save('glove.twitter.27B.100d.dat', np.array(matrix))
