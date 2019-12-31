#!/usr/bin/env python3

import numpy as np 
import h5py


with h5py.File("mf/sift1m-hamming-64.h5", "r") as mf:
    with h5py.File("sf/sift1m-hamming-64.h5", "r") as sf:
        for key in ['train', 'test']:
            if mf[key][:].shape != sf[key][:].shape or np.all(mf[key][:] - sf[key][:] == 0):
                print(f"{key} not equal")
