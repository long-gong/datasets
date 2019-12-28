#!/usr/bin/env python3

import numpy as np 
import h5py 
import os.path

def get_enc_dim(fn):
    name,_ = os.path.splitext(fn)
    return int(int(name.split('-')[-1]) / 64)

def verify(fn):
    with h5py.File(fn, 'r') as fp:
        data = np.concatenate((fp['train'][:], fp['test'][:]))
        mn = data.shape[0]
        if mn > 1e7:
            print("Dataset is too large to be verified with this script")
            return
        enc_dim = get_enc_dim(fn)
        matrix = np.reshape(data, (int(mn / enc_dim), enc_dim))
        u_matrix = np.unique(matrix, axis=0)
        if matrix.shape != u_matrix.shape:
            print(f"{fn} is no correct: There are depulications")

if __name__ == "__main__":
    CUR_PATH = os.path.dirname(__file__)
    for fn in os.listdir(CUR_PATH):
        if fn.endswith(".h5"):
            print("Verifying {fn} ...")
            verify(fn)
            print("Done")
