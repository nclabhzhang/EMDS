import numpy as np
import torch

import logging
import os
import urllib

from os.path import join as join
import urllib.request

def is_int(str):
    try:
        int(str)
        return True
    except:
        return False


gdb9_url_excluded = 'https://springernature.figshare.com/ndownloader/files/3195404'
gdb9_txt_excluded = 'uncharacterized.txt'
urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)

with open(gdb9_txt_excluded) as f:
    lines = f.readlines()
    excluded_strings = [line.split()[0]
                        for line in lines if len(line.split()) > 0]

excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

assert len(excluded_idxs) == 3054, 'There should be exactly 3054 excluded atoms. Found {}'.format(len(excluded_idxs))

Ngdb9 = 133885
Nexcluded = 3054

included_idxs = np.array(sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

Nmols = Ngdb9 - Nexcluded

Ntrain = 100000
Ntest = int(0.1 * Nmols)
Nvalid = Nmols - (Ntrain + Ntest)

np.random.seed(0)
data_perm = np.random.permutation(Nmols)

train, valid, test, extra = np.split(data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

assert (len(extra) == 0), 'Split was inexact {} {} {} {}'.format(len(train), len(valid), len(test), len(extra))

train = included_idxs[train]
valid = included_idxs[valid]
test = included_idxs[test]

splits = {'train_idx': train, 'valid_idx': valid, 'test_idx': test}
np.savez('split.npz', train_idx=train, val_idx=valid, test_idx=test)
