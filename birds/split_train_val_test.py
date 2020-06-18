import numpy as np
import pickle

import config

ALL_FILENAMES = '../data/CUB_200_2011/CUB_200_2011/images.txt'
REDO_SPLIT_FILE = 'redo_train_val_test_split.txt'
DMGAN_TEST_FILENAMES = 'dmgan_test_filenames.pickle'

NUM_VALID = config.VALID_SIZE
np.random.seed(6789)

file_id = {}
with open(ALL_FILENAMES, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        id, file = line.split()
        file_id[file] = id


redo_split = {}
with open(REDO_SPLIT_FILE, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        id, sp = line.split()
        redo_split[id] = int(sp)


with open(DMGAN_TEST_FILENAMES, 'rb') as f:
    dmgan_test_filenames = [file+'.jpg' for file in pickle.load(f)]


our_test_ids = []
for file in dmgan_test_filenames:
    if redo_split[file_id[file]] != 0:
        our_test_ids.append(file_id[file])

our_trainval_ids = []
for id in file_id.values():
    if id not in our_test_ids:
        our_trainval_ids.append(id)

np.random.shuffle(our_trainval_ids)
our_val_ids = our_trainval_ids[:NUM_VALID]


with open('our_train_test_split.txt', 'w') as f:
    for id in file_id.values():
        if id in our_test_ids:
            f.write("%s %d\n" % (id, 2))
        elif id in our_val_ids:
            f.write("%s %d\n" % (id, 1))
        else:
            f.write("%s %d\n" % (id, 0))
