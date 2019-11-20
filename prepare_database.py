# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:50:20 2016

@author: valterf

Generate training and validation sets of a list of wave files with
speaker ids. Each line is of type "path speaker", as follows:

path/to/1st.wav spk1
path/to/2nd.wav spk2
path/to/3rd.wav spk1

and so on.

Output files will be named "train", "valid" and "test",
and will be generated in the current directory.
"""
import os
from sys import argv
from random import random, seed

from config import DEEPC_BASE


def main():
    """Perform processing when invoked from command line"""
    seed(1)
    spk = {}
    f = open(argv[1])
    for l in f:
        l = l.strip().split()
        if len(l) != 2:
            continue
        w, s = l
        if s not in spk:
            spk[s] = set()
        spk[s].add(w)
    f.close()

    path_trn = os.path.join(DEEPC_BASE, 'train')
    path_val = os.path.join(DEEPC_BASE, 'valid')
    path_tst = os.path.join(DEEPC_BASE, 'test')
    train = open(path_trn, 'w')
    valid = open(path_val, 'w')
    test = open(path_tst, 'w')
    for s in spk:
        r = random()
        if r > .5:
            for w in spk[s]:
                train.write(w + " " + s + "\n")
        elif r > .25:
            for w in spk[s]:
                valid.write(w + " " + s + "\n")
        else:
            for w in spk[s]:
                test.write(w + " " + s + "\n")
    for f in [train, valid, test]:
        f.close()


if __name__ == "__main__":
    main()
