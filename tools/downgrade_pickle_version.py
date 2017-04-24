import os
import sys
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import _pickle as pickle
import matplotlib.pyplot as pyplot
import theano
from nolearn.lasagne import BatchIterator
from collections import OrderedDict
from sklearn.base import clone

# set recursion limit due to handle updates to pickle files for the large network models
sys.setrecursionlimit(100000)

pickleFile = "/mnt/dev/Facial_KeyPoints_Detection/Improvement_Tingwen/net10.pickle"
newPickleFile = "/mnt/dev/Facial_KeyPoints_Detection/Improvement_Tingwen/net10_prot2.pickle"
net = pickle.load(open(pickleFile,'rb'))

#pickle.dump(your_object, your_file, protocol=2)
with open(newPickleFile, 'wb') as f:
    pickle.dump(net, f,protocol=2)
