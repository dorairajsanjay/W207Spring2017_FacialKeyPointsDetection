## import libraries
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

# change the directory to where you put the datasets

FTRAIN = '~/Facial_KeyPoints_Detection/training.csv'
FTEST = '~/Facial_KeyPoints_Detection/test.csv'


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
        
class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb
    
def load2d(test=False, cols=None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

# increase las two hidden layer from 500 to 1000 and increase max epochs to 10000#
net10 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1_1', layers.Conv2DLayer),
        ('conv1_2', layers.Conv2DLayer),
        ('conv1_3', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  
        ('conv2_1', layers.Conv2DLayer),
        ('conv2_2', layers.Conv2DLayer),
        ('conv2_3', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  
        ('conv3_1', layers.Conv2DLayer),
        ('conv3_2', layers.Conv2DLayer),
        ('conv3_3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  
        ('hidden5', layers.DenseLayer),
        ('dropout5', layers.DropoutLayer),  
        ('hidden6', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_1_num_filters=32, conv1_1_filter_size=(3, 3), conv1_1_pad=1,
    conv1_2_num_filters=32, conv1_2_filter_size=(3, 3), conv1_2_pad=1,
    conv1_3_num_filters=32, conv1_3_filter_size=(3, 3), conv1_3_pad=1,
    pool1_pool_size=(2, 2),
    dropout1_p=0.1,  
    conv2_1_num_filters=64, conv2_1_filter_size=(3, 3), conv2_1_pad=1,
    conv2_2_num_filters=64, conv2_2_filter_size=(3, 3), conv2_2_pad=1,
    conv2_3_num_filters=64, conv2_3_filter_size=(3, 3), conv2_3_pad=1,
    pool2_pool_size=(2, 2),
    dropout2_p=0.2,  
    conv3_1_num_filters=128, conv3_1_filter_size=(3, 3), conv3_1_pad=1,
    conv3_2_num_filters=128, conv3_2_filter_size=(3, 3), conv3_2_pad=1,
    conv3_3_num_filters=128, conv3_3_filter_size=(3, 3), conv3_3_pad=1,
    pool3_pool_size=(2, 2),
    dropout3_p=0.3,  
    hidden4_num_units=1000,
    dropout4_p=0.5,  
    hidden5_num_units=1000,
    dropout5_p=0.5,  
    hidden6_num_units=1000,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=3000,
    verbose=1,
    )

sys.setrecursionlimit(10000)

X, y = load2d()
net10.fit(X, y)

with open('net10.pickle', 'wb') as f:
    pickle.dump(net10, f, -1)


net11 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1_1', layers.Conv2DLayer),
        ('conv1_2', layers.Conv2DLayer),
        ('conv1_3', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  
        ('conv2_1', layers.Conv2DLayer),
        ('conv2_2', layers.Conv2DLayer),
        ('conv2_3', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  
        ('conv3_1', layers.Conv2DLayer),
        ('conv3_2', layers.Conv2DLayer),
        ('conv3_3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  
        ('hidden5', layers.DenseLayer),
       
       
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_1_num_filters=32, conv1_1_filter_size=(3, 3), conv1_1_pad=1,
    conv1_2_num_filters=32, conv1_2_filter_size=(3, 3), conv1_2_pad=1,
    conv1_3_num_filters=32, conv1_3_filter_size=(3, 3), conv1_3_pad=1,
    pool1_pool_size=(2, 2),
    dropout1_p=0.1,  
    conv2_1_num_filters=64, conv2_1_filter_size=(3, 3), conv2_1_pad=1,
    conv2_2_num_filters=64, conv2_2_filter_size=(3, 3), conv2_2_pad=1,
    conv2_3_num_filters=64, conv2_3_filter_size=(3, 3), conv2_3_pad=1,
    pool2_pool_size=(2, 2),
    dropout2_p=0.2,  
    conv3_1_num_filters=128, conv3_1_filter_size=(3, 3), conv3_1_pad=1,
    conv3_2_num_filters=128, conv3_2_filter_size=(3, 3), conv3_2_pad=1,
    conv3_3_num_filters=128, conv3_3_filter_size=(3, 3), conv3_3_pad=1,
    pool3_pool_size=(2, 2),
    dropout3_p=0.3,  
    hidden4_num_units=1000,
    dropout4_p=0.5,  
    hidden5_num_units=1000,
   
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=3000,
    verbose=1,
    )

sys.setrecursionlimit(10000)

X, y = load2d()
net11.fit(X, y)

with open('net11.pickle', 'wb') as f:
    pickle.dump(net11, f, -1)
