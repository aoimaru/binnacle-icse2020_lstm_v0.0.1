from libs.doc2vecs import *
from libs.files import *
from libs.lstms import *

import sys
import pprint
import glob
import numpy as np
import os

OPTION_01 = 1

D2V_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/data/model/doc2vec"

SAMPLE_INDEX_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/index/trace-gold"

INDEX_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/index"
INDEX_02_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/index02"

TR_SIZE = 4
VECTOR_SIZE = 10

# from keras.models import Sequential  
# from keras.layers.core import Dense, Activation  

# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.optimizers import Adam

# from keras.utils import np_utils

# from keras.models import load_model


D2V_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/data/model/doc2vec"

def main(args):

    d2v_model = D2V._load_model("{}/{}".format(D2V_MODEL_ROOT_PATH, "trace-gold.model"))

    training_indexes = DLSTM_V1._create_training_indexes(target=args[OPTION_01])
    for training_index in training_indexes:
        print(training_index)
        


if __name__ == "__main__":
    main(sys.argv)