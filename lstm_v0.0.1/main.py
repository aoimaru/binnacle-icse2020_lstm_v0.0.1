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

def main(args):
    training_indexes = DLSTM_V1._create_training_indexes(target=args[OPTION_01])
    for training_index in training_indexes:
        print(training_index)
    # training_x_datas, training_y_datas = DLSTM_V1._create_training_datas(training_indexes=training_indexes, target=args[OPTION_01], vector_size=VECTOR_SIZE)
    # for vec_size in range(VECTOR_SIZE):
    #     DLSTM_V1._create_model(
    #         x_trains=training_x_datas[vec_size], 
    #         y_trains=training_y_datas[vec_size],
    #         target=args[OPTION_01],
    #         dim_id=vec_size
    #     )
    # DLSTM_V1._create_model(
    #     x_trains=training_x_datas[0],
    #     y_trains=training_y_datas[0],
    #     target=args[OPTION_01],
    #     dim_id=0
    # )
    # print(len(training_x_datas[0]), len(training_y_datas[0]))
    # for train_x_zero, train_y_zero in zip(training_x_datas[0], training_y_datas[0]):
    #     print(train_x_zero, train_y_zero)
    




if __name__ == "__main__":
    main(sys.argv)