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

TR_SIZE = 5
VECTOR_SIZE = 10


def _create_model(args):
    training_indexes = DLSTM_V1._create_training_indexes(target=args[OPTION_01])
    training_x_datas, training_y_datas = DLSTM_V1._create_training_datas(training_indexes=training_indexes, target=args[OPTION_01], vector_size=VECTOR_SIZE)
    for vec_size in range(VECTOR_SIZE):
        DLSTM_V1._create_model(
            x_trains=training_x_datas[vec_size], 
            y_trains=training_y_datas[vec_size],
            target=args[OPTION_01],
            dim_id=vec_size
        )

def _test(args):
    queries = [
        "0b1975d451426f9858f59b812411970f4e2ac49c:13:7",
        "0b1975d451426f9858f59b812411970f4e2ac49c:13:8",
        "0b1975d451426f9858f59b812411970f4e2ac49c:13:9",
        "0b1975d451426f9858f59b812411970f4e2ac49c:13:10"
    ]
    excepted = "0b1975d451426f9858f59b812411970f4e2ac49c:13:11"
    """
    <---------- Result ---------->
    ('0b1975d451426f9858f59b812411970f4e2ac49c:13:11', 0.9965061247348785)
    ('7ac2d91f901a59a80cfde3dee6c166799d524942:17:8', 0.9924890398979187)
    ('e895200f2b9f9fc48e0d02be4c15ef81ff2b6b17:4:6', 0.9843948781490326)
    ('6e482708d3cafd1b0361e981702a95b023033688:6:1', 0.9837763607501984)
    ('3a1d8dd926a95fd10100d63ff41009620c2654da:13:1', 0.9829370379447937)
    ('746523d4420e299f747110865881e150f072085c:7:1', 0.9826214909553528)
    ('8bb0da230203a6be21b4097ef0f79e07406388ab:3:0', 0.9820409119129181)
    ('3870ccc23fd41eddfaf6f7b999e04eea654f52b0:14:0', 0.9813095927238464)
    ('d9b087bdfe0fb6f18d180555027ee465135bf1f2:9:6', 0.9809336960315704)
    ('8e4ac4930397fef9fad25ff8820dc6cd1752a503:16:11', 0.9779320657253265)
    """

    queries = [
        "1c11182d763188889c00d8f44a91d0df09e0147b:6:1",
        "1c11182d763188889c00d8f44a91d0df09e0147b:6:2",
        "1c11182d763188889c00d8f44a91d0df09e0147b:6:3",
        "1c11182d763188889c00d8f44a91d0df09e0147b:6:4"
    ]
    """
    <---------- Result ---------->
    "1c11182d763188889c00d8f44a91d0df09e0147b:6:5"
    """

    DLSTM_V1._predict_next_command(queries=queries, target=args[OPTION_01])


def _test_v2(args):
    Response = list()
    windows = [window for window in range(1, TR_SIZE)]
    windows.sort()
    file_paths = JsonFile._get_file_path(root_path="{}/{}".format(INDEX_02_PATH, args[OPTION_01]))
    for file_path in file_paths:
        file_id = os.path.basename(file_path); file_id = file_id.replace(".json", "")
        contents = JsonFile._get_contents(file_path)
        for run_id, command_ids in contents.items():
            # print(run_id, command_ids)
            if len(command_ids) < 2:
                continue
            command_ids = [int(command_id) for command_id in command_ids]
            command_ids.sort()
            for index_y in range(1, max(command_ids)):
                index_xs = list()
                for window in windows:
                    index_x = index_y-window
                    if index_x >= 0:
                        contents_id = "{}:{}:{}".format(file_id, run_id, index_x)
                        index_xs.append(contents_id)
                while True:
                    if len(index_xs) >= 4:
                        break
                    index_xs.insert(0, "")
                index_xs.reverse()
                index_y = "{}:{}:{}".format(file_id, run_id, index_y)
                # print(index_xs)
                # print(index_y)
                # index_01_contents = JsonFile._get_file_path(
                #     root_path="{}/{}/{}.json".format(INDEX_PATH, args[OPTION_01], index_y)
                # )
                results = DLSTM_V1._predict_next_command(queries=index_xs, target=args[OPTION_01])
                for result in results:
                    if result[0] == index_y:
                        Response.append(True)
                        break
                else:
                    Response.append(False)
    print("正解率:", len([Res for Res in Response if Res])/len(Response))

    """
    正解率: 0.6099893730074389
    python3 main.py trace-gold  19538.06s user 456.41s system 102% cpu 5:25:59.12 total
    """








def main(args):
    # _create_model(args)
    _test_v2(args)

    
    




if __name__ == "__main__":
    main(sys.argv)