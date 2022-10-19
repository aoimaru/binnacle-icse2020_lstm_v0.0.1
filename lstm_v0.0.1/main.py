from libs.doc2vecs import *
from libs.files import *

import sys
import pprint
import glob

OPTION_01 = 1

D2V_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/data/model/doc2vec"

SAMPLE_INDEX_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/index/trace-gold"

def main(args):
    d2v_model_path = "{d2v_model_root_path}/{target}.model".format(
        d2v_model_root_path=D2V_MODEL_ROOT_PATH,
        target=args[OPTION_01]
    )
    d2v_model = D2V._load_model(d2v_model_path)
    pprint.pprint(d2v_model)

    for sample_index_path in glob.glob("{}/**/*".format(SAMPLE_INDEX_PATH), recursive=True)[:2]:
        sample_contents = JsonFile._get_contents(sample_index_path)
        for sample_key, sample_values in sample_contents.items():
            for result in d2v_model.docvecs.most_similar(sample_key):
                print(result)



if __name__ == "__main__":
    main(sys.argv)