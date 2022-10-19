from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/data/model"

class D2V(object):
    @staticmethod
    def _load_model(model_path):
        return Doc2Vec.load(model_path)