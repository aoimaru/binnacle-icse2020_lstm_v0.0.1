from abc import *

from libs.doc2vecs import *
from libs.files import *

# from keras.models import Sequential  
# from keras.layers.core import Dense, Activation  

# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.optimizers import Adam

# from keras.utils import np_utils

# from keras.models import load_model

import os
import numpy as np
import pprint

D2V_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/data/model/doc2vec"
LSTM_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/data/model/lstm"

INDEX_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/index"
INDEX_02_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/index02"

TR_SIZE = 5
VECTOR_SIZE = 10

class DLSTM(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def _create_training_indexes():
        pass
    
    @staticmethod
    @abstractmethod
    def _create_training_datas():
        pass

    @staticmethod
    @abstractmethod
    def _create_model():
        pass
    
    @staticmethod
    @abstractmethod
    def _predict_next_command():
        pass


class DLSTM_V1(DLSTM):
    @staticmethod
    def _create_training_indexes(target: str):
        d2v_model_path = "{d2v_model_root_path}/{target}.model".format(
            d2v_model_root_path=D2V_MODEL_ROOT_PATH,
            target=target
        )
        d2v_model = D2V._load_model(d2v_model_path)

        index_02_path = "{}/{}".format(INDEX_02_PATH, target)
        index_01_path = "{}/{}".format(INDEX_PATH, target)

        windows = [num for num in range(1, TR_SIZE)]; windows.sort(reverse=True)

        training_indexes = list()

        for _02_path in JsonFile._get_file_path(index_02_path):
            base_name = os.path.basename(_02_path); base_name = base_name.replace(".json", "")
            contents = JsonFile._get_contents(_02_path)
            for cnt_key, cnt_values in contents.items():
                if len(cnt_values) <= 1:
                    continue
                cnt_values = list(map(lambda x: int(x), cnt_values)); cnt_values.sort()

                for cnt_value in cnt_values:
                    train_y = "{}:{}:{}".format(base_name, cnt_key, cnt_value)
                    training_index = {
                        "train_x": list(),
                        "train_y": train_y
                    }
                    NaN = 0
                    for window in windows:
                        if not (cnt_value-window) in cnt_values:
                            training_index["train_x"].append("")
                            NaN += 1
                        else:
                            train_x = "{}:{}:{}".format(base_name, cnt_key, cnt_value-window)
                            
                            training_index["train_x"].append(train_x)
                    if NaN >= 3:
                        continue
                    training_indexes.append(training_index)
        
        return training_indexes
    
    @staticmethod
    def _create_training_datas(training_indexes: list, target: str, vector_size: int):
        d2v_model_path = "{d2v_model_root_path}/{target}.model".format(
            d2v_model_root_path=D2V_MODEL_ROOT_PATH,
            target=target
        )
        d2v_model = D2V._load_model(d2v_model_path)

        training_x_datas = dict()
        training_y_datas = dict()


        for training_index in training_indexes:
            x_vectors = list()
            for x_train in training_index["train_x"]:
                try:
                    x_vector = d2v_model[x_train]
                except Exception as e:
                    x_vector = [0]*vector_size
                else:
                    # x_vector = x_vector+1; x_vector = x_vector*1000; x_vector = x_vector.astype(int)
                    x_vector = x_vector.tolist()
                finally:
                    x_vectors.append(x_vector)
            x_vectors = np.array(x_vectors).T.tolist()

            y_vectors = d2v_model[training_index["train_y"]]
            y_vectors = y_vectors+1; y_vectors = y_vectors*1000; y_vectors = y_vectors.astype(int)
            y_vectors = y_vectors.tolist()

            for dim_id, (x_vec, y_vec) in enumerate(zip(x_vectors, y_vectors)):
                if not dim_id in training_y_datas:
                    training_y_datas[dim_id] = list()
                training_y_datas[dim_id].append(y_vec)
                if not dim_id in training_x_datas:
                    training_x_datas[dim_id] = list()
                training_x_datas[dim_id].append(x_vec)
                

        return training_x_datas, training_y_datas

    @staticmethod
    def _create_model(x_trains, y_trains, target, dim_id):
        from keras.models import Sequential  
        from keras.layers.core import Dense, Activation  

        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.optimizers import Adam

        from keras.utils import np_utils

        from keras.models import load_model

        
        x_trains = np.reshape(x_trains, (len(x_trains), 4, 1))
        # x_trains = x_trains/float(len(x_trains))
        # y_trains = np.array(y_trains)
        y_trains = np_utils.to_categorical(y_trains, len(y_trains)+1)
        layer = len(y_trains)+1

        # pprint.pprint(x_trains)
        # pprint.pprint(y_trains)

        models = Sequential()
        models.add(LSTM(128, input_shape=(4, 1)))
        models.add(Dense(layer, activation="softmax"))
        optimizer = Adam(learning_rate=0.01)
        models.summary()
        models.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        models.fit(x_trains, y_trains, epochs=200, verbose=1)
        try:
            os.mkdir("{}/{}".format(LSTM_MODEL_ROOT_PATH, target))
        except Exception as e:
            print(e)
        model_path = "{lstm_root_path}/{target}/{dim_id}.h5".format(
            lstm_root_path=LSTM_MODEL_ROOT_PATH,
            target=target,
            dim_id=dim_id
        )
        
        models.save(model_path)
    
    @staticmethod
    def _predict_next_command(queries: list, target: str):
        from keras.models import load_model
        from gensim.similarities.nmslib import NmslibIndexer

        d2v_model_path = "{d2v_model_root_path}/{target}.model".format(
            d2v_model_root_path=D2V_MODEL_ROOT_PATH,
            target=target
        )
        d2v_model = D2V._load_model(d2v_model_path)
        vectors = list()
        for query in queries:
            try:
                vector = d2v_model[query]
            except Exception as e:
                pass
            else:
                vectors.append(vector)
        
        while True:
            if len(vectors) >= 4:
                break
            vectors.insert(0, [float(0)]*VECTOR_SIZE)
        
        vectors = np.array(vectors).T.tolist()

        result_vecs = list()

        for vec_id, vector in enumerate(vectors):
            lstm_model_path = "{lstm_model_root_path}/{target}/{vec_id}.h5".format(
                lstm_model_root_path=LSTM_MODEL_ROOT_PATH,
                target=target,
                vec_id=vec_id
            )
            vector = np.reshape(vector, (1, 4, 1))
            lstm_model = load_model(lstm_model_path)
            prediction = lstm_model.predict(vector, verbose=1)
            result_vec = np.argmax(prediction)
            result_vecs.append(result_vec)

        result_vecs = np.array(result_vecs)
        result_vecs = result_vecs/1000
        result_vecs = result_vecs-1

        indexer = NmslibIndexer(d2v_model)
        # for result in indexer.most_similar(vector=result_vecs, num_neighbors=10):
        #     print(result)
        #     contents_id, similarity = result[0], result[1]
        #     contents_tag = contents_id.split(":")[0]
        #     contents_path = "{index_path}/{target}/{contents_tag}.json".format(
        #         index_path=INDEX_PATH,
        #         target=target,
        #         contents_tag=contents_tag
        #     )
        #     contents = JsonFile._get_contents(contents_path)
        #     pprint.pprint(contents[contents_id])
        return indexer.most_similar(vector=result_vecs, num_neighbors=10)













