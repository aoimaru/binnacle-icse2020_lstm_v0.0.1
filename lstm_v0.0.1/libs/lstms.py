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

D2V_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/data/model/doc2vec"
LSTM_MODEL_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/data/model/lstm"

INDEX_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/index"
INDEX_02_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/index02"

TR_SIZE = 4

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
                    for window in windows:
                        if not (cnt_value-window) in cnt_values:
                            training_index["train_x"].append("")
                        else:
                            train_x = "{}:{}:{}".format(base_name, cnt_key, cnt_value-window)
                            
                            training_index["train_x"].append(train_x)
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
                    x_vector = x_vector.tolist()
                finally:
                    x_vectors.append(x_vector)
            x_vectors = np.array(x_vectors).T.tolist()

            y_vectors = d2v_model[training_index["train_y"]]
            y_vectors = y_vectors+1; y_vectors = y_vectors*1000; y_vectors = y_vectors.astype(int)
            y_vectors = y_vectors.tolist()

            for dim_id, (x_vector, y_vector) in enumerate(zip(x_vectors, y_vectors)):
                if not dim_id in training_y_datas:
                    training_y_datas[dim_id] = list()
                training_y_datas[dim_id].append(y_vector)
                if not dim_id in training_x_datas:
                    training_x_datas[dim_id] = list()
                training_x_datas[dim_id].append(x_vector)
                

        return training_x_datas, training_y_datas

    @staticmethod
    def _create_model(x_trains, y_trains, target, dim_id):
        from keras.models import Sequential  
        from keras.layers.core import Dense, Activation  

        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.optimizers import Adam

        from keras.utils import np_utils

        from keras.models import load_model
        layer = max(y_trains)+1
        x_trains = np.reshape(x_trains, (len(x_trains), 3, 1))
        y_trains = np.array(y_trains)

        models = Sequential()
        models.add(LSTM(128, input_shape=(4, 1)))
        models.add(Dense(1))
        models.add(Activation("linear"))
        optimizer = Adam(learning_rate=0.01)
        models.summary()
        models.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        models.fit(x_trains, y_trains, epochs=100, verbose=1)
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








