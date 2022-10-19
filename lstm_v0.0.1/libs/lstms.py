from abc import *

class DLSTM(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def _create_training_data():
        pass


class DLSTM_V1(DLSTM):
    @staticmethod
    def _create_training_data():
        