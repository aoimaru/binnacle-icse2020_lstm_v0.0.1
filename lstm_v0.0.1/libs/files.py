from abc import *

import json
import glob

DATA_ROOT_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_doc2vec_v0.0.1/data"

class File(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def _get_contents():
        pass

    @staticmethod
    @abstractmethod
    def _get_file_path():
        pass


class JsonFile(File):
    @staticmethod
    def _get_contents(file_path: str):
        try:
            with open(file_path, mode="r") as f:
                contents = json.load(f)
        except Exception as e:
            print(e)
            return []
        else:
            return contents
        
    @staticmethod
    def _get_file_path(target: str):
        return glob.glob(
            "{data_root_path}/{target}/**/*.json".format(
                data_root_path=DATA_ROOT_PATH,
                target=target
            ),
            recursive=True
        )
    
