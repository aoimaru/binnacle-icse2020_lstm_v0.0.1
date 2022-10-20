from libs.files import *

import pprint
import sys
import os

INDEX_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/index"
INDEX_02_PATH = "/Users/nakamurahekikai/Desktop/binnacle-icse2020_lstm_v0.0.1/index02"


OPTION_01 = 1

def main(args):
    root_path = "{}/{}".format(INDEX_PATH, args[OPTION_01])
    index_paths = JsonFile._get_file_path(root_path=root_path)
    for index_path in index_paths:
        base_name = os.path.basename(index_path); base_name = base_name.replace(".json", "")
        contents = JsonFile._get_contents(index_path)
        base_dict = dict()
        for content in contents.keys():
            content = content.replace(base_name+":", "")
            first, second = content.split(":")
            print(first, second)
            if not first in base_dict:
                base_dict[first] = list()
            base_dict[first].append(second)
        
        for base_key, base_value in base_dict.items():
            base_dict[base_key] = sorted(base_value)
        
        pprint.pprint(base_dict)

        with open("{}/{}/{}.json".format(INDEX_02_PATH, args[OPTION_01], base_name), mode="w") as f:
            json.dump(base_dict, f, indent=4)




if __name__ == "__main__":
    main(sys.argv)