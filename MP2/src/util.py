import json
import pickle


class Json:
    @staticmethod
    def loadf(filepath):
        with open(filepath) as rf:
            data = json.load(rf)
        return data

    @staticmethod
    def dumpf(data, filepath):
        with open(filepath, 'w') as wf:
            json.dump(data, wf, indent=4)


class Pickle:
    @staticmethod
    def loadf(filepath):
        with open(filepath, 'rb') as rf:
            data = pickle.load(rf)
        return data

    @staticmethod
    def dumpf(data, filepath):
        with open(filepath, 'wb') as wf:
            pickle.dump(data, wf)