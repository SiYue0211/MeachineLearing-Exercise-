import numpy as np
import pickle

class Classify():
    def __init__(self, filepath):
        self._filepath = filepath

    def getData(self):
        with open(self._filepath) as f:
            lines = f.readlines()
            length = len(lines)
        dataSet = []
        labels = []
        index = 0
        for line in lines:
            line = line.strip().split('\t')
            data = line[0:4]
            label = line[-1]
            dataSet.append(data)
            labels.append(label)

        assert len(dataSet) == len(labels), "num_dataSet != num_labels"

        self._dataSet = dataSet
        self._labels = labels







if __name__ == "__main__":
    cls = Classify('lenses.txt')
    cls.getData()