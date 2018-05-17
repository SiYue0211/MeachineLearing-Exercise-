import numpy as np

class KNNClassifier():
    def __init__(self, k=3):
        self._k = k

    def file2matrix(self, filepath):
        """
        从指定路径中读取数据，将特征和标签分开存储
        :param filepath:
        :return:
        """
        with open(filepath) as f:
            all_data = f.readlines()
            num_data = len(all_data) # 数据个数

        data = np.zeros(shape=(num_data, 3))
        labels = np.zeros(shape=(num_data,))

        index = 0

        for line in all_data:
            line = line.strip()
            line = line.split('\t')
            data[index, :] = line[0:3]
            labels[index] = line[-1]
            index += 1

        data.astype(np.float32)
        labels.astype(np.int32)
        return data, labels

    def normMatrix(self, data):
        """
        将数据的每一维正规化
        :param data:
        :return: 正规化后的数据， 在[0, 1]之间
        """
        minVals = data.min(0)
        maxVals = data.max(0)
        length = len(data)
        gap = maxVals - minVals

        data = data - np.tile(minVals, (length, 1))
        data = data / np.tile(gap, (length, 1))

        return data

    def test(self, train_data, train_labels, test_data, test_labels):
        """
        测试
        :param train_data:
        :param train_labels:
        :param test_data:
        :param test_labels:
        :return:
        """
        num_test = len(test_data)

        error_num = 0.0
        for i in range(num_test):
            predict_label = self._classfier0(train_data, test_data[i, :], train_labels)
            if predict_label != test_labels[i]:
                error_num += 1
        print(error_num)
        print('accuracy: %f' % (error_num / float(num_test)))


    def _classfier0(self, train_data, test_data, train_label):
        """
        用测试的数据和每一个训练的数据进行求欧式距离，找出举例最小的前k个，
        让后看这k个里面哪个标号最多。
        :param train_data:
        :param test_data:
        :param train_label:
        :return:
        """
        num_train = len(train_data)
        gap = train_data - np.tile(test_data, (num_train, 1))
        distance = gap**2
        distance = np.sum(distance, axis=1)
        distance = distance**0.5
        max_index = np.argsort(distance)

        cateCount = {}
        for i in range(self._k):
            voteLabel = train_label[max_index[i]]
            cateCount[voteLabel] = cateCount.get(voteLabel, 0) + 1
        sortedValues = sorted(cateCount.items(), key=lambda d:d[1], reverse=True)
        return sortedValues[0][0]

    def shuffle_matrix(self, data1, labels1):
        # 创建两个空数据集
        data = np.empty(shape=data1.shape)
        labels = np.empty(shape=labels1.shape)

        # 将初始的数据打乱
        shuffle_index = np.arange(data.shape[0])
        np.random.shuffle(shuffle_index)
        for i in range(data1.shape[0]):
            data[i, :] = data1[shuffle_index[i], :]
            labels[i] = labels1[shuffle_index[i]]
        return data, labels


if __name__ == "__main__":
    classifier = KNNClassifier(k=3)
    # 从数据集中读取数据
    data, labels = classifier.file2matrix('KNN/datingTestSet2.txt')

    # 打乱初始数据
    data, labels = classifier.shuffle_matrix(data, labels)

    # 一共1000个数据，前900为训练集，后100为测试集
    data = classifier.normMatrix(data)
    train_data = data[:900, :]
    train_labels = labels[:900]
    test_data = data[900:, :]
    test_labels = labels[900:]
    classifier.test(train_data, train_labels, test_data, test_labels)

