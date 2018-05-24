import math
import pickle

class Classify():
    def __init__(self, filepath):
        self._filepath = filepath

    def getData(self):
        """
        从给定路径获取数据集，并将数据集前4列放进dataSet，最后一列作为结果放进result
        :return: dataSet, results
        """
        with open(self._filepath) as f:
            lines = f.readlines()

        dataSet = []
        results = []

        for line in lines:
            line = line.strip().split('\t')
            data = line[:4]
            result = line[-1]
            dataSet.append(data)
            results.append(result)

        assert len(dataSet) == len(results),"the length of dataset must equal to result"
        return dataSet, results

    @staticmethod
    def calculate_entropy(labels):
        """
        计算信息熵
        :param labels:
        :return:
        """
        num_labels = len(labels)
        # 词典类型，用来记录每个分类有多少个
        labels_count = {}
        for label in labels:
            if label not in labels_count.keys():
                labels_count[label] = 0
            labels_count[label] += 1
        result = 0.0
        for value in labels_count.values():
            p = float(value) / float(num_labels)
            result -=  p * math.log(p, 2)
        return result

    @staticmethod
    def majorityCnt(labels):
        maxnum_lab = -1
        max_feature = -1
        count_labels = {}
        for label in labels:
            if label not in count_labels.keys():
                count_labels[label] = 0
            count_labels[label] += 1
        for key, value in count_labels.items():
            if value > maxnum_lab:
                maxnum_lab = value
                max_feature = key
        return max_feature

    @staticmethod
    def choose_best_feat(dataSet, result):
        feat_lists = result
        base_entropy = Classify.calculate_entropy(feat_lists)
        # 列数
        num_feature = len(dataSet[0])
        num_dataSet = len(dataSet)
        max_entropy = 0
        best_feature = -1

        for i in range(num_feature):
            sub_entropys = 0
            feat_list = [example[i] for example in dataSet]
            unique_feat = set(feat_list)
            for feat in unique_feat:
                sub_data, sub_feat = Classify.split_dataSet(dataSet, feat_lists, i, feat)
                sub_entropy = Classify.calculate_entropy(sub_feat)
                p = float(len(sub_data)) / float(num_dataSet)
                sub_entropys -= p * sub_entropy
            cur_entropy = base_entropy + sub_entropys
            if cur_entropy > max_entropy:
                max_entropy = cur_entropy
                best_feature = i
        return best_feature




    @staticmethod
    def split_dataSet(dataSet, feature_list, axis, value):
        sub_feats = []
        sub_dataSet = []
        for i, data in enumerate(dataSet):
            if data[axis] == value:
                sub_data = data[0: axis]
                sub_data.extend(data[axis+1: ])
                sub_dataSet.append(sub_data)
                sub_feats.append(feature_list[i])
        return sub_dataSet, sub_feats


    def create_tree(self, dataSet, result, labels):
        """
        用递归创建一个树
        :param dataSet:
        :param result:
        :param labels:
        :return:
        """
        cur_labels = labels
        feature_list = result
        # 如果labels里的值都是相同的， 则返回标号
        if feature_list.count(feature_list[0]) == len(feature_list):
            return feature_list[0]
        # 如果只有一个特征，那么就选择这个特征里labels中数量最多的
        elif len(feature_list[0]) == 1:
            return Classify.majorityCnt(feature_list)
        # 选择信息增益最大的属性作为划分属性
        best_feature = Classify.choose_best_feat(dataSet, result)
        best_feature_labels = cur_labels[best_feature]

        my_tree = {best_feature_labels:{}}
        del cur_labels[best_feature]

        # 这个属性下由什么子属性组成
        feature_values = [example[best_feature] for example in dataSet]
        feature_values = set(feature_values)

        for feature_value in feature_values:
            sub_label = cur_labels[:]
            sub_dataset, sub_feat = Classify.split_dataSet(dataSet, result, best_feature, feature_value)
            my_tree[best_feature_labels][feature_value] = self.create_tree(sub_dataset, sub_feat, sub_label)
        return my_tree

def store(tree, filepath):
    with open(filepath, 'wb+') as f:
        return pickle.dump(tree, f)

def load(filepath):
    with open(filepath, 'rb+') as f:
        return pickle.load(f)

if __name__ == "__main__":
    cls = Classify('lenses.txt')
    dataSet, result = cls.getData()
    # 给的数据是关于眼睛状况的各种参数，看不出来每列是什么，姑且用A,B,C,D
    labels = ['A', 'B', 'C', 'D']
    tree = cls.create_tree(dataSet, result, labels)
    store(tree, 'tree.txt')
    print(load('tree.txt'))




