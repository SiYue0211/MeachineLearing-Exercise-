from math import log
import pickle


def createTree(dataSet, labels):
    labels_temp = labels.copy()
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    best_feature = chooseBestFeatureToSplit(dataSet)
    best_feature_labels = labels_temp[best_feature]
    myTree = {best_feature_labels:{}}
    del(labels_temp[best_feature])
    feature_values = [example[best_feature] for example in dataSet]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels_temp[:]
        myTree[best_feature_labels][value] = createTree(splitDataSet(dataSet, best_feature, value), \
                                                        sub_labels)
    return myTree

def majorityCnt(classList):
    class_count = {}
    for vote in classList:
        if vote not in class_count.keys():
            class_count[vote] = 0;
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=lambda d:d[1], reverse=True)
    return sorted_class_count[0][0]


def chooseBestFeatureToSplit(dataSet):
    # 列数
    num_feature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_feature):
        feat_list = [example[i] for example in dataSet]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            subDataSet = splitDataSet(dataSet, i, value)
            prop = float(len(subDataSet)) / float(len(dataSet))
            new_entropy += prop * calcShannonEnt(subDataSet)
        info_gain = baseEntropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
        return best_feature

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for feat_vec in dataSet:
        if feat_vec[axis] == value:
            reduce_fea_vec = feat_vec[:axis]
            reduce_fea_vec.extend(feat_vec[axis+1:])
            retDataSet.append(reduce_fea_vec)
    return retDataSet

def calcShannonEnt(dataset):
    num_entries = len(dataset)
    lable_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in lable_counts.keys():
            lable_counts[current_label] = 0
        lable_counts[current_label] += 1
    shannon_ent = 0.0
    for key in lable_counts:
        prob = float(lable_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def createDataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no sufacing', 'flippers']
    return dataSet, labels

def classify(inputTree, feat_labels, testVec):
    first_str = list(inputTree.keys())[0]
    print(first_str)
    global class_label
    seconde_dict = inputTree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in seconde_dict.keys():
        if testVec[feat_index] == key:
            if type(seconde_dict[key]).__name__ == "dict":
                classify(seconde_dict[key], feat_labels, testVec)
            else:
                class_label = seconde_dict[key]
    return class_label

def storeTree(inTree, filename):
    with open(filename, 'wb+') as f:
        pickle.dump(inTree, f)

def grabTree(filename):
    with open(filename, 'rb+') as f:
        return pickle.load(f)

if __name__ == "__main__":
    dataSet, labels = createDataset()
    my_tree = createTree(dataSet, labels)
    storeTree(my_tree, 'classifyStore.txt')
    print(grabTree('classifyStore.txt'))
    label = classify(my_tree, labels, [1, 0])
    print(label)
















