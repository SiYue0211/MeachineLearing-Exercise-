import numpy as np
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] # 1表示侮辱性文字，0表示正常言论
    return postingList, classVec

def createVocabList(dataSet):
    vacabSet = set([])
    for document in dataSet:
        vacabSet = vacabSet | set(document) # | 表示求并集
    return list(vacabSet)

def setOfWord2Vect(vocabSet, inputSet):
    """
    词集模型， 只要一个单词出现一次，这个单词对应的位置就是1, 出现过多次也是1
    :param vocabSet:
    :param inputSet:
    :return:
    """
    returnVev = [0] * len(vocabSet)
    for word in inputSet:
        if word in vocabSet:
            returnVev[vocabSet.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary!' %word)
    return returnVev

def bagOfWord2Vect(vocabSet, inputSet):
    """
    词袋模型
    :param vocabSet:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabSet)
    for word in inputSet:
        if word in vocabSet:
            returnVec[vocabSet.index(word)] += 1
        else:
            print("the word %s is not in my vocabulary!" %word)
    return returnVec



def trainNBO(trainMatrix, trainCategory):
    """

    :param trainMatrix: 在一个list里有所有单词数据， 如果train数据的某个词存在，则标号为1,否则0
    :param trainCategory: label
    :return:
    """
    # train 样本数
    num_train_doc = len(trainMatrix)
    # train 的特征数
    num_train_feat = len(trainMatrix[0])
    # 负例的概率
    p_aubsive = float(sum(trainCategory)) / float(num_train_doc)
    p0_num = 2.0
    p1_num = 2.0
    p0_vec = np.ones(num_train_feat)
    p1_vec = np.ones(num_train_feat)
    for i in range(num_train_doc):
        if trainCategory[i] == 1:
            # 负样本
            p1_vec += trainMatrix[i]
            p1_num += np.sum(trainMatrix[i])
        elif trainCategory[i] == 0:
            # 正样本
            p0_vec += trainMatrix[i]
            p0_num += np.sum(trainMatrix[i])
    p1_result = np.log(p1_vec / p1_num)
    p0_result = np.log(p0_vec / p0_num)
    return p1_result, p0_result, p_aubsive

def classifyNB(vec2Classify, p0_vec, p1_vec, p_class):
    p0 = np.sum(vec2Classify * p0_vec) + np.log(1.0 - p_class)
    p1 = np.sum(vec2Classify * p1_vec) + np.log(p_class)
    if p0 > p1:
        return 0
    else:
        return 1

def test():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vect(myVocabList, postinDoc))
    p1_vec, p0_vec, p_aubsive = trainNBO(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    test_doc = setOfWord2Vect(myVocabList, testEntry)
    print(testEntry , "classified as : " + str(classifyNB(test_doc, p0_vec, p1_vec, p_aubsive)))
    testEntry = ['stupid', 'garbage']
    test_doc = setOfWord2Vect(myVocabList, testEntry)
    print(testEntry , "classified as :" + str(classifyNB(test_doc, p0_vec, p1_vec, p_aubsive)))




if __name__ == "__main__":
    test()




