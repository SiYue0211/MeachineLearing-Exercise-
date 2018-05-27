import numpy as np
import os
import sys
sys.path.append(os.getcwd())

from BayesianClassification.demo import createVocabList, setOfWord2Vect, trainNBO, classifyNB

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []

    for i in range(1, 26):
        wordList = open('BayesianClassification/MailClassification/email/spam/%d.txt' %i, encoding="ISO-8859-1").read()
        wordList = textParse(wordList)
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open('BayesianClassification/MailClassification/email/ham/%d.txt' %i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)


    vocaList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []

    # 随机选取十个作为验证集
    for i in range(10):
        randIdex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(randIdex)
        del(trainingSet[randIdex])

    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vect(vocaList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p1V, p0V, pSpam = trainNBO(np.array(trainMat), np.array(trainClasses))

    errorCount = 0

    for docIndex in testSet:
        wordVector = setOfWord2Vect(vocaList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print(errorCount)
    print('The error rate is: ', float(errorCount)/float(len(testSet)))


if __name__  == "__main__":
    spamTest()




















