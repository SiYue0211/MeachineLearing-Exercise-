import numpy as np
import matplotlib.pyplot as plt

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = len(dataMatIn)
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        lines = fr.readlines()

    for line in lines:
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0, m))
    return j

def calcEk(oS, k):
    """
    计算预测值和实际值的差值
    :param oS: 对象
    :param k: 计算的是f(x_k)和实际值的差
    :return:
    """
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0

    oS.eCache[i] = [1, Ei]
    # 找到eCache不等于0的标号
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]

    if (len(validEcacheList) > 1):
        for k in validEcacheList:
            if k == i:
                continue
            else:
                Ek = calcEk(oS, k)
                deltaE = abs(Ei - Ej)
                if deltaE > maxDeltaE:
                    maxDeltaE = deltaE
                    maxK = k
                    Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updataEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0

        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - \
              oS.X[i, :] * oS.X[i, :].T - \
              oS.X[j, :] * oS.X[j, :].T

        if eta >= 0:
            print("eta >= 0")
            return 0

        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updataEk(oS, j)

        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0

        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updataEk(oS, i)

        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
             oS.X[i, :] * oS.X[i, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * \
             oS.X[j, :] * oS.X[j, :].T

        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
             oS.X[i, :] * oS.X[j, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * \
             oS.X[j, :] * oS.X[j, :].T
        if (oS.alphas[i] > 0) and (oS.alphas[i] < oS.C):
            oS.b = b1
        elif (oS.alphas[i] > 0) and (oS.alphas[i] < oS.C):
            oS.b = b2
        else:
            b = (b1 + b2) / 2.0
        return 1
    return 0

def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter  = 0
    entireSet = True
    alphaPairsChanged = 0

    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            # os.m所有样本数量
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print('fullSet, iter: %d i:%d, pairs changed %d' %(iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter: %d i: %d pairs changed %d'%(iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" %iter)
    return oS.b, oS.alphas

def drawPicture(data, w, b):
    data = np.array(data)
    label1_x = []
    label1_y = []
    label2_x = []
    label2_y = []
    label3_x = []
    label3_y = []
    size = len(data)
    for i in range(size):
        temp = data[i] * np.mat(w) + b
        if temp > 1:
            label1_x.append(float(data[i][0]))
            label1_y.append(float(data[i][1]))
        elif temp < -1:
            label2_x.append(float(data[i][0]))
            label2_y.append(float(data[i][1]))
        else:
            label3_x.append(float(data[i][0]))
            label3_y.append(float(data[i][1]))
    max_index = np.argmax(data, 0)[0]
    print("max index", max_index)
    max_ = np.array(data[max_index,:])
    min_index = np.argmin(data, 0)
    min_ = np.array(data[min_index])
    plt.scatter(label1_x, label1_y, c='red', s= 40)
    plt.scatter(label2_x, label2_y, c='green', s=40)
    plt.scatter(label3_x, label3_y, c='black', s=40)
    plt.plot(min_, max_)
    plt.show()


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alpha = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    ws = calcWs(alpha, dataArr, labelArr)
    print("ws: ", ws)
    drawPicture(dataArr, ws, b)

































