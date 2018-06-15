import numpy as np
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

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """

    :param dataMatIn: 输入数据
    :param classLabels: 输入数据的标签
    :param C: 惩罚因子
    :param toler: 容错值
    :param maxIter: 最大迭代次数
    :return:
    """
    dataMatrix = np.mat(dataMatIn)
    labelMatrix = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # 预测值 f(x_i)
            fXi = float(np.multiply(alphas, labelMatrix).T * \
                        (dataMatrix * dataMatrix[i, :].T)) + b
            # 真实值和预测值的差值
            Ei = fXi - float(labelMatrix[i])
            # 如果在容错范围内，不需要处理，如果超出容错范围，进入if语句
            if ((labelMatrix[i]*Ei < - toler) and (alphas[i] < C)) or \
                    ((labelMatrix[i]*Ei > toler) and (alphas[i] > 0)):
                # 选一个和i不同的数作为j
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMatrix).T *\
                            (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMatrix[j])
                # alpha_i old
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # data_i, data_j不同的标签，处理方法不同
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    print("L == H")
                    continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T

                if eta >= 0:
                    print("eta >= 0")
                    continue

                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue

                alphas[i] += labelMatrix[j]*labelMatrix[i] * (alphaJold - alphas[j])

                b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i,:] * dataMatrix[i, :].T - \
                    labelMatrix[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j,:] * dataMatrix[j, :].T

                b2 = b - Ej - labelMatrix[i] * (alphas[i] - alphaIold)*\
                    dataMatrix[i,:] * dataMatrix[j,:].T - \
                    labelMatrix[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if(alphas[i] > 0) and (alphas[i] < C):
                    b = b1
                elif(alphas[i] > 0) and (alphas[i] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter: %d i:%d, pairs changed %d' %(iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number : %d" %iter)
    return b, alphas



def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas =smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)