import numpy as np
import matplotlib.pyplot as plt

def loadDataset():
    dataMat = []
    labelMat = []
    with open('testSet.txt') as f:
        lines = f.readlines()

    for line in lines:
        line_arr = line.strip().split()
        dataMat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        labelMat.append(int(line_arr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    data_matrix = np.mat(dataMatIn)
    label_matrix = np.mat(classLabels).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = label_matrix - h
        weights = weights + alpha * data_matrix.transpose() * error
    return weights

def stocGradAscent(dataMatrix, classLabel, num_iter=250):
    m, n = np.shape(dataMatrix)
    dataMatrix = np.array(dataMatrix, dtype=np.float64)
    weight = np.ones(n)
    for j in range(num_iter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randomIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(dataMatrix[i]*weight))
            error = classLabel[i] - h
            weight = weight + alpha * error* dataMatrix[randomIndex]
            del(dataIndex[randomIndex])
    return weight

def plotBestFit(wei):
    weight = wei
    # weight = wei.getA()
    data_mat, label_mat = loadDataset()
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]
    x_coord1 = []
    y_coord1 = []
    x_coord2 = []
    y_coord2 = []

    for i in range(n):
        if label_mat[i] == 1:
            x_coord1.append(data_arr[i, 1])
            y_coord1.append(data_arr[i, 2])
        else:
            x_coord2.append(data_arr[i, 1])
            y_coord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_coord1, y_coord1, s=30, c='red', marker='s')
    ax.scatter(x_coord2, y_coord2, s=30, c='green')

    x = np.arange(-3.0, 3.0, 0.1)

    y = -(weight[0] + weight[1]*x) / weight[2]

    ax.plot(x, y)

    plt.xlabel('X1')
    plt.ylabel("X2")

    plt.show()




if __name__ == '__main__':
    data_arr, label_arr = loadDataset()
    weight = stocGradAscent(data_arr, label_arr)
    plotBestFit(weight)


















