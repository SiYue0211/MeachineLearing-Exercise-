import numpy as np

def classifyVector(inX, weights):
    prob = sigmoid(np.sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def sigmoid(n):
    return 1.0 / (1.0 + np.exp(-n))


def colicTest():
    training_set = []
    training_label = []
    with open('horseColicTraining.txt') as t_train:
        file_train = t_train.readlines()

    for line in file_train:
        cur_line = line.strip().split('\t')
        training_set.append(cur_line[:21])
        training_label.append(float(cur_line[-1]))

    training_set = np.array(training_set, dtype=np.float)
    training_label = np.array(training_label, dtype=np.float)

    weight = stocGradAscent(training_set, training_label, 500)

    error_count = 0

    with open('horseColicTest.txt') as f_test:
        file_test = f_test.readlines()

    test_set = []
    test_label = []

    for line in file_test:
        cur_line = line.strip().split('\t')
        test_set.append(cur_line[:21])
        test_label.append(float(cur_line[-1]))

    test_set = np.array(test_set, dtype=np.float)
    test_label = np.array(test_label, dtype=np.float)

    num_test_vec = len(file_test)

    for i in range(num_test_vec):
        pre_label = classifyVector(test_set[i], weight)
        if pre_label != test_label[i]:
            error_count += 1

    error_rate = float(error_count) / float(num_test_vec)
    print("\nThe error rate of this test is: %f" %error_rate)
    return error_rate

def multi_test():
    num_test = 10
    error_sum = 0.0
    for k in range(num_test):
        error_sum += colicTest()

    print("\nAfter %d iterations the average error rate is: %f"%(num_test, error_sum/float(num_test)))

def stocGradAscent(train_set, train_label, num_iter=200):
    m, n = np.shape(train_set)
    weight = np.ones(n)
    for i in range(num_iter):
        data_index = list(range(m))
        for j in range(m):
            lr = 0.01 + 4.0 / (1.0 + i + j)
            random_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(np.sum(train_set[random_index]*weight))
            error = train_label[random_index] - h
            weight = weight + train_set[random_index] * error * lr
            del(data_index[random_index])
    return weight





if __name__ == '__main__':
    multi_test()













