import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
np.random.seed(7)
LEARNING_RATE= 0.01
EPOCHS = 20
TRAIN_IMAGE_FILENAME='data/mnist/train-images-idx3-ubyte'
TRAIN_LABEL_FILENAME='data/mnist/train-labels-idx1-ubyte'
TEST_IMAGE_FILENAME='data/mnist/t10k-images-idx3-ubyte'
TEST_LABEL_FILENAME='data/mnist/t10k-labels-idx1-ubyte'

def read_mnist():
    train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
    train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
    test_images  = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
    test_labels  = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

    x_train = train_images.reshape(60000,784)
    mean = np.mean(x_train)
    stddev = np.std(x_train)
    x_train = (x_train-mean) / stddev
    x_test = test_images.reshape(10000,784)
    x_test =(x_test - mean) / stddev

    y_train = np.zeros((60000,10))
    y_test = np.zeros((10000,10)) 

    for i,y in enumerate(train_labels):
        y_train[i][y]=1
    for i,y in enumerate(test_labels):
        y_test[i][y]=1
    return x_train,y_train,x_test,y_test


x_train,y_train,x_test,y_test = read_mnist()
index_list = list(range(len(x_train)))

