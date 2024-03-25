import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
np.random.seed(7)
LEARNING_RATE= 0.01
EPOCHS = 20
TRAIN_IMAGE_FILENAME='data/mnist/train-images-idx3-ubyte'
TRAIN_LABEL_FILENAME='data/mnist/train-labels-idx1-ubyte'
TEST_IMAGE_FILENAME='data/mnist/t10k-images-idx3-ubyte'
TEST_LABEL_FILENAME='data/mnist/t10k-labels-idx3-ubyte'
