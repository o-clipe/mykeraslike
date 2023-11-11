import numpy as np
import matplotlib.pyplot as plt
import random

from mnist_data import *


mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = np.array(x_train).astype('float32') / 255
x_train = x_train.reshape(60000, 28*28)
x_test = np.array(x_test).astype('float32') / 255
x_test = x_test.reshape(10000, 28*28)


def one_hot_encode(Y):
    y = np.zeros(shape=(Y.shape + (np.max(Y) + 1,)))
    y[np.arange(Y.size), Y] = 1.
    return y


y_train = one_hot_encode(np.array(y_train))
y_test = one_hot_encode(np.array(y_test))

W1 = np.random.randn(28*28, 512)
B1 = np.zeros(512)
W2 = np.random.randn(512, 10)
B2 = np.zeros(10)

sgW1 = np.ones(shape=(28*28, 512))
sgB1 = np.ones(shape=(512))
sgW2 = np.ones(shape=(512, 10))
sgB2 = np.ones(shape=(10))

def ReLU(Z):
    return np.multiply(Z > 0., Z)

def ReLU_grad(Z):
    return (Z > 0.).astype('float32')


def SoftMax(Z):
    return np.divide(np.exp(Z), np.sum(np.exp(Z), axis=-1).reshape(-1, 1), )

def SoftMax_grad(Z):
    # return 1.
    # ignore this bullshit, messes everything up
    return np.exp(Z) / (np.square(np.sum(np.exp(Z))) + 1e-8)

def calc_moving_gradient(sG, G):
    return 0.9 * sG + 0.1 * np.square(G)


def calc_loss(Y, y):
    log_arr = np.array([-1 * np.log(Y[i][np.argmax(y[i])]) for i in range(len(Y))])
    # print(Y[1])
    # print(y[1])
    # print(" loss of 1: ", Y[1][np.argmax(y[1])], "to", np.log(Y[1][np.argmax(y[1])]), end=" ")
    # print(np.sum(log_arr))
    return np.average(log_arr)


# def calc_loss(Y, y):
#     return np.average(-1 * np.log(Y[np.argmax(y)]))

def calc_acc(Y, y):
    acc_arr = np.array([np.argmax(Y[i]) == np.argmax(y[i]) for i in range(len(Y))]).astype('float32')
    return np.average(acc_arr)


def get_signs(G):
    return G > 0.

def update_RProp(dW1, dB1, dW2, dB2):
    global aW1, aW2, aB1, aB2
    global sdW1, sdW2, sdB1, sdB2

    signs = get_signs(dW1)
    sW1 = (sdW1 == signs)
    sdW1 = signs
    sW1 = np.select([sW1 is True, sW1 is False], [1.2, 0.5], 1.)
    aW1 = np.multiply(aW1, sW1)
    aW1 = np.select([aW1 > 50, aW1 < 1e-6], [50., 1e-6], aW1)

    signs = get_signs(dW2)
    sW2 = (sdW2 == signs)
    sdW2 = signs
    sW2 = np.select([sW2 is True, sW2 is False], [1.2, 0.5], 1.)
    aW2 = np.multiply(aW2, sW2)
    aW2 = np.select([aW2 > 50, aW2 < 1e-6], [50., 1e-6], aW2)

    signs = get_signs(dB1)
    sB1 = (sdB1 == signs)
    sdB1 = signs
    sB1 = np.select([sB1 is True, sB1 is False], [1.2, 0.5], 1.)
    aB1 = np.multiply(aB1, sB1)
    aB1 = np.select([aB1 > 50, aB1 < 1e-6], [50., 1e-6], aB1)

    signs = get_signs(dB2)
    sB2 = (sdB2 == signs)
    sdB2 = signs
    sB2 = np.select([sB2 is True, sB2 is False], [1.2, 0.5], 1.)
    aB2 = np.multiply(aB2, sB2)
    aB2 = np.select([aB2 > 50, aB2 < 1e-6], [50., 1e-6], aB2)


def update_RMSProp(dW1, dB1, dW2, dB2):
    global sgW1, sgW2, sgB1, sgB2
    sgW1 = calc_moving_gradient(sgW1, dW1)
    sgB1 = calc_moving_gradient(sgB1, dB1)
    sgW2 = calc_moving_gradient(sgW2, dW2)
    sgB2 = calc_moving_gradient(sgB2, dB2)
    return np.sqrt(sgW1) + 1e-8, np.sqrt(sgB1) + 1e-8, np.sqrt(sgW2) + 1e-8, np.sqrt(sgB2) + 1e-8


def predict_batch(X):
    Z1 = np.add(np.dot(X, W1) / 28 * 28, B1)
    A1 = ReLU(Z1)
    Z2 = np.add(np.dot(A1, W2) / 512, B2)
    A2 = SoftMax(Z2)
    return A2


def forward_prop(X):
    Z1 = np.add(np.dot(X.reshape(1, -1), W1) / 28*28, B1)
    A1 = ReLU(Z1)
    Z2 = np.add(np.dot(A1, W2) / 512, B2)
    A2 = SoftMax(Z2)
    return X, Z1, A1, Z2, A2


def back_prop(X, Z1, A1, Z2, A2, Y):
    # dZ2 means dC/dZ2 and keeps like that for the others
    dA2 = np.subtract(A2, Y)
    dZ2 = np.multiply(dA2, SoftMax_grad(Z2))
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = np.multiply(dA1, ReLU_grad(Z1))
    dW2 = np.dot(A1.T, dZ2)
    dB2 = dZ2
    dW1 = np.dot(X.reshape(1, -1).T, dZ1)
    dB1 = dZ1
    return dW1, dB1, dW2, dB2

def train(x, y, epochs=10, lr=0.1):
    global W1, W2, B1, B2

    for e in range(epochs):
        print("epoch:", e, end="")
        pred = predict_batch(x)
        print(" loss:", calc_loss(pred, y_train), end="")
        print(" acc: ", calc_acc(pred, y_train))
        range_ = np.arange(len(x))
        np.random.shuffle(range_)
        for p in range_:
            dW1, dB1, dW2, dB2 = back_prop(*forward_prop(x[p]), y[p])
            rmsW1, rmsB1, rmsW2, rmsB2 = update_RMSProp(dW1, dB1, dW2, dB2)
            W1 = np.subtract(W1, dW1 / rmsW1 * lr)
            B1 = np.subtract(B1, dB1 / rmsB1 * lr)
            W2 = np.subtract(W2, dW2 / rmsW2 * lr)
            B2 = np.subtract(B2, dB2 / rmsB2 * lr)


train(x_train, y_train, epochs=10, lr=0.001)

print("final acc: ", calc_acc(predict_batch(x_test), y_test))
