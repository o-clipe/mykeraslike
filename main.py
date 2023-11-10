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


def one_hot_encode(Y: np.ndarray):
    y = np.zeros(shape=(Y.shape + (np.max(Y) + 1,)))
    y[np.arange(Y.size), Y] = 1.
    return y


y_train = one_hot_encode(np.array(y_train))
y_test = one_hot_encode(np.array(y_test))

W1 = np.random.randn(28*28, 512)
B1 = np.zeros(512)
W2 = np.random.randn(512, 10)
B2 = np.zeros(10)

mW1 = np.ones(shape=(28*28, 512))
mB1 = np.ones(shape=(512))
mW2 = np.ones(shape=(512, 10))
mB2 = np.ones(shape=(10))

sdW1 = np.ones(shape=(28*28, 512))
sdB1 = np.ones(shape=(512))
sdW2 = np.ones(shape=(512, 10))
sdB2 = np.ones(shape=(10))

def ReLU(Z):
    return np.multiply(Z > 0., Z)

def ReLU_grad(Z):
    return (Z > 0.).astype('float32')


def SoftMax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def SoftMax_grad(Z):
    return 1.
    # ignore this bullshit, messes everything up
    # return np.exp(Z) / (np.square(np.sum(np.exp(Z))) + 1e-8)

def RMS(G):
    return np.sqrt(np.mean(np.square(G)) + 1e-8)


def calc_loss(Y, y):
    log_arr = np.array([-1 * np.log(Y[i][np.argmax(y[i])]) for i in range(len(Y))])
    # print(" loss of 1: ", Y[1][np.argmax(y[1])], "to", np.log(Y[1][np.argmax(y[1])]), end=" ")
    return np.average(log_arr)


# def calc_loss(Y, y):
#     return np.average(-1 * np.log(Y[np.argmax(y)]))

def calc_acc(Y, y):
    acc_arr = np.array([np.argmax(Y[i]) == np.argmax(y[i]) for i in range(len(Y))]).astype('float32')
    return np.average(acc_arr)


def get_signs(G):
    return G > 0.

def update_momentum(dW1, dB1, dW2, dB2):
    global mW1, mW2, mB1, mB2
    global sdW1, sdW2, sdB1, sdB2

    signs = get_signs(dW1)
    sW1 = (sdW1 == signs)
    sdW1 = signs
    sW1 = np.select([sW1 is True, sW1 is False], [1.2, 0.5], 1.)
    mW1 = np.multiply(mW1, sW1)
    mW1 = np.select([mW1 > 100, mW1 < 1e-8], [100., 1e-8], mW1)

    signs = get_signs(dW2)
    sW2 = (sdW2 == signs)
    sdW2 = signs
    sW2 = np.select([sW2 is True, sW2 is False], [1.2, 0.5], 1.)
    mW2 = np.multiply(mW2, sW2)
    mW2 = np.select([mW2 > 100, mW2 < 1e-8], [100., 1e-8], mW2)

    signs = get_signs(dB1)
    sB1 = (sdB1 == signs)
    sdB1 = signs
    sB1 = np.select([sB1 is True, sB1 is False], [1.2, 0.5], 1.)
    mB1 = np.multiply(mB1, sB1)
    mB1 = np.select([mB1 > 100, mB1 < 1e-8], [100., 1e-8], mB1)

    signs = get_signs(dB2)
    sB2 = (sdB2 == signs)
    sdB2 = signs
    sB2 = np.select([sB2 is True, sB2 is False], [1.2, 0.5], 1.)
    mB2 = np.multiply(mB2, sB2)
    mB2 = np.select([mB2 > 100, mB2 < 1e-8], [100., 1e-8], mB2)


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

def train(x, y, epochs=10, lr=0.1, momentum=0.1):
    global W1, W2, B1, B2
    global mW1, mW2, mB1, mB2

    for e in range(epochs):
        print("epoch:", e, end="")
        pred = predict_batch(x)
        print(" loss:", calc_loss(pred, y_train), end="")
        print(" acc: ", calc_acc(pred, y_train))
        range_ = np.arange(len(x))
        np.random.shuffle(range_)
        for p in range_:
            dW1, dB1, dW2, dB2 = back_prop(*forward_prop(x[p]), y[p])
            # print(np.sum(mW1))
            # print(np.sum(mW2))
            if momentum > 0:
                update_momentum(dW1, dB1, dW2, dB2)
            W1 = np.subtract(W1, dW1 / RMS(dW1) * (lr * (1 - momentum) + mW1 * momentum))
            B1 = np.subtract(B1, dB1 / RMS(dB1) * (lr * (1 - momentum) + mB1 * momentum))
            W2 = np.subtract(W2, dW2 / RMS(dW2) * (lr * (1 - momentum) + mW2 * momentum))
            B2 = np.subtract(B2, dB2 / RMS(dB2) * (lr * (1 - momentum) + mB2 * momentum))


train(x_train[:10], y_train[:10], epochs=100, lr=0.001, momentum=0.1)

print("final acc: ", calc_acc(predict_batch(x_test), y_test))










