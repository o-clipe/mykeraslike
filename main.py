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


def ReLU(Z):
    return np.multiply(Z > 0., Z)

def ReLU_grad(Z):
    return (Z > 0.).astype('float32')


def SoftMax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def SoftMax_grad(Z):
    return np.exp(Z) / np.square(np.sum(np.exp(Z)))


def calc_loss(Y, y):
    log_arr = np.array([-1 * np.log(Y[i, np.argmax(y[i])]) for i in range(len(Y))])
    # print(" loss of 1: ", Y[1][np.argmax(y[1])], "to", np.log(Y[1, np.argmax(y[1])]), end=" ")
    return np.mean(log_arr)


# def calc_loss(Y, y):
#     return np.average(-1 * np.log(Y[np.argmax(y)]))

def calc_acc(Y, y):
    acc_arr = np.array([np.argmax(Y[i]) == np.argmax(y[i]) for i in range(len(Y))]).astype('float32')
    return np.average(acc_arr)


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
        for p in range(len(x)):
            dW1, dB1, dW2, dB2 = back_prop(*forward_prop(x[p]), y[p])
            W1 = np.subtract(W1, dW1 * lr)
            B1 = np.subtract(B1, dB1 * lr)
            W2 = np.subtract(W2, dW2 * lr)
            B2 = np.subtract(B2, dB2 * lr)


train(x_train[:100], y_train[:100], epochs=10000, lr=0.1)











