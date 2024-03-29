import numpy as np

from mnist_data import MnistDataloader, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath


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

vW1 = np.ones(shape=(28 * 28, 512))
vB1 = np.ones(shape=(512))
vW2 = np.ones(shape=(512, 10))
vB2 = np.ones(shape=(10))

def ReLU(Z):
    return np.multiply(Z > 0., Z)

def ReLU_grad(Z):
    return (Z > 0.).astype('float32')


def SoftMax(Z):
    actual_expZ = np.nan_to_num(np.exp(Z - np.max(Z)))
    return np.divide(actual_expZ, np.sum(actual_expZ, axis=-1).reshape(-1, 1))

def CrossEntropy(Y, y):
    log_arr = np.sum(np.array(-1 * y * np.log(np.clip(Y, 1e-7, 1 - 1e-7))), axis=-1)
    return np.average(log_arr)


def CrossEntropy_Softmax_grad(Y, y):
    return np.subtract(Y, y)  # this represents SoftMax(Z) - y which is equivalent to f' being f(Z) = CrossEntropy(SoftMax(Z))


def calc_moving_gradient(vG, G):
    return 0.9 * vG + 0.1 * np.square(G)

def calc_acc(Y, y):
    acc_arr = np.array([np.argmax(Y[i]) == np.argmax(y[i]) for i in range(len(Y))]).astype('float32')
    return np.average(acc_arr)


def update_RMSProp(dW1, dB1, dW2, dB2):
    global vW1, vW2, vB1, vB2
    vW1 = calc_moving_gradient(vW1, dW1)
    vB1 = calc_moving_gradient(vB1, dB1)
    vW2 = calc_moving_gradient(vW2, dW2)
    vB2 = calc_moving_gradient(vB2, dB2)
    return np.sqrt(vW1 + 1e-7),  np.sqrt(vB1 + 1e-7), np.sqrt(vW2 + 1e-7), np.sqrt(vB2 + 1e-7)


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
    dZ2 = CrossEntropy_Softmax_grad(A2, Y)  # Z2 is deterministic by A2
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = np.multiply(dA1, ReLU_grad(Z1))
    dW2 = np.dot(A1.T, dZ2)
    dB2 = dZ2
    dW1 = np.dot(X.reshape(1, -1).T, dZ1)
    dB1 = dZ1
    return dW1, dB1, dW2, dB2


def train(x, y, epochs=10, lr=0.001):
    global W1, W2, B1, B2, vW1, vW2, vB1, vB2

    for e in range(epochs):
        print("epoch:", e, end="")
        pred = predict_batch(x)
        print(" loss:", CrossEntropy(pred, y), end="")
        print(" acc: ", calc_acc(pred, y))

        range_ = np.arange(len(x))
        np.random.shuffle(range_)
        for p in range_:
            dW1, dB1, dW2, dB2 = back_prop(*forward_prop(x[p]), y[p])
            rmsW1, rmsB1, rmsW2, rmsB2 = update_RMSProp(dW1, dB1, dW2, dB2)
            W1 = np.subtract(W1, dW1 * lr / rmsW1)
            B1 = np.subtract(B1, dB1 * lr / rmsB1)
            W2 = np.subtract(W2, dW2 * lr / rmsW2)
            B2 = np.subtract(B2, dB2 * lr / rmsB2)

    print("epoch:", epochs, end="")
    pred = predict_batch(x)
    print(" loss:", CrossEntropy(pred, y), end="")
    print(" acc: ", calc_acc(pred, y))


train(x_train[:1000], y_train[:1000], epochs=100, lr=0.001)

print("final acc: ", calc_acc(predict_batch(x_test), y_test))
