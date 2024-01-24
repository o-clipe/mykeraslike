from snake import Snake, inference, cancel_inference
from random import randint, random
import numpy as np
import time


def ArgMax(Y):
    y = np.zeros(shape=Y.shape)
    y[0][np.argmax(Y)] = 1.
    return y

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
    return np.subtract(Y, y)


def MSE(Y, y):
    squared_error = np.square(np.subtract(Y, y))
    return np.average(squared_error)


def MSE_grad(Y, y):
    return 2 * np.subtract(Y, y)


def calc_moving_gradient(vG, G):
    return 0.9 * vG + 0.1 * np.square(G)

def update_Q_RMSProp(dQw1, dQb1, dQw2, dQb2):
    global vQw1, vQb1, vQw2, vQb2
    vQw1 = calc_moving_gradient(vQw1, dQw1)
    vQb1 = calc_moving_gradient(vQb1, dQb1)
    vQw2 = calc_moving_gradient(vQw2, dQw2)
    vQb2 = calc_moving_gradient(vQb2, dQb2)
    return np.sqrt(vQw1 + 1e-7),  np.sqrt(vQb1 + 1e-7), np.sqrt(vQw2 + 1e-7), np.sqrt(vQb2 + 1e-7)


GRID = [11, 11]
RAND_GRID = lambda: [randint(3, 11) * 2 - 1 for _ in range(2)]
new_game = lambda: Snake(RAND_GRID(), 2)

PROXIMITY_NEURONS_ROOT = 5  # make it odd
RANGE = PROXIMITY_NEURONS_ROOT // 2


Qw1 = np.random.randn(6 + PROXIMITY_NEURONS_ROOT ** 2, 256)  # [candy (2), position(4), vision (n^2)]
Qb1 = np.zeros(256)
Qw2 = np.random.randn(256, 4)
Qb2 = np.zeros(4)

vQw1 = np.ones(shape=(6 + PROXIMITY_NEURONS_ROOT ** 2, 256))
vQb1 = np.ones(shape=(256))
vQw2 = np.ones(shape=(256, 4))
vQb2 = np.ones(shape=(4))


def features(game: Snake):
    features_ = np.zeros(6 + PROXIMITY_NEURONS_ROOT ** 2)
    features_[0] = np.float32(game.candy[0] - game.position[0][0]) / 100  # candy x: relative
    features_[1] = np.float32(game.candy[1] - game.position[0][1]) / 100  # candy y: relative
    features_[2] = np.float32(game.position[0][0]) / 100  # position x: absolute
    features_[3] = np.float32(game.position[0][1]) / 100  # position y: absolute
    features_[4] = np.float32(game.grid[0] - game.position[0][0] - 1) / 100   # position x: max - absolute
    features_[5] = np.float32(game.grid[1] - game.position[0][1] - 1) / 100   # position y: max - absolute

    for position in game.position:
        if game.position[0][0] - RANGE <= position[0] <= game.position[0][0] + RANGE and \
                game.position[0][1] - RANGE <= position[1] <= game.position[0][1]:
            features_[6 + (position[0] - game.position[0][0] + RANGE) * PROXIMITY_NEURONS_ROOT + (
                        position[1] - game.position[0][1] + RANGE)] = 1.

    if game.position[0][0] - RANGE < 0:
        difference = RANGE - game.position[0][0]
        for i in range(difference):
            for j in range(PROXIMITY_NEURONS_ROOT):
                features_[6 + i * PROXIMITY_NEURONS_ROOT + j] = 1.

    if game.position[0][0] + RANGE >= game.grid[0]:
        difference = (RANGE + game.position[0][0]) - game.grid[0]
        for i in range(0, difference):
            for j in range(PROXIMITY_NEURONS_ROOT):
                features_[6 + i * PROXIMITY_NEURONS_ROOT + j] = 1.

    if game.position[0][1] - RANGE < 0:
        difference = RANGE - game.position[0][1]
        for j in range(difference):
            for i in range(PROXIMITY_NEURONS_ROOT):
                features_[6 + i * PROXIMITY_NEURONS_ROOT + j] = 1.

    if game.position[0][1] + RANGE >= game.grid[1]:
        difference = (RANGE + game.position[0][1]) - game.grid[0]
        for j in range(PROXIMITY_NEURONS_ROOT - difference, PROXIMITY_NEURONS_ROOT):
            for i in range(PROXIMITY_NEURONS_ROOT):
                features_[6 + i * PROXIMITY_NEURONS_ROOT + j] = 1.

    return features_

def forward_Q_prop(X):
    Z1 = np.add(np.dot(X.reshape(1, -1), Qw1) / 6 + PROXIMITY_NEURONS_ROOT ** 2, Qb1)
    A1 = ReLU(Z1)
    Z2 = np.add(np.dot(A1, Qw2) / 256, Qb2)
    return X, Z1, A1, Z2

def back_Q_prop(X, Z1, A1, Z2, Y):
    dZ2 = MSE_grad(Z2, Y)
    dA1 = np.dot(dZ2, Qw2.T)
    dZ1 = np.multiply(dA1, ReLU_grad(Z1))
    dW2 = np.dot(A1.T, dZ2)
    dB2 = dZ2
    dW1 = np.dot(X.reshape(1, -1).T, dZ1)
    dB1 = dZ1
    return dW1, dB1, dW2, dB2


def act(X):
    return np.argmax(get_Q(X))

def get_Q(X):
    return forward_Q_prop(X)[-1]


def train_episodes(function, episodes=100, epsilon=0.5):
    for i in range(episodes):
        game = new_game()
        for j in range(1000):
            if not game.alive:
                break
            observations = features(game)
            a = random()
            if a > epsilon:
                action = act(observations)
            else:
                action = randint(0, 3)
            function(game, action, observations)

def train_Q(episodes=100, epsilon=0.5, alpha=0.01, gamma=0.2, show=False):
    def one_pass(game, action, observations):
        global Qw1, Qw2, Qb1, Qb2
        X, Z1, A1, currentQ = forward_Q_prop(observations)
        reward = game.move(action)
        maxQ = np.max(get_Q(features(game)))
        updatedQ = np.copy(currentQ)
        updatedQ[0][action] = reward + gamma * maxQ

        if show:
            inference(game, action, f"{currentQ}: best Q {np.argmax(currentQ)} but taken action: {action}")
            time.sleep(0.05)

        dQw1, dQb1, dQw2, dQb2 = back_Q_prop(X, Z1, A1, currentQ, updatedQ)
        rmsW1, rmsB1, rmsW2, rmsB2 = update_Q_RMSProp(dQw1, dQb1, dQw2, dQb2)
        Qw1 = np.subtract(Qw1, dQw1 * alpha / rmsW1)
        Qb1 = np.subtract(Qb1, dQb1 * alpha / rmsB1)
        Qw2 = np.subtract(Qw2, dQw2 * alpha / rmsW2)
        Qb2 = np.subtract(Qb2, dQb2 * alpha / rmsB2)

    train_episodes(one_pass, episodes, epsilon)


def train(episodes):

    def ld(curr, max):  # linear decrease
        return (max // 100 - curr) / (max //100)

    for e in range(episodes // 100):
        print(f"\r Episode: {e * 100}/{episodes}", end="")
        ld_factor = ld(e, episodes)
        train_Q(100, epsilon=ld_factor * 2, alpha=ld_factor * 0.2 + 0.01, gamma=(1 - ld_factor) * 0.2 + 0.1)


def test():
    game = Snake(GRID, 2)
    for i in range(10000):
        time.sleep(0.1)
        if not game.alive:
            game = Snake(GRID, 2)
        observations = features(game)
        action = act(observations)
        inference(game, action, f"testing")

    cancel_inference()


train(episodes=300_000)
test()

