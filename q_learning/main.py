from snake import Snake, inference, cancel_inference
from random import randint
import numpy as np
import time

GRID = [11, 11]
RAND_GRID = lambda: [randint(3, 11) * 2 - 1 for _ in range(2)]
new_game = lambda: Snake(RAND_GRID(), 2)

Q = np.zeros(shape=(4, 8, 2, 2, 2, 2, 4), dtype='float32')  # [direction (4), food (8), danger(2^4), actions (4)]


def Q_idx(game: Snake):
    head = game.position[0]
    x_direction = head[0] - game.position[1][0]
    y_direction = head[1] - game.position[1][1]
    direction = 0

    if x_direction < 0:
        direction = 0
    if x_direction > 0:
        direction = 1
    if y_direction < 0:
        direction = 2
    if y_direction > 0:
        direction = 3

    x_candy = game.candy[0] - head[0]
    y_candy = game.candy[1] - head[1]
    candy = -1

    if x_candy < 0:
        if y_candy < 0:
            candy = 0
        elif y_candy == 0:
            candy = 7
        elif y_candy > 0:
            candy = 6
    elif x_candy == 0:
        if y_candy < 0:
            candy = 1
        elif y_candy > 0:
            candy = 5
    elif x_candy > 0:
        if y_candy < 0:
            candy = 2
        if y_candy == 0:
            candy = 3
        if y_candy > 0:
            candy = 4

    danger = 0

    def check_point(point):
        for pixel in game.position:  # stumbles
            if point[0] == pixel[0] and point[1] == pixel[1]:
                return True
        for pt, max in zip(point, game.grid):  # hits wall
            if pt >= max or pt < 0:
                return True
        return False

    nx = 1 if check_point((head[0] - 1, head[1])) else 0
    px = 1 if check_point((head[0] + 1, head[1])) else 0
    ny = 1 if check_point((head[0], head[1] - 1)) else 0
    py = 1 if check_point((head[0], head[1] + 1)) else 0

    for exp, do in zip(reversed(range(4)), (nx, px, ny, py)):
        danger += 2 ** exp if do else 0

    assert direction >= 0 and candy >= 0
    return direction, candy, nx, px, ny, py


def train():
    iterations = 100_000
    for i in range(iterations):
        game = new_game()
        for j in range(1000):
            if not game.alive:
                break
            observations = Q_idx(game)
            a = randint(0, i // (iterations // 10))
            if a:
                action = np.argmax(Q[observations])
            else:
                action = randint(0, 3)
            currentQ = Q[observations + (action,)]
            reward = game.move(action)
            print(f"\r{i+1}/{iterations}", end='')
            maxQ = np.max(Q[Q_idx(game)])
            Q[observations + (action,)] = currentQ + 0.1 * (reward + 0.2 * maxQ - currentQ)


def test():
    game = new_game()
    for i in range(10000):
        time.sleep(0.1)
        if not game.alive:
            game = Snake(GRID, 2)
        observations = Q_idx(game)
        action = np.argmax(Q[observations])
        inference(game, action, f"{Q[observations]}: {np.argmax(Q[observations])} with Q idx :{observations}")

    cancel_inference()


def test_with_future_predictions(n_rounds: int, look_at_two: bool = False):
    def assess_state_action(game_state: Snake, rounds: int, look_at_two: bool) -> (bool, int):
        actions = np.flip(np.argsort(Q[Q_idx(game_state)]))
        if not game_state.alive:
            return False, actions[0]
        if rounds == 0:
            return True, actions[0]

        new_state = game_state.__copy__()
        new_state.move(actions[0])
        if assess_state_action(new_state, rounds-1, look_at_two)[0]:
            return True, actions[0]
        new_state = game_state.__copy__()
        new_state.move(actions[1])
        if assess_state_action(new_state, rounds-1, look_at_two)[0]:
            return True, actions[1]
        new_state = game_state.__copy__()
        new_state.move(actions[2])
        if assess_state_action(new_state, rounds-1, look_at_two)[0] and not look_at_two:
            return True, actions[2]

        return False, actions[0]

    for i in range(100):
        game = Snake(GRID, 2)
        while game.alive:
            time.sleep(0.05)
            observations = Q_idx(game)
            expected, action = assess_state_action(game.__copy__(), n_rounds, look_at_two)
            future_message = "All good"
            if not expected:
                future_message = "DOOMED"
            elif action != np.argmax(Q[observations]):
                future_message = f"changed course to {action}"

            inference(game, action, f"{Q[observations]}: {np.argmax(Q[observations])} with Q idx :{observations}"
                                    f"\n Future message: {future_message}")


if __name__ == "__main__":
    train()
    test_with_future_predictions(27, look_at_two=True)
