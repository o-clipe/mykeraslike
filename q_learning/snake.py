import random
import curses

GRID = [11, 11]
START_SIZE = 1
ZERO = 0

class Snake:
    def __init__(self, grid, length):
        self.grid = grid
        self.dimensions = len(grid)
        self.position = [tuple([d // 2 for d in self.grid]) for _ in range(length)]
        self.place_candy()
        self.alive = True
        self.score = len(self.position) * 100

    def __copy__(self):
        copy = Snake(self.grid, 0)
        copy.position = self.position.copy()
        copy.candy = self.candy
        copy.alive = self.alive
        copy.score = self.score
        return copy


    def place_candy(self):
        self.candy = tuple([random.randint(0, dimension - 1) for dimension in self.grid])
        if self.candy in self.position:
            self.place_candy()

    def move(self, direction: int):
        """Take as direction 2 => [0, 0, 1, 0] => -1 second dimension ... 1 => [0, 1, 0, 0] => +1 first dimension"""
        sign = -1 if direction % self.dimensions == 0 else 1
        dimension = direction // 2
        new_pos = list(self.position[0])
        new_pos[dimension] += sign
        return self.event(tuple(new_pos))

    def event(self, new_point):
        if new_point in self.position:  # stumbles
            return self.die()
        for pt, max in zip(new_point, self.grid):  # hits wall
            if pt >= max or pt < 0:
                return self.die()
        if new_point == self.candy:
            self.position.insert(0, new_point)  # eats candy
            self.place_candy()
            self.score += 100 - self.score % 100
            return self.turn(100)
        else:
            self.position.insert(0, new_point)  # moves
            self.position.pop()
            self.score += 1 if self.score % 100 != 99 else 0
            return 1

    def turn(self, reward):
        return reward

    def die(self):
        self.alive = False
        return -100


def visualize_game(game):
    scrgrid = [["" for _ in range(game.grid[0] + 2)] for __ in range(game.grid[1] + 2)]
    for i in range(game.grid[0] + 2):
        for j in range(game.grid[1] + 2):
            scrgrid[j][i] = "w" if (i == 0 or i == game.grid[0] + 1 or j == 0 or j == game.grid[1] + 1) else " "

    for pixel in game.position:
        scrgrid[pixel[1]+1][pixel[0]+1] = "\u25A0"

    scrgrid[game.candy[1]+1][game.candy[0]+1] = "*"
    return scrgrid

def main(stdscr):
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    game = Snake(GRID, START_SIZE)
    while game.alive == True:

        scrgrid = visualize_game(game)
        for line in scrgrid:
            stdscr.addstr(" ".join(line) + "\n")

        move = stdscr.getch()
        if move == curses.KEY_LEFT:
            game.move(int(0))
            stdscr.clear()
        if move == curses.KEY_RIGHT:
            game.move(int(1))
            stdscr.clear()
        if move == curses.KEY_UP:
            game.move(int(2))
            stdscr.clear()
        if move == curses.KEY_DOWN:
            game.move(int(3))
            stdscr.clear()


    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()
    print("YOU DIED")


class Inference:
    _instance = None
    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_)
        return class_._instance

    def __init__(self):
        self._stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()

    def __call__(self, game: Snake, action: int, bottom_text: str):
        self._stdscr.clear()
        scrgrid = visualize_game(game)
        for line in scrgrid:
            self._stdscr.addstr(" ".join(line) + "\n")
        self._stdscr.addstr(bottom_text)
        reward = game.move(int(action))
        self._stdscr.refresh()
        return reward

    def __del__(self):
        curses.nocbreak()
        curses.echo()
        curses.endwin()


def inference(game: Snake, action: int, bottom_text=None):
    obj = Inference()
    return obj(game, action, bottom_text if bottom_text is not None else "")


def cancel_inference():
    curses.nocbreak()
    curses.echo()
    curses.endwin()



if __name__ == "__main__":
    stdscr = curses.initscr()
    curses.wrapper(main)

    # import time
    #
    # game = Snake(GRID, 2)
    # for i in range(1000):
    #     time.sleep(0.03)
    #     if game.alive == False:
    #         game = Snake(GRID, 2)
    #     inference(game, 1)


