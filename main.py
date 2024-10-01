from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

GRID_SIZE = 1024


def makeInitgrid(size: int):
    return np.zeros((size, size), dtype=np.uint8)


def randomize(s: int):
    return np.random.randint(0, 2, size=(s, s), dtype=np.uint8)


def showGrid(grid: np.ndarray):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=1)
    plt.show()
    plt.close()


def ushift(grid: np.ndarray):
    return np.vstack((grid[1:, :], grid[:1, :]))


def dshift(grid: np.ndarray):
    return np.vstack((grid[-1:, :], grid[:-1, :]))


def rshift(grid: np.ndarray):
    return np.hstack((grid[:, -1:], grid[:, :-1]))


def lshift(grid: np.ndarray):
    return np.hstack((grid[:, 1:], grid[:, :1]))


g = np.zeros((1, 1))


def animate(grid):
    global g
    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap="gray", vmin=0, vmax=1, animated=True)
    g = grid

    def init():
        im.set_array(g)
        return [im]

    def update(frame):
        global g
        g = step(g)
        im.set_array(g)
        return [im]

    ani = FuncAnimation(fig, update, init_func=init, interval=0, blit=True)

    plt.show()


def step(grid: np.ndarray):
    u = ushift(grid)
    d = dshift(grid)
    ul = lshift(u)
    ur = rshift(u)
    dl = lshift(d)
    dr = rshift(d)
    l = lshift(grid)
    r = rshift(grid)
    n = u + d + l + r + ul + ur + dl + dr
    b = np.bitwise_or(grid, n)
    return np.array(b == 3, dtype=np.uint8)


def main():
    # grid = makeInitgrid(GRID_SIZE)
    # grid[0, 0] = 1
    # grid[0, 1] = 1
    # grid[0, 2] = 1
    # grid[1, 2] = 1
    # grid[2, 1] = 1
    grid = randomize(GRID_SIZE)
    animate(grid)


if __name__ == "__main__":
    main()
