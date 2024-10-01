import numpy as np
from numpy.typing import NDArray


# Will take a binary grid of size (a,b) and
# return a grid of size (a-2, b-2) with the
# next step computation with Game of Life rules
def nextStep(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
    n = (
        grid[:-2, :-2]
        + grid[:-2, 1:-1]
        + grid[:-2, 2:]
        + grid[1:-1, :-2]
        + grid[1:-1, 2:]
        + grid[2:, :-2]
        + grid[2:, 1:-1]
        + grid[2:, 2:]
    )
    return np.array(np.bitwise_or(n, grid[1:-1, 1:-1]) == 3, dtype=np.uint8)


def constIndex(grid: NDArray[np.uint8]) -> np.uint32:
    ind = np.uint32(0)
    for v in grid.flatten():
        ind = ind << 1 + v
    return ind


def knownIndex(grid: NDArray[np.uint8]) -> np.uint32:
    ind = np.uint32(0)
    for i, v in enumerate(grid.flatten()):
        if i == 4:
            continue
        ind = ind << 1 + v
    return ind


def unknownIndex(grid: NDArray[np.uint8]) -> np.uint32:
    ind = np.uint32(0)
    for i, v in enumerate(grid.flatten()):
        if i == 4:
            continue
        ind = (ind * 3) + v
    return ind


def getKnownIndex(curr: NDArray, prev: NDArray) -> np.uint32:
    a = constIndex(curr)
    b = knownIndex(prev)
    return b << 9 + a


def getUnknownIndex(curr: NDArray, prev: NDArray) -> np.uint32:
    a = constIndex(curr)
    b = unknownIndex(prev)
    return b << 9 + a
