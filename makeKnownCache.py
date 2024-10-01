import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from utils import getKnownIndex, getUnknownIndex, nextStep


def grids():
    grid = np.zeros([25], dtype=np.uint8)
    yield grid
    while np.sum(grid == 1) != 25:
        i = 0
        while grid[i] != 0:
            grid[i] = 0
            i += 1
        grid[i] = 1
        yield grid


def getRandom(knownGrid: NDArray):
    randGrid = knownGrid.copy()
    while np.sum(randGrid == 2) != 9:
        i = 0
        while randGrid[i] == 2:
            randGrid[i] = knownGrid[i]
            i += 1
        randGrid[i] = 2
        if randGrid[4] == 2:
            continue
        yield randGrid


tots = np.zeros([(3**8) << 9], dtype=np.uint)
trus = np.zeros([(3**8) << 9], dtype=np.uint)

totals = np.zeros([1 << 17], dtype=np.uint)
sets = np.zeros([1 << 17], dtype=np.uint)

try:
    for grid in tqdm(grids(), total=1 << 25, desc="Grids", position=1):
        regrid = grid.reshape((5, 5))
        curr = nextStep(regrid)
        prev = regrid[1:-1, 1:-1]
        ind = getKnownIndex(curr, prev)
        for randgrid in getRandom(prev.flatten()):
            unind = getUnknownIndex(curr, randgrid)
            tots[unind] += 1
            if regrid[1, 1] == 1:
                trus[unind] += 1
        totals[ind] += 1
        if regrid[1, 1] == 1:
            sets[ind] += 1
except KeyboardInterrupt:
    print(np.sum(totals))
    print(np.sum(sets))
finally:
    known_exp = sets * 1.0 / totals
    unknown_exp = trus * 1.0 / tots
    print("Known Exp", known_exp)
    print("Unknown Exp", unknown_exp)
    print("Known On", np.sum(known_exp == 1.0) * 100.0 / (1 << 17))
    print("Known Off", np.sum(known_exp == 0.0) * 100.0 / (1 << 17))
    print("Unknown On", np.sum(unknown_exp == 1.0) * 100.0 / ((3**8) << 9))
    print("Unknown Off", np.sum(unknown_exp == 0.0) * 100.0 / ((3**8) << 9))
    np.savez_compressed(
        "cached_data",
        known_totals=totals,
        unknown_totals=tots,
        known_trues=sets,
        unknown_trues=trus,
        known_expectation=known_exp,
        unknown_expectation=unknown_exp,
    )
