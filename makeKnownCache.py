import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from utils import getKnownIndex, getUnknownIndex, nextStep
from multiprocessing import Pool, cpu_count, shared_memory

shape_known = 1 << 17
shape_unknown = (3**8) << 9
BATCH_SIZE = 1 << 6

shared_known_total_memory = shared_memory.SharedMemory(
    create=True, size=shape_known * np.dtype(np.uint32).itemsize
)

shared_known_total = np.ndarray(
    shape_known, dtype=np.uint32, buffer=shared_known_total_memory.buf
)

shared_known_on_memory = shared_memory.SharedMemory(
    create=True, size=shape_known * np.dtype(np.uint32).itemsize
)

shared_known_on = np.ndarray(
    shape_known, dtype=np.uint32, buffer=shared_known_on_memory.buf
)

shared_unknown_total_memory = shared_memory.SharedMemory(
    create=True, size=shape_unknown * np.dtype(np.uint32).itemsize
)

shared_unknown_total = np.ndarray(
    shape_unknown, dtype=np.uint32, buffer=shared_unknown_total_memory.buf
)

shared_unknown_on_memory = shared_memory.SharedMemory(
    create=True, size=shape_unknown * np.dtype(np.uint32).itemsize
)

shared_unknown_on = np.ndarray(
    shape_unknown, dtype=np.uint32, buffer=shared_unknown_on_memory.buf
)

pbar = tqdm(total=(1 << 25) / BATCH_SIZE, desc="Grids")


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


def batch_grids(n=BATCH_SIZE):
    grids_arr = []
    for grid in grids():
        grids_arr.append(grid)
        if len(grids_arr) == n:
            yield grids_arr
            grids_arr = []


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


def process_grid(grids, shm_names, shapes):
    skom = shared_memory.SharedMemory(name=shm_names[0])
    sko = np.ndarray(shapes[0], dtype=np.uint32, buffer=skom.buf)
    sktm = shared_memory.SharedMemory(name=shm_names[1])
    skt = np.ndarray(shapes[1], dtype=np.uint32, buffer=sktm.buf)
    suom = shared_memory.SharedMemory(name=shm_names[2])
    suo = np.ndarray(shapes[2], dtype=np.uint32, buffer=suom.buf)
    sutm = shared_memory.SharedMemory(name=shm_names[3])
    sut = np.ndarray(shapes[3], dtype=np.uint32, buffer=sutm.buf)
    for grid in grids:
        regrid = grid.reshape((5, 5))
        curr = nextStep(regrid)
        prev = regrid[1:-1, 1:-1]
        ind = getKnownIndex(curr, prev)
        for randgrid in getRandom(prev.flatten()):
            unind = getUnknownIndex(curr, randgrid)
            np.add.at(sut, unind, 1)
            if regrid[1, 1] == 1:
                np.add.at(suo, unind, 1)
        np.add.at(skt, ind, 1)
        if regrid[1, 1] == 1:
            np.add.at(sko, ind, 1)
    skom.close()
    sktm.close()
    suom.close()
    sutm.close()


try:
    print(cpu_count())
    with Pool(cpu_count() - 1) as pool:
        for grid_arr in batch_grids():
            pool.apply_async(
                process_grid,
                args=(
                    grid_arr,
                    (
                        shared_known_on_memory.name,
                        shared_known_total_memory.name,
                        shared_unknown_on_memory.name,
                        shared_unknown_total_memory.name,
                    ),
                    (shape_known, shape_known, shape_unknown, shape_unknown),
                ),
                callback=lambda _: pbar.update(),
            )
except KeyboardInterrupt:
    print("HALTED")
finally:
    known_exp = shared_known_on * 1.0 / shared_known_total
    unknown_exp = shared_unknown_on * 1.0 / shared_unknown_total
    print("Known Exp", known_exp)
    print("Unknown Exp", unknown_exp)
    print("Known On", np.sum(known_exp == 1.0) * 100.0 / (1 << 17))
    print("Known Off", np.sum(known_exp == 0.0) * 100.0 / (1 << 17))
    print("Unknown On", np.sum(unknown_exp == 1.0) * 100.0 / ((3**8) << 9))
    print("Unknown Off", np.sum(unknown_exp == 0.0) * 100.0 / ((3**8) << 9))
    np.savez_compressed(
        "cached_data",
        known_totals=shared_known_total,
        unknown_totals=shared_unknown_total,
        known_trues=shared_known_on,
        unknown_trues=shared_unknown_on,
        known_expectation=known_exp,
        unknown_expectation=unknown_exp,
    )
