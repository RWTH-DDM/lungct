
import lungct.neighbourhood as neighbourhood
import numpy as np


def flood_fill(data, start, visited=None):

    shape = data.shape

    # initialize target
    if visited is None:
        visited = np.zeros(shape, bool)

    # define neighbourhood function
    # if possible use a version optimized for some specific shape
    neighbours = neighbourhood.get_neighbours
    if len(shape) == 3:
        neighbours = neighbourhood.get_neighbours_3d

    to_visit = {start}
    while to_visit:

        current = to_visit.pop()
        visited[current] = True

        # add neighbors to queue
        for neighbour in neighbours(shape, current):

            if visited[neighbour]:
                continue

            # check if neighbour is truthy
            if data[neighbour]:
                to_visit.add(neighbour)

    return visited
