
from lungct.neighbourhood import get_neighbours
import numpy as np


def flood_fill(data, start):

    shape = data.shape
    visited = np.zeros(shape, bool)

    to_visit = {start}

    while to_visit:

        current = to_visit.pop()
        visited[current] = True

        # add neighbors to queue
        for neighbour in get_neighbours(shape, current):

            if visited[neighbour]:
                continue

            # check if neighbour is truthy
            if data[neighbour]:
                to_visit.add(neighbour)

    return visited
