
def get_neighbours(shape, coordinates):

    coordinates = list(coordinates)

    for idx, _ in enumerate(coordinates):

        if coordinates[idx] > 0:

            neighbour = coordinates.copy()
            neighbour[idx] -= 1

            yield tuple(neighbour)

        if coordinates[idx] < (shape[idx] - 1):

            neighbour = coordinates.copy()
            neighbour[idx] += 1

            yield tuple(neighbour)
