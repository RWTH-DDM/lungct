
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


def get_neighbours_3d(shape, coordinates):

    coordinates = list(coordinates)

    for dim in range(0, 3):

        if coordinates[dim] > 0:
            coordinates[dim] -= 1
            yield tuple(coordinates)
            coordinates[dim] += 1

        if coordinates[dim] < (shape[dim] - 1):
            coordinates[dim] += 1
            yield tuple(coordinates)
            coordinates[dim] -= 1
