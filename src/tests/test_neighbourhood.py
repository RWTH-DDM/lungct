
import unittest
import lungct.neighbourhood as nh


class TestNeighbourhood(unittest.TestCase):

    def test_1d(self):

        result = nh.get_neighbours((3,), (1,))

        expected = [
            (0,),
            (2,)
        ]

        self.assertTrue(sorted(result) == sorted(expected))

    def test_2d(self):

        result = nh.get_neighbours((3, 3), (1, 1))

        expected = [
            (0, 1),
            (2, 1),
            (1, 0),
            (1, 2)
        ]

        self.assertTrue(sorted(result) == sorted(expected))

    def test_3d(self):

        result = nh.get_neighbours((3, 3, 3), (1, 1, 1))

        expected = [
            (0, 1, 1),
            (2, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (1, 1, 0),
            (1, 1, 2)
        ]

        self.assertTrue(sorted(result) == sorted(expected))

    def test_edge_behaviour(self):

        result = nh.get_neighbours((3, 3, 3), (0, 1, 1))

        expected = [
            (1, 1, 1),
            (0, 0, 1),
            (0, 2, 1),
            (0, 1, 0),
            (0, 1, 2)
        ]

        self.assertTrue(sorted(result) == sorted(expected))

    def test_corner_behaviour(self):

        result = nh.get_neighbours((3, 3, 3), (0, 0, 0))

        expected = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ]

        self.assertTrue(sorted(result) == sorted(expected))

    def test_return_type(self):

        result = nh.get_neighbours((2, 2), (0, 0))

        self.assertTrue(type(list(result).pop()) == tuple)
