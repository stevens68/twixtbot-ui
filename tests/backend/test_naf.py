import unittest
import numpy as np
from src.backend.naf import NetInputs
from src.backend.point import Point

class DummyGame:
    SIZE = 24
    WHITE = 0
    BLACK = 1
    def __init__(self):
        self.turn = self.WHITE
        self.links = [np.zeros((self.SIZE, self.SIZE), dtype=np.uint8) for _ in range(8)]
        self.pegs = [np.zeros((self.SIZE, self.SIZE), dtype=np.uint8) for _ in range(2)]
        self.history = []

class TestNetInputs(unittest.TestCase):
    def setUp(self):
        self.game = DummyGame()
        self.game.history = [Point(1, 2), Point(3, 4), Point(5, 6), Point(7, 8)]

    def test_init(self):
        ni = NetInputs(self.game)
        self.assertEqual(ni.naf.shape, (self.game.SIZE, self.game.SIZE, 11))
        self.assertIsInstance(ni.recents, list)

    def test_hflip(self):
        ni = NetInputs(self.game)
        rec_before = list(ni.recents)
        ni.hflip()
        self.assertEqual(len(ni.recents), len(rec_before))
        for p, pflip in zip(rec_before, ni.recents):
            self.assertEqual(pflip.x, self.game.SIZE - 1 - p.x)
            self.assertEqual(pflip.y, p.y)

    def test_vflip(self):
        ni = NetInputs(self.game)
        rec_before = list(ni.recents)
        ni.vflip()
        self.assertEqual(len(ni.recents), len(rec_before))
        for p, pflip in zip(rec_before, ni.recents):
            self.assertEqual(pflip.x, p.x)
            self.assertEqual(pflip.y, self.game.SIZE - 1 - p.y)

    def test_to_input_arrays(self):
        ni = NetInputs(self.game)
        pegs, links, locs = ni.to_input_arrays(use_recents=True)
        self.assertEqual(pegs.shape, (self.game.SIZE, self.game.SIZE, 2))
        self.assertEqual(links.shape, (self.game.SIZE, self.game.SIZE, 8))
        self.assertEqual(locs.shape, (self.game.SIZE, self.game.SIZE, 3))

if __name__ == '__main__':
    unittest.main()

