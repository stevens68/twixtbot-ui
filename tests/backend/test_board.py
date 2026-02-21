import unittest
from src.backend.board import TBWHistory, TwixtBoard

class DummyStgs:
    def get(self, key):
        return 120

class TestTBWHistory(unittest.TestCase):
    def test_init(self):
        h = TBWHistory('move')
        self.assertEqual(h.move, 'move')
        self.assertEqual(h.objects, [])

class TestTwixtBoard(unittest.TestCase):
    def setUp(self):
        self.stgs = DummyStgs()
        self.board = TwixtBoard(self.stgs)

    def test_init(self):
        self.assertEqual(self.board.size, 24)
        self.assertEqual(self.board.history, [])
        self.assertEqual(self.board.known_moves, set())
        self.assertEqual(self.board.visit_offset, 1.0)
        self.assertEqual(self.board.cell_width, 40)
        self.assertEqual(self.board.peg_radius, 10)
        self.assertIsNone(self.board.graph)

    def test_point_to_coords(self):
        coords = self.board._point_to_coords((1, 2))
        self.assertIsInstance(coords, tuple)
        self.assertEqual(len(coords), 2)

    def test_move_to_point(self):
        point = self.board._move_to_point('b3')
        self.assertEqual(point.x, 1)
        self.assertEqual(point.y, 2)

if __name__ == '__main__':
    unittest.main()

