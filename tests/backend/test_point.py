import unittest
from src.backend.point import Point

class TestPoint(unittest.TestCase):
    def test_init_xy(self):
        p = Point(3, 5)
        self.assertEqual(p.x, 3)
        self.assertEqual(p.y, 5)

    def test_init_str_lower(self):
        p = Point('d6')
        self.assertEqual(p.x, 3)
        self.assertEqual(p.y, 5)
        self.assertEqual(str(p), 'd6')

    def test_init_str_upper(self):
        p = Point('D6')
        self.assertEqual(p.x, 3)
        self.assertEqual(p.y, 5)
        self.assertEqual(str(p), 'd6')

    def test_init_tuple(self):
        p = Point((2, 7))
        self.assertEqual(p.x, 2)
        self.assertEqual(p.y, 7)

    def test_add(self):
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        self.assertEqual(p1 + p2, Point(4, 6))
        self.assertEqual(p1 + (3, 4), Point(4, 6))
        self.assertEqual((3, 4) + p1, Point(4, 6))

    def test_sub(self):
        p1 = Point(5, 7)
        p2 = Point(2, 3)
        self.assertEqual(p1 - p2, Point(3, 4))
        self.assertEqual(p1 - (2, 3), Point(3, 4))
        self.assertEqual((8, 10) - p1, Point(3, 3))

    def test_mul(self):
        p = Point(2, 3)
        self.assertEqual(p * 3, Point(6, 9))
        self.assertEqual(3 * p, Point(6, 9))

    def test_str_and_repr(self):
        p = Point(1, 9)
        self.assertEqual(str(p), 'b10')
        self.assertEqual(repr(p), 'b10')

    def test_flip(self):
        p = Point(2, 5)
        self.assertEqual(p.flip(), Point(5, 2))

    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            Point('invalid')
        with self.assertRaises(ValueError):
            Point(1, 2, 3)
        with self.assertRaises(ValueError):
            Point([1])

if __name__ == '__main__':
    unittest.main()

