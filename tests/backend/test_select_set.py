import unittest
from src.backend.select_set import SelectSet

class TestSelectSet(unittest.TestCase):
    def setUp(self):
        self.ss = SelectSet()

    def test_add_and_contains(self):
        self.ss.add('a')
        self.ss.add('b')
        self.assertIn('a', self.ss)
        self.assertIn('b', self.ss)
        self.assertEqual(len(self.ss), 2)
        with self.assertRaises(ValueError):
            self.ss.add('a')  # duplicate

    def test_remove(self):
        self.ss.add('x')
        self.ss.add('y')
        self.ss.remove('x')
        self.assertNotIn('x', self.ss)
        self.assertIn('y', self.ss)
        self.ss.remove('y')
        self.assertEqual(len(self.ss), 0)

    def test_clone(self):
        self.ss.add('foo')
        clone = self.ss.clone()
        self.assertIn('foo', clone)
        self.assertEqual(len(clone), 1)
        clone.add('bar')
        self.assertIn('bar', clone)
        self.assertNotIn('bar', self.ss)

    def test_getitem(self):
        self.ss.add('a')
        self.ss.add('b')
        self.assertEqual(self.ss[0], 'a')
        self.assertEqual(self.ss[1], 'b')

    def test_pick(self):
        import random
        self.ss.add('a')
        self.ss.add('b')
        picked = self.ss.pick(random)
        self.assertIn(picked, ['a', 'b'])

if __name__ == '__main__':
    unittest.main()

