import unittest
import numpy as np
from src.backend.nnmcts import EvalNode


class DummyGame:
    class SIZE:
        pass

    SIZE = 24


class TestEvalNode(unittest.TestCase):
    def setUp(self):
        self.node = EvalNode()

    def test_init(self):
        k = 24 * (24 - 2)
        self.assertEqual(self.node.N.shape, (k,))
        self.assertEqual(self.node.Q.shape, (k,))
        self.assertEqual(self.node.P.shape, (k,))
        self.assertFalse(self.node.proven)
        self.assertIsNone(self.node.score)
        self.assertIsNone(self.node.winning_move)
        self.assertIsNone(self.node.drawing_move)
        self.assertEqual(len(self.node.subnodes), k)
        self.assertIsNone(self.node.LM)
        self.assertIsNone(self.node.LMnz)


if __name__ == "__main__":
    unittest.main()
