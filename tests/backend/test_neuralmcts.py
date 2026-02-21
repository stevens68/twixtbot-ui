import unittest
import numpy as np
from src.backend.nnmcts import NeuralMCTS, EvalNode


class DummyGame:
    SIZE = 24
    WHITE = 1
    BLACK = 0

    def __init__(self):
        self.turn = self.WHITE
        self.links = [
            np.zeros((self.SIZE, self.SIZE), dtype=np.uint8) for _ in range(8)
        ]
        self.pegs = [np.zeros((self.SIZE, self.SIZE), dtype=np.uint8) for _ in range(2)]
        self.history = []

    def just_won(self):
        return False


def dummy_sap(game):
    # Returns poseval, movelogits
    return 0.5, np.zeros(DummyGame.SIZE * (DummyGame.SIZE - 2))


class TestNeuralMCTS(unittest.TestCase):
    def setUp(self):
        self.mcts = NeuralMCTS(dummy_sap)
        self.game = DummyGame()

    def test_init(self):
        self.assertIsNone(self.mcts.root)
        self.assertIsNone(self.mcts.history_at_root)
        self.assertFalse(self.mcts.proven)
        self.assertIsNone(self.mcts.score)
        self.assertIsNone(self.mcts.drawing_move)
        self.assertIsNone(self.mcts.report)

    def test_expand_leaf(self):
        leaf = self.mcts.expand_leaf(self.game)
        self.assertIsInstance(leaf, EvalNode)
        self.assertFalse(leaf.proven)
        self.assertIsNotNone(leaf.LM)
        self.assertIsNotNone(leaf.LMnz)
        self.assertEqual(leaf.score, 0.5)

    def test_expand_leaf_just_won(self):
        class WinGame(DummyGame):
            def just_won(self):
                return True

        win_game = WinGame()
        leaf = self.mcts.expand_leaf(win_game)
        self.assertTrue(leaf.proven)
        self.assertEqual(leaf.score, -1)
        self.assertEqual(leaf.LMnz, "just_won")


if __name__ == "__main__":
    unittest.main()
