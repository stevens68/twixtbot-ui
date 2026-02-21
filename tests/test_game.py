import unittest
import numpy as np
from src.backend.twixt import Game


class TestGame(unittest.TestCase):
    def setUp(self):
        self.game = Game(allow_scl=False)

    def test_init(self):
        self.assertEqual(self.game.SIZE, 24)
        self.assertEqual(self.game.turn, Game.WHITE)
        self.assertEqual(len(self.game.pegs), 2)
        self.assertEqual(len(self.game.links), 8)
        self.assertIsInstance(self.game.history, list)
        self.assertIsInstance(self.game.open_pegs, list)
        self.assertIsInstance(self.game.reachable, list)
        self.assertIsInstance(self.game.reachable_history, list)

    def test_flip_turn(self):
        orig_turn = self.game.turn
        self.game._flip_turn()
        self.assertNotEqual(self.game.turn, orig_turn)
        self.game._flip_turn()
        self.assertEqual(self.game.turn, orig_turn)

    def test_clone(self):
        self.game.history.append("move")
        clone = self.game.clone()
        self.assertEqual(clone.history, self.game.history)
        self.assertTrue(np.array_equal(clone.pegs, self.game.pegs))
        self.assertTrue(np.array_equal(clone.links, self.game.links))
        self.assertEqual(clone.turn, self.game.turn)
        self.assertEqual(len(clone.open_pegs), 2)
        self.assertEqual(len(clone.reachable), 2)
        self.assertEqual(clone.reachable_history, self.game.reachable_history)


if __name__ == "__main__":
    unittest.main()
