import unittest
from src.heatmap import Heatmap

class DummyGame:
    SIZE = 24
    pass

class DummyNM:
    def eval_game(self, game, maxbest=None):
        return 0.0, [], [], []

class DummyBot:
    def __init__(self):
        self.nm = DummyNM()

class TestHeatmap(unittest.TestCase):
    def test_init_missing_args(self):
        with self.assertRaises(ValueError):
            Heatmap()
        with self.assertRaises(ValueError):
            Heatmap(game=DummyGame())
        with self.assertRaises(ValueError):
            Heatmap(bot=DummyBot())

    def test_init_with_args(self):
        h = Heatmap(game=DummyGame(), bot=DummyBot())
        self.assertIsInstance(h, Heatmap)
        self.assertIs(h.game, h.game)
        self.assertIs(h.bot, h.bot)
        self.assertIsInstance(h.p_values, dict)
        self.assertIsInstance(h.rgb_colors, dict)
        self.assertIsInstance(h.policy_moves, list)

if __name__ == '__main__':
    unittest.main()
