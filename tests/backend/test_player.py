import unittest
from src.backend.nnmplayer import Player

class TestPlayer(unittest.TestCase):
    def test_init_defaults(self):
        # Should raise because model is required
        with self.assertRaises(Exception):
            Player()

    def test_invalid_temperature(self):
        # Should raise ValueError for unsupported temperature
        with self.assertRaises(ValueError):
            Player(model='dummy', temperature=2.0)

    def test_valid_init_with_model(self):
        class DummyEval:
            def __init__(self, model):
                pass
        class DummyNNEval:
            def eval_one(self, n):
                return [0.5], [0.5]
        # Patch nneval.NNEvaluater to DummyNNEval
        import src.backend.nneval
        old_eval = src.backend.nneval.NNEvaluater
        src.backend.nneval.NNEvaluater = lambda model: DummyNNEval()
        try:
            p = Player(model='dummy', temperature=1.0)
            self.assertEqual(p.temperature, 1.0)
            self.assertEqual(p.model, 'dummy')
            self.assertTrue(hasattr(p, 'nm'))
        finally:
            src.backend.nneval.NNEvaluater = old_eval

if __name__ == '__main__':
    unittest.main()

