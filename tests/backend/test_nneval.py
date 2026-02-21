import unittest
from src.backend.nneval import NNEvaluater

class DummyModel:
    pass

class TestNNEvaluater(unittest.TestCase):
    def test_init(self):
        # Should raise error because model directory and TensorFlow are not available
        with self.assertRaises(Exception):
            NNEvaluater('nonexistent_model_dir')

if __name__ == '__main__':
    unittest.main()

