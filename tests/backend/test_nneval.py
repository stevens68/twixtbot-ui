import unittest
from src.backend.nneval import NNEvaluater


class DummyModel:
    pass


class TestNNEvaluater(unittest.TestCase):
    def test_init(self):
        # Should raise an error because the model checkpoint does not exist
        with self.assertRaises(Exception):
            NNEvaluater("nonexistent_model_dir")


if __name__ == "__main__":
    unittest.main()
