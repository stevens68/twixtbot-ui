import unittest

import numpy as np
import torch

from src.backend.torchmodel import TwixtNet


class TestTwixtNet(unittest.TestCase):
    def test_scalar_head_shapes(self):
        model = TwixtNet()
        model.eval()

        pegs = torch.zeros((2, 24, 24, 2))
        links = torch.zeros((2, 24, 24, 8))
        locs = torch.zeros((2, 24, 24, 2))

        with torch.no_grad():
            pwin, movelogits = model(pegs, links, locs)

        self.assertEqual(tuple(pwin.shape), (2, 1))
        self.assertEqual(tuple(movelogits.shape), (2, 527))
        self.assertTrue(np.isfinite(pwin.numpy()).all())
        self.assertTrue(np.isfinite(movelogits.numpy()).all())

    def test_recent_move_and_triple_heads(self):
        model = TwixtNet(use_recents=True, value_triple=True)
        model.eval()

        inputs = [
            torch.zeros((1, 24, 24, 2)),
            torch.zeros((1, 24, 24, 8)),
            torch.zeros((1, 24, 24, 3)),
        ]

        with torch.no_grad():
            pwin, movelogits = model(*inputs)

        self.assertEqual(tuple(pwin.shape), (1, 3))
        self.assertEqual(tuple(movelogits.shape), (1, 527))


if __name__ == "__main__":
    unittest.main()
