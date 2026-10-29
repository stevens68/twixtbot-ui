"""Parity tests for the Python-backed NN visualizer adapter."""
from __future__ import annotations

import unittest

import numpy as np

from tools.nn_visualizer_adapter import DEFAULT_MOVES, snapshot
from src.backend import twixt
from src.backend.naf import NetInputs


class VisualizerAdapterTests(unittest.TestCase):
    def test_serialized_tensors_match_backend_outputs(self):
        result = snapshot(DEFAULT_MOVES)
        game = twixt.Game(allow_scl=False)
        for move in DEFAULT_MOVES:
            game.play(move, check_draw=False)
        net = NetInputs(game)
        pegs, links, locations = net.to_input_arrays()

        # The API intentionally returns only tensors consumed by the browser.
        self.assertEqual(
            set(result["tensors"]),
            {"pegs", "links", "locations"},
        )
        self.assertNotIn("naf", result["tensors"])
        np.testing.assert_array_equal(np.asarray(result["tensors"]["pegs"]), pegs)
        np.testing.assert_array_equal(np.asarray(result["tensors"]["links"]), links)
        np.testing.assert_array_equal(
            np.asarray(result["tensors"]["locations"]), locations
        )

    def test_snapshot_reports_moves_turn_and_tensor_shapes(self):
        result = snapshot(DEFAULT_MOVES)

        self.assertEqual(result["moves"], DEFAULT_MOVES)
        self.assertEqual(result["next_player"], twixt.Game.BLACK)
        self.assertEqual(
            result["shapes"],
            {
                "pegs": [24, 24, 2],
                "links": [24, 24, 8],
                "locations": [24, 24, 2],
            },
        )
        self.assertNotIn("naf", result["shapes"])

    def test_snapshot_reports_backend_source_and_board(self):
        result = snapshot(DEFAULT_MOVES)

        self.assertEqual(
            result["source"]["transform"],
            "NetInputs(game) -> NetInputs.to_input_arrays()",
        )
        self.assertIn("pegs", result["board"])
        self.assertIn("links", result["board"])
        self.assertGreater(len(result["board"]["pegs"]), 0)


if __name__ == "__main__":
    unittest.main()
