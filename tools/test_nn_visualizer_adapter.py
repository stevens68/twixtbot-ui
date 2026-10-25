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
            game.play(move)
        net = NetInputs(game)
        pegs, links, locations = net.to_input_arrays()

        np.testing.assert_array_equal(np.asarray(result["tensors"]["naf"]), net.naf)
        np.testing.assert_array_equal(np.asarray(result["tensors"]["pegs"]), pegs)
        np.testing.assert_array_equal(np.asarray(result["tensors"]["links"]), links)
        np.testing.assert_array_equal(np.asarray(result["tensors"]["locations"]), locations)

    def test_snapshot_reports_requested_move_sequence_and_turn(self):
        result = snapshot(DEFAULT_MOVES)
        self.assertEqual(result["moves"], DEFAULT_MOVES)
        self.assertEqual(result["next_player"], twixt.Game.BLACK)
        self.assertEqual(result["shapes"]["naf"], [24, 24, 11])
        self.assertEqual(result["shapes"]["pegs"], [24, 24, 2])
        self.assertEqual(result["shapes"]["links"], [24, 24, 8])
        self.assertEqual(result["shapes"]["locations"], [24, 24, 2])


if __name__ == "__main__":
    unittest.main()
