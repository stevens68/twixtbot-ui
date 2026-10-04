#! /usr/bin/env python

"""Export deterministic TensorFlow reference evaluations for PyTorch parity tests."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow.compat.v1 as tf  # noqa: E402

from src.backend.naf import NetInputs  # noqa: E402
from src.backend.nneval import NNEvaluater  # noqa: E402
from src.backend.point import Point  # noqa: E402
from src.backend.twixt import Game, SWAP  # noqa: E402


def load_t1_moves(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return [line for line in lines[13:] if line and not line.startswith("#")]


def build_games(t1_path, count):
    game = Game(allow_scl=True)
    games = [game.clone()]
    for raw_move in load_t1_moves(t1_path):
        move = SWAP if raw_move.lower() == SWAP else Point(raw_move)
        game.play(move)
        games.append(game.clone())
        if len(games) >= count:
            break
    return games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/pb")
    parser.add_argument("--game", default="games/full_board.T1")
    parser.add_argument("--output", default="reference/tf_reference.npz")
    parser.add_argument("--positions", type=int, default=32)
    args = parser.parse_args()

    games = build_games(args.game, args.positions)
    evaluator = NNEvaluater(args.model)

    pegs, links, locs, pwin, movelogits = [], [], [], [], []
    for game in games:
        nip = NetInputs(game)
        p, m = evaluator.eval_one(nip)
        pwin.append(np.asarray(p))
        movelogits.append(np.asarray(m))
        p_arr, l_arr, loc_arr = nip.to_input_arrays(evaluator.use_recents)
        pegs.append(p_arr)
        links.append(l_arr)
        locs.append(loc_arr)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        args.output,
        pegs=np.asarray(pegs),
        links=np.asarray(links),
        locs=np.asarray(locs),
        pwin=np.asarray(pwin),
        movelogits=np.asarray(movelogits),
    )

    print(f"wrote {len(games)} reference positions to {args.output}")
    print(f"  pegs:       {np.asarray(pegs).shape}")
    print(f"  links:      {np.asarray(links).shape}")
    print(f"  locs:       {np.asarray(locs).shape}")
    print(f"  pwin:       {np.asarray(pwin).shape}")
    print(f"  movelogits: {np.asarray(movelogits).shape}")


if __name__ == "__main__":
    tf.disable_v2_behavior()
    main()
