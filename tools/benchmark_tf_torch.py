#! /usr/bin/env python

"""Compact TF/PyTorch backend benchmark with matched greedy games."""

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import constants as ct
from src.backend import swapmodel, torchnneval, twixt
from src.backend.nnmplayer import Player


class TimedEvaluator:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.calls = 0
        self.seconds = 0.0
        self.use_recents = evaluator.use_recents

    def eval_one(self, nip):
        started = time.perf_counter()
        result = self.evaluator.eval_one(nip)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        return result


def make_player(model, evaluator, trials):
    return Player(
        model=model,
        evaluator=evaluator,
        trials=trials,
        temperature=0.0,
        rotation=ct.ROT_OFF,
        smart_root=0,
        allow_swap=1,
        add_noise=0.0,
        cpuct=1.0,
        level=1.0,
    )


def play_game(player, seed, opening_move):
    player.reset()
    random.seed(seed)
    game = twixt.Game(allow_scl=False)
    game.play(opening_move)

    started = time.perf_counter()
    while True:
        response = player.pick_move(game, window=None)
        moves = response.get("moves", [])
        if not moves:
            raise RuntimeError("no move returned")
        game.play(moves[0])

        if game.result == twixt.DRAW:
            winner = None
            break
        if game.is_winning(twixt.Game.WHITE):
            winner = twixt.Game.WHITE
            break
        if game.is_winning(twixt.Game.BLACK):
            winner = twixt.Game.BLACK
            break

    return {
        "seconds": time.perf_counter() - started,
        "winner": winner,
        "moves": [str(move) for move in game.history],
        "move_count": len(game.history),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "model" / "pb"),
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    model = str(Path(args.model))
    print("Loading TensorFlow model...", flush=True)
    tf_eval = TimedEvaluator(torchnneval.create_evaluator("tensorflow", model))
    print("Loading PyTorch model...", flush=True)
    torch_eval = TimedEvaluator(torchnneval.create_evaluator("pytorch", model))

    print("Creating players...", flush=True)
    tf_player = make_player(model, tf_eval, args.trials)
    torch_player = make_player(model, torch_eval, args.trials)

    tf_times = []
    torch_times = []
    tf_moves = []
    torch_moves = []
    identical = 0

    for i in range(args.games):
        seed = args.seed + i
        random.seed(seed)
        opening = swapmodel.choose_first_move()

        tf_eval.calls = tf_eval.seconds = 0.0
        torch_eval.calls = torch_eval.seconds = 0.0

        tf_game = play_game(tf_player, seed, opening)
        torch_game = play_game(torch_player, seed, opening)

        same = (
            tf_game["winner"] == torch_game["winner"]
            and tf_game["moves"] == torch_game["moves"]
        )
        identical += int(same)
        tf_times.append(tf_game["seconds"])
        torch_times.append(torch_game["seconds"])
        tf_moves.append(tf_game["move_count"])
        torch_moves.append(torch_game["move_count"])

        print(
            f"game {i + 1:3d}/{args.games}: "
            f"{'same' if same else 'DIFF':4s} "
            f"moves={tf_game['move_count']:3d} "
            f"TF={tf_game['seconds']:.3f}s "
            f"Torch={torch_game['seconds']:.3f}s "
            f"TF-NN={tf_eval.seconds:.3f}s/{tf_eval.calls} "
            f"Torch-NN={torch_eval.seconds:.3f}s/{torch_eval.calls}",
            flush=True,
        )

    tf_total = sum(tf_times)
    torch_total = sum(torch_times)
    tf_nn = tf_eval.seconds
    torch_nn = torch_eval.seconds
    tf_calls = tf_eval.calls
    torch_calls = torch_eval.calls

    print()
    print("=== Benchmark Summary ===")
    print(f"games:                  {args.games}")
    print(f"identical games:        {identical}/{args.games}")
    print(f"TF avg moves:           {statistics.mean(tf_moves):.2f}")
    print(f"Torch avg moves:        {statistics.mean(torch_moves):.2f}")
    print(f"TF total time:          {tf_total:.2f}s")
    print(f"Torch total time:       {torch_total:.2f}s")
    print(f"Torch/TF total ratio:   {torch_total / tf_total:.3f}x")
    print(f"TF NN evaluations:      {tf_calls}")
    print(f"Torch NN evaluations:   {torch_calls}")
    print(f"TF NN time:             {tf_nn:.2f}s")
    print(f"Torch NN time:          {torch_nn:.2f}s")
    print(f"Torch/TF NN ratio:      {torch_nn / tf_nn:.3f}x")
    print(f"TF NN avg:              {tf_nn / tf_calls * 1000:.3f} ms")
    print(f"Torch NN avg:           {torch_nn / torch_calls * 1000:.3f} ms")


if __name__ == "__main__":
    main()
