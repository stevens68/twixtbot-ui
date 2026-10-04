#! /usr/bin/env python

"""Run matched self-play games with TensorFlow and PyTorch backends.

Each PyTorch game uses the same random seed as its TensorFlow counterpart.
With deterministic MCTS settings this makes the games directly comparable:
winner, move count, and complete move sequence are reported.
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

# Allow this script to be run directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import constants as ct
from src.backend import torchnneval, twixt
from src.backend.nnmplayer import Player


class _NullWindow:
    def write_event_value(self, *_args):
        pass


NULL_WINDOW = _NullWindow()


def play_game(backend, model, seed, trials, allow_swap, cpuct, level, evaluator):
    random.seed(seed)

    player = Player(
        model=model,
        evaluator=evaluator,
        trials=trials,
        temperature=0.0,
        rotation=ct.ROT_OFF,
        smart_root=0,
        allow_swap=allow_swap,
        add_noise=0.0,
        cpuct=cpuct,
        level=level,
    )

    game = twixt.Game(allow_scl=False)

    while True:
        response = player.pick_move(game, window=NULL_WINDOW)
        moves = response.get("moves", [])
        if not moves:
            raise RuntimeError(
                f"{backend}: no move returned at history length "
                f"{len(game.history)}"
            )

        game.play(moves[0])

        if len(game.history) % 10 == 0:
            print(f"           {backend}: {len(game.history)} moves...", flush=True)

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
        "seed": seed,
        "winner": winner,
        "moves": [str(move) for move in game.history],
        "move_count": len(game.history),
    }


def compare_games(args):
    model = str(Path(args.model))
    results = []

    print("Loading TensorFlow model...", flush=True)
    tf_evaluator = torchnneval.create_evaluator("tensorflow", model)
    print("Loading PyTorch model...", flush=True)
    torch_evaluator = torchnneval.create_evaluator("pytorch", model)

    for i in range(args.games):
        print(f"\n=== Game {i + 1}/{args.games} ===", flush=True)
        seed = args.seed + i

        started = time.perf_counter()
        tf_game = play_game(
            "tensorflow",
            model,
            seed,
            args.trials,
            args.allow_swap,
            args.cpuct,
            args.level,
            tf_evaluator,
        )
        tf_seconds = time.perf_counter() - started

        started = time.perf_counter()
        torch_game = play_game(
            "pytorch",
            model,
            seed,
            args.trials,
            args.allow_swap,
            args.cpuct,
            args.level,
            torch_evaluator,
        )
        torch_seconds = time.perf_counter() - started

        same_winner = tf_game["winner"] == torch_game["winner"]
        same_moves = tf_game["moves"] == torch_game["moves"]

        result = {
            "game": i + 1,
            "seed": seed,
            "same_winner": same_winner,
            "same_moves": same_moves,
            "tf": tf_game,
            "torch": torch_game,
            "tf_seconds": tf_seconds,
            "torch_seconds": torch_seconds,
        }
        results.append(result)

        print(
            f"game {i + 1:3d}/{args.games}: "
            f"winner={'same' if same_winner else 'DIFF'} "
            f"moves={'same' if same_moves else 'DIFF'} "
            f"TF={tf_game['move_count']:3d} "
            f"Torch={torch_game['move_count']:3d}"
        )

        if not same_moves:
            first_diff = next(
                (
                    j
                    for j, (tf_move, torch_move)
                    in enumerate(zip(tf_game["moves"], torch_game["moves"]))
                    if tf_move != torch_move
                ),
                min(len(tf_game["moves"]), len(torch_game["moves"])),
            )
            print(
                f"           first move difference: index {first_diff} "
                f"(TF={tf_game['moves'][first_diff] if first_diff < len(tf_game['moves']) else 'END'}, "
                f"Torch={torch_game['moves'][first_diff] if first_diff < len(torch_game['moves']) else 'END'})"
            )

    same_winners = sum(r["same_winner"] for r in results)
    same_moves = sum(r["same_moves"] for r in results)
    tf_lengths = [r["tf"]["move_count"] for r in results]
    torch_lengths = [r["torch"]["move_count"] for r in results]
    tf_time = [r["tf_seconds"] for r in results]
    torch_time = [r["torch_seconds"] for r in results]

    print()
    print("=== Summary ===")
    print(f"games:                  {args.games}")
    print(f"identical winners:      {same_winners}/{args.games}")
    print(f"identical move lists:   {same_moves}/{args.games}")
    print(
        f"TF avg moves:           {statistics.mean(tf_lengths):.2f}"
    )
    print(
        f"Torch avg moves:        {statistics.mean(torch_lengths):.2f}"
    )
    print(
        f"TF total time:          {sum(tf_time):.2f}s"
    )
    print(
        f"Torch total time:       {sum(torch_time):.2f}s"
    )
    print(
        f"Torch/TF time ratio:    "
        f"{sum(torch_time) / sum(tf_time):.3f}x"
    )

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "settings": vars(args),
                    "summary": {
                        "games": args.games,
                        "identical_winners": same_winners,
                        "identical_move_lists": same_moves,
                        "tf_average_moves": statistics.mean(tf_lengths),
                        "torch_average_moves": statistics.mean(torch_lengths),
                        "tf_total_seconds": sum(tf_time),
                        "torch_total_seconds": sum(torch_time),
                    },
                    "games": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"results written to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="number of matched games to run (default: 100)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="MCTS trials per move (default: 100)",
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "model" / "pb"),
        help="TensorFlow SavedModel directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed for the first matched game (default: 1)",
    )
    parser.add_argument(
        "--allow-swap",
        type=int,
        choices=(0, 1),
        default=1,
        help="enable the normal swap rule (default: 1)",
    )
    parser.add_argument("--cpuct", type=float, default=1.0)
    parser.add_argument("--level", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default="tf_torch_games.json",
        help="JSON results file (default: tf_torch_games.json)",
    )
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.trials < 0:
        parser.error("--trials must be non-negative")

    compare_games(args)


if __name__ == "__main__":
    main()
