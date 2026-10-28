#!/usr/bin/env python3
"""Source-of-truth adapter for the Three.js NN visualizer.

All game rules and input transforms are delegated to src.backend.twixt and
src.backend.naf. This module only serializes their results for the browser.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from src.backend import twixt
from src.backend.naf import NetInputs


DEFAULT_MOVES = [
    "L12", "L8", "O9", "L17", "H16", "I14", "J13",
    "N9", "Q10", "K6", "H12", "O11", "F7",
]


def _tensor(array: np.ndarray) -> list:
    return np.asarray(array).tolist()


def snapshot(moves: list[str], allow_scl: bool = False, allow_swap: bool = True) -> dict[str, Any]:
    snapshot_started = time.perf_counter()
    game = twixt.Game(allow_scl=allow_scl)
    normalized_moves = [str(move).strip() for move in moves]
    lowered_moves = [move.lower() for move in normalized_moves]
    resign_indices = [i for i, move in enumerate(lowered_moves) if move == twixt.RESIGN]
    if resign_indices and resign_indices != [len(normalized_moves) - 1]:
        raise ValueError("'resign' is only valid as the final move in a trajectory.")
    if lowered_moves.count(twixt.SWAP) > 1:
        raise ValueError("A trajectory can contain at most one swap move.")
    if twixt.SWAP in lowered_moves and lowered_moves.index(twixt.SWAP) != 1:
        raise ValueError("'swap' is only valid as the second move in a trajectory.")

    terminal_resign = bool(lowered_moves and lowered_moves[-1] == twixt.RESIGN)
    playable_moves = normalized_moves[:-1] if terminal_resign else normalized_moves

    replay_started = time.perf_counter()
    move_times_ms = []
    for move in playable_moves:
        move_started = time.perf_counter()
        game.play(twixt.SWAP if move.lower() == twixt.SWAP else move)
        move_times_ms.append((time.perf_counter() - move_started) * 1000)
    replay_ms = (time.perf_counter() - replay_started) * 1000

    net_started = time.perf_counter()
    net = NetInputs(game)
    pegs, links, locations = net.to_input_arrays()
    net_inputs_ms = (time.perf_counter() - net_started) * 1000

    board_started = time.perf_counter()
    board_pegs = []
    for player in (twixt.Game.BLACK, twixt.Game.WHITE):
        rows, cols = np.nonzero(game.pegs[player])
        board_pegs.extend({
            "row": int(y), "col": int(x), "player": int(player),
            "coordinate": f"{chr(65 + int(x))}{int(y) + 1}",
        } for x, y in zip(rows, cols))

    board_links = []
    for channel, plane in enumerate(game.links):
        xs, ys = np.nonzero(plane)
        for x, y in zip(xs, ys):
            desc = twixt.Game.describe_link(channel, int(x), int(y))
            a, b = desc.p1, desc.p2
            board_links.append({
                "player": int(desc.owner),
                "from": {"row": int(a.y), "col": int(a.x)},
                "to": {"row": int(b.y), "col": int(b.x)},
            })
    board_ms = (time.perf_counter() - board_started) * 1000

    tensors_started = time.perf_counter()
    tensors = {
        "naf": _tensor(net.naf),
        "pegs": _tensor(pegs),
        "links": _tensor(links),
        "locations": _tensor(locations),
    }
    tensor_conversion_ms = (time.perf_counter() - tensors_started) * 1000

    total_ms = (time.perf_counter() - snapshot_started) * 1000
    slowest_moves = sorted(enumerate(move_times_ms, start=1), key=lambda item: item[1], reverse=True)[:3]
    slowest_summary = ",".join(f"{index}:{duration:.1f}ms" for index, duration in slowest_moves) or "none"
    print(
        "[NN perf adapter] "
        f"moves={len(moves)} replay_ms={replay_ms:.1f} "
        f"slowest_move_ms={slowest_summary} net_inputs_ms={net_inputs_ms:.1f} "
        f"board_extract_ms={board_ms:.1f} tensor_conversion_ms={tensor_conversion_ms:.1f} "
        f"snapshot_total_ms={total_ms:.1f}",
        flush=True,
    )

    return {
        "source": {
            "game_class": f"{twixt.Game.__module__}.{twixt.Game.__name__}",
            "net_inputs_class": f"{NetInputs.__module__}.{NetInputs.__name__}",
            "transform": "NetInputs(game) -> NetInputs.to_input_arrays()",
        },
        "moves": normalized_moves,
        "result": "resign" if terminal_resign else game.result,
        "next_player": int(game.turn),
        "next_player_name": game.COLOR_NAME[game.turn],
        "board": {"pegs": board_pegs, "links": board_links},
        "tensors": tensors,
        "shapes": {
            "naf": list(net.naf.shape),
            "pegs": list(pegs.shape),
            "links": list(links.shape),
            "locations": list(locations.shape),
        },
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moves", nargs="*", default=DEFAULT_MOVES)
    parser.add_argument("--allow-scl", action="store_true")
    parser.add_argument("--no-allow-swap", action="store_true")
    args = parser.parse_args()
    print(json.dumps(snapshot(args.moves, args.allow_scl, not args.no_allow_swap), indent=2))


if __name__ == "__main__":
    main()
