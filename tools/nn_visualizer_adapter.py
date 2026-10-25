#!/usr/bin/env python3
"""Source-of-truth adapter for the Three.js NN visualizer.

All game rules and input transforms are delegated to src.backend.twixt and
src.backend.naf. This module only serializes their results for the browser.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
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


def snapshot(moves: list[str], allow_scl: bool = False) -> dict[str, Any]:
    game = twixt.Game(allow_scl=allow_scl)
    for move in moves:
        game.play(move.strip())

    net = NetInputs(game)
    pegs, links, locations = net.to_input_arrays()
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

    return {
        "source": {
            "game_class": f"{twixt.Game.__module__}.{twixt.Game.__name__}",
            "net_inputs_class": f"{NetInputs.__module__}.{NetInputs.__name__}",
            "transform": "NetInputs(game) -> NetInputs.to_input_arrays()",
        },
        "moves": list(moves),
        "next_player": int(game.turn),
        "next_player_name": game.COLOR_NAME[game.turn],
        "board": {"pegs": board_pegs, "links": board_links},
        "tensors": {
            "naf": _tensor(net.naf),
            "pegs": _tensor(pegs),
            "links": _tensor(links),
            "locations": _tensor(locations),
        },
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
    args = parser.parse_args()
    print(json.dumps(snapshot(args.moves, args.allow_scl), indent=2))


if __name__ == "__main__":
    main()
