#!/usr/bin/env python3
"""Trace Twixt link-channel transformations using the repository's own backend.

Run from the repository root:
    python tools/nn-visualizer/trace_link_inputs.py
    python tools/nn-visualizer/trace_link_inputs.py --last-moves 4

This intentionally calls src.backend.twixt.Game and src.backend.naf.NetInputs
instead of reimplementing the transformation rules in JavaScript.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend import twixt
from src.backend.naf import NetInputs


# Keep this sequence in sync with createSampleGame() in main.js.
SAMPLE_MOVES = [
    "L12", "L8", "O9", "L17", "H16", "I14", "J13",
    "N9", "Q10", "K6", "H12", "O11", "F7",
]

# The bits are the actual constants used by twixt.Game.
def direction_name(channel: int) -> str:
    vertical = bool(channel & twixt.Game.LINK_LONGY)
    diff = bool(channel & twixt.Game.LINK_DIFFSIGN)
    # Directions are described in board coordinates: x increases A -> X,
    # y increases 1 -> 24. The channel bit flags describe the link's slope
    # and whether its x displacement is long.
    if vertical:
        return "SSE" if diff else "NNE"
    return "ESE" if diff else "ENE"


def coord(row: int, col: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def occupied_cells(array: np.ndarray, channel: int) -> list[str]:
    rows, cols = np.nonzero(array[:, :, channel])
    return [coord(int(r), int(c)) for r, c in zip(rows, cols)]


def print_channels(title: str, array: np.ndarray, channels: range = range(8)) -> None:
    print(f"\n{title}  shape={array.shape}")
    for ch in channels:
        cells = occupied_cells(array, ch)
        color = "Black" if ch % 2 == 0 else "White"
        print(
            f"  ch {ch}: {direction_name(ch):3s} / {color:5s} "
            f"count={len(cells):2d} cells={', '.join(cells) if cells else '-'}"
        )


def snapshot(game: twixt.Game, move_number: int, move: str) -> None:
    net = NetInputs(game)
    pegs, input_links, _locations = net.to_input_arrays()
    print("\n" + "=" * 88)
    print(f"After move {move_number:2d} {move}; next player={game.COLOR_NAME[game.turn]}")
    print_channels("A) game.links: original backend planes [x, y, channel]", np.stack(game.links, axis=2))
    print_channels("B) NetInputs.naf: after init_from_game perspective transform [row, col, channel]", net.naf[:, :, :8])
    print_channels("C) to_input_arrays(): final model input links [row, col, channel]", input_links)
    print("  Peg counts by final channel:", [int(np.count_nonzero(pegs[:, :, ch])) for ch in range(2)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--last-moves", type=int, default=3,
        help="show snapshots for the last N moves (default: 3; use 0 for all)",
    )
    args = parser.parse_args()

    game = twixt.Game(allow_scl=False)
    snapshots: list[tuple[int, str, twixt.Game]] = []
    for i, move in enumerate(SAMPLE_MOVES, start=1):
        game.play(move)
        snapshots.append((i, move, game.clone()))

    start = 0 if args.last_moves <= 0 else max(0, len(snapshots) - args.last_moves)
    print("Moves:", " ".join(SAMPLE_MOVES))
    print("Backend class:", f"{twixt.Game.__module__}.{twixt.Game.__name__}")
    print("NetInputs class:", f"{NetInputs.__module__}.{NetInputs.__name__}")
    print(f"Showing {len(snapshots) - start} of {len(snapshots)} move snapshots.")
    for i, move, snapshot_game in snapshots[start:]:
        snapshot(snapshot_game, i, move)


if __name__ == "__main__":
    main()
