import PySimpleGUI as sg
from backend import twixt
import layout as lt


def get_game():
    file_name = sg.PopupGetFile('Choose file', file_types=(
        ("All Files", "*.*"), ("T1j Files", "*.T1")), no_window=True)
    if file_name is None:
        return None, None

    try:
        with open(file_name, "tr") as f:
            f.read()
    except Exception:
        lt.popup(file_name + " seems to be a binary file.")
        return None, None

    content = []
    with open(file_name) as f:
        content = [line.rstrip() for line in f]

    offset = 13
    # basic file format check
    if len(content) < offset:
        lt.popup(file_name + " should have more than 13 lines")
        return None, None

    try:
        # get player names
        players = [content[3].split('#')[0], content[4].split('#')[0]]
    except Exception:
        lt.popup(file_name + ": cannot read player names from line 4 and 5.")
        return None, None

    moves = []
    for i, m in enumerate(content[offset:]):
        if m.lower() == twixt.SWAP and len(moves) == 1:
            # swap as second move is ok
            moves.append(m)
        elif m.lower() == twixt.RESIGN and len(moves) == len(content) - offset - 1:
            # resign as last move is ok
            moves.append(m)
        else:
            try:
                move = twixt.Point(m)
                if move in moves and (moves[1] != twixt.SWAP or move in moves[1:]):
                    lt.popup(file_name + ": duplicate move in line " +
                             str(i + offset + 1) + ": " + str(move))
                    return None, None
                moves.append(move)
            except Exception:
                lt.popup(file_name + ": invalid move in line " +
                         str(i + offset + 1) + ": " + m)
                return None, None

    return players, moves
