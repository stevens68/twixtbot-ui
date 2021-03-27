import PySimpleGUI as sg
from backend import twixt
import layout as lt
import string


def str2twixt(move):
    """ Converts one move string to a twixt backend class move.

    Handles both T1-style coordinates (e.g.: 'd5', 'f18'') as well as tsgf-
    style coordinates (e.g.: 'fg', 'bi') as well as special strings
    ('swap' and 'resign'). It can handle letter in upper as well as lowercase.

    Args:
        move: string with a move

    Returns:
        twixt.SWAP or twixt.RESIGN or twixt.Point

    Raises
        ValueError if the move_str can't be parsed in any valid format

    Examples:
        >>> str2twixt('b3')
        b3
        >>> str2twixt('i18')
        i18
        >>> str2twixt('fj')
        f10
        >>> str2twixt('swap')
        'swap'
        >>> str2twixt('resign')
        'resign'
        >>> str2twixt('123')
        ValueError: Can't parse move: '123'
        >>> str2twixt('invalid')
        ValueError: Can't parse move: 'invalid'
    """

    # Handle swap and resign
    if move.lower() == twixt.SWAP.lower():
        return twixt.SWAP
    elif move.lower() == twixt.RESIGN.lower():
        return twixt.RESIGN

    # Handle T1-style moves
    elif move[0] in string.ascii_letters and move[-1] in string.digits:
        return twixt.Point(move)

    # Handle tsgf-stype moves
    elif len(move) == 2 and all(c in string.ascii_letters for c in move):
        return twixt.Point(move[0] + str(ord(move[1].lower()) - ord('a') + 1))

    # Can't handle move. Throw exception
    raise ValueError(f"Can't parse move: '{move}'")


def parse_t1_file(content):
    """Returns (players, moves) from a list of strings from a T1 file

    Args:
        content: list of strings: content from a T1 file

    Returns:
        tuple: (list: players as strings, list: twixt moves)

    Raises:
        ValueError: if players or moves data can't be interpreted

    Examples:
        >>> content = [
            '# File created by T1j',
            '# T1j is a program to play TwixT (mail@johannes-schwagereit.de)',
            '1 # version of file-format',
            'Player# Name of Player 1',
            'Computer# Name of Player 2',
            '24# y-size of board',
            '24# x-size of board',
            'H# player 1 human or computer',
            'C# player 2 human or computer',
            '1# starting player (1 plays top-down)',
            'V# Direction of letters',
            'N# pierule?',
            'N# game already over?',
            'L10', 'L17', 'Q15', 'Q8',  'S12', 'P11', 'O14', 'P19', 'V18', 'U15',
            'V16', 'T17', 'U14', 'V17', 'W16', 'W15', 'F16', 'L19', 'F20', 'I14',
            'F12', 'X13', 'G14', 'G8',  'I9',  'J9',  'J7',  'E9',  'G10', 'N18',
            'J3', 'G20', 'G18', 'E21'
        ]
        >>> parse_t1_file(content)
            (['Player', 'Computer'],
             [l10, l17, q15, q8,  s12, p11, o14, p19, v18, u15, v16, t17, u14, v17, w16, w15, f16, 
             l19, f20, i14, f12, x13, g14,  g8, i9,  j9, j7, e9, g10, n18, j3, g20, g18, e21])
    """
    MOVES_STARTLINE = 13
    PLAYER_LINES = [3, 4]
    COMMENT_CHAR = '#'

    try:
        players = [content[linenr].split(COMMENT_CHAR)[0]
                   for linenr in PLAYER_LINES]
    except Exception:
        raise ValueError("Can't read player names from T1 file")

    try:
        moves = [str2twixt(move) for move in content[MOVES_STARTLINE:]
                 if len(move) > 0]
    except Exception:
        # Just pass on the exception from str2twixt
        raise

    return players, moves


def parse_tsgf_file(content):
    """Returns (players, moves) from a list of strings from a tsgf file

    Args:
        content: list of strings: content from a tsgf file

    Returns:
        tuple: (list: players as strings, list: twixt moves)

    Raises:
        ValueError: if players or moves data can't be interpreted

    Examples:
        >>> content = [
            '(;FF[4]EV[twixt.ld.DEFAULT]PB[agtoever]PW[Jan Krabbenbos]SZ[24]SO[https://www.littlegolem.net];b[pl];r[ps];b[pr];r[rt];b[ot];r[po];b[pn];r[qq];b[op];r[pg];b[nh];r[oj];b[oi];r[qi];b[nk];r[nf];b[mf])'
        ]
        >>> parse_tsgf_file(content)
            (['agtoever', 'Jan Krabbenbos'], [p12, p19, p18, r20, o20, p15, p14, q17, o16, p7, n8, o10, o9, q9, n11, n6, m6])
    """
    PLAYERS_STR = ('PB', 'PW')
    TURN_STR = ('r[', 'b[')

    if len(content) > 1:
        raise ValueError('Found more than 1 line in a tsgf file.')

    try:
        player_idx = [content[0].find(key) for key in PLAYERS_STR]
        players = [content[0][idx + 3:content[0].find(']', idx)]
                   for idx in player_idx]
    except Exception:
        raise ValueError("Can't read player names from tsgf file")

    try:
        raw_moves = [line[2:-1].strip(']) ')
                     for line in content[0].split(';')
                     if line[:2] in TURN_STR]
        moves = list(map(str2twixt, raw_moves))
    except Exception:
        # Just pass on the exception from str2twixt
        raise

    return players, moves


def get_game():
    """Returns (players, moves) from a file, chosen by the user

    Shows a file-open dialog to the user.
    The chosen file is read and parsed into players and moves.
    The resulting player name list and moves list is returned.
    Exceptions that occur while opening and/or parsing the file
    are handled within this function.

    Args:
        None

    Returns:
        tuple: (list: players as strings, list: twixt moves)
    """

    # Get filename
    file_name = sg.PopupGetFile('Choose file', file_types=(
        ("All Files", "*.*"),
        ("T1j Files", "*.T1"),
        ("Little Golem Files", "*.tsgf")), no_window=True, keep_on_top=True)

    if file_name is None or file_name == "":
        return None, None

    # Open file
    try:
        with open(file_name, "tr") as f:
            content = list(map(lambda s: s.strip(), f.readlines()))
    except Exception:
        sg.popup_ok(f"Can't open {file_name} as a valid Twixt file.")
        return None, None

    # Parse file
    try:
        if file_name[-2:].upper() == 'T1':
            return parse_t1_file(content)
        elif file_name[-4:].lower() == 'tsgf':
            return parse_tsgf_file(content)
        else:
            lt.popup("Didn't recognize the filename extension.")
    except Exception as e:
        sg.popup_ok(f"Error '{e}' while opening file {file_name}")

    return None, None


def save_game(players=['Player1', 'Player2'],
              moves=[''],
              board_size=24,
              game_over=False):
    """ Saves a Twixt game to T1 file, chosen by the user

    Shows a file-save dialog to the user.
    The twixt game given by the function parameters are saved to the file.
    Only .T1 file format is currently supported.
    Exceptions that occur while saving the file are handled within
    this function.

    Args:
        players: list of two strings with player names
        moves: list of twixt moves
        board_size: int with board size (defaults to 24)
        game_over: boolean, true if the game is over (defaults to False)

    Returns:
        None
    """

    # Get filename
    file_name = sg.PopupGetFile('Choose file', file_types=(
        ("T1j Files", "*.T1"),), no_window=True, save_as=True, keep_on_top=True)

    if file_name is None or file_name == "":
        return

    # Build file contents
    try:
        content = [
            '# File created by twixtbot-ui',
            '# twixtbot-ui is a program to play TwixtT (https://github.com/stevens68/twixtbot-ui)',
            '1 # version of file-format',
            str(players[0]) + ' # Name of player 1',
            str(players[1]) + ' # Name of player 2',
            str(board_size) + ' # y-size of board',
            str(board_size) + ' # x-size of board',
            'H # player 1 human or computer',
            'H # player 2 human or computer',
            '1 # starting player (1 plays top-down)',
            'V # direction of letters',
            'Y # pierule?',
            ('Y' if game_over else 'N') + ' # game already over?'
        ]

        content += [str(m).upper() for m in moves]

    except Exception as e:
        sg.popup_ok('Could not create file contents. Game is NOT saved!')
        print(e)
        return

    # Write file
    try:
        with open(file_name, "tw") as f:
            f.write('\n'.join(content))
    except Exception:
        sg.popup_ok(f"Can't write {file_name}. Game is NOT saved!")
        return

    sg.popup_ok(f'Game saved successfully as {file_name}')
    return
