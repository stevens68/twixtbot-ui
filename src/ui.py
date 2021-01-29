import PySimpleGUI as sg
import board
import settings as st
from backend import twixt
from os import path

# constants
OFFSET = 5
POPUP_COLOR = "blue"

# globals:

# bot1: thinker[1]
# bot2: thinker[2]
global thinker
thinker = [None, None]

global game
global window


def same_models():
    return path.abspath(settings['-P1_MODEL_FOLDER-']) == path.abspath(settings['-P2_MODEL_FOLDER-'])


def text_label(text, key=None):
    label = text + ':' if len(text) > 0 else text
    return sg.Text(label, justification='r', size=(15, 1), key=key)


def text_field(text, key=None):
    return sg.Text(text, justification='l', size=(10 + OFFSET, 1), key=key)


def pad(s):
    return sg.Text("", size=(s, 1))


def text_output(key, length):
    return sg.Input(size=(length, 1), disabled_readonly_background_color="lightgrey", readonly=True,
                    key=key)


def get_peg_icon(size, key):
    miniSquare = sg.Graph(canvas_size=(size, size),
                          graph_bottom_left=(0, 0),
                          graph_top_right=(size, size),
                          background_color=sg.theme_background_color(),
                          key=key,
                          pad=((6, 72 + OFFSET * 8), (0, 0)),
                          enable_events=False)
    return miniSquare


def update_turn_indicator():

    if game.turn == 1:
        turn = ['x', ' ']
    else:
        turn = [' ', 'x']

    window['-P1_TURNINDICATOR-'].Update(turn[0])
    window['-P2_TURNINDICATOR-'].Update(turn[1])


def update_history():

    text = ""
    for i, move in enumerate(game.history):
        text += "\n" if i > 0 and i % 2 == 0 else ""
        text += str(i + 1).rjust(2, ' ') + '. ' + str(move).upper()

        if move == "swap":
            m1 = game.history[0]
            text += " " + chr(m1.y + ord('A')) + str(m1.x + 1)

        text += "\t\t" if i % 2 == 0 else ""

    window['-HISTORY-'].Update(text)


def update_evals():

    if settings['-P' + str(2 - game.turn) + '_EVAL_BEFORE-']:
        score, best_moves = thinker[game.turn].nm.eval_game(game)
        # get score from white's perspective
        score = round((2 * game.turn - 1) * score, 3)

        window['-P1_EVAL_NUM-'].Update(score)
        window['-P1_EVAL_BAR-'].Update(1000 * score + 1000)
        window['-P1_CURRENT_BEST-'].Update(str(best_moves))

        if not same_models():
            score, best_moves = thinker[1 - game.turn].nm.eval_game(game)
            # get score from white's perspective
            score = round((2 * game.turn - 1) * score, 3)
            print("turn, score, best:", game.turn, score, best_moves)

            window['-P2_EVAL_NUM-'].Update(score)
            window['-P2_EVAL_BAR-'].Update(1000 * score + 1000)
            window['-P2_CURRENT_BEST-'].Update(best_moves)

    else:
        window['-P1_EVAL_NUM-'].Update('')
        window['-P1_EVAL_BAR-'].Update(0)
        window['-P1_CURRENT_BEST-'].Update('')
        if not same_models():
            window['-P2_EVAL_NUM-'].Update('')
            window['-P2_EVAL_BAR-'].Update(0)
            window['-P2_CURRENT_BEST-'].Update('')


def update_thinkers():

    for t in [0, 1]:
        if thinker[t]:
            p = str(2 - t)
            thinker[t].set_temperature(
                float(settings['-P' + p + '_TEMPERATURE-']))
            thinker[t].set_num_trials(int(settings['-P' + p + '_TRIALS-']))


def apply_settings():

    board.draw(game)

    # update settings area
    window['-P1_NAME-'].Update(settings['-P1_NAME-'])
    window['-P1_PEGSQUARE-'].erase()
    window['-P1_PEGSQUARE-'].DrawCircle((7, 9), 6,
                                        settings['-P1_COLOR-'], settings['-P1_COLOR-'])
    window['-P1_AUTO_MOVE-'].Update(settings['-P1_AUTO_MOVE-'])
    window['-P1_MODEL_FOLDER-'].Update(settings['-P1_MODEL_FOLDER-'])
    window['-P1_TRIALS-'].Update(settings['-P1_TRIALS-'])

    window['-P2_NAME-'].Update(settings['-P2_NAME-'])
    window['-P2_PEGSQUARE-'].erase()
    window['-P2_PEGSQUARE-'].DrawCircle((7, 9), 6,
                                        settings['-P2_COLOR-'], settings['-P2_COLOR-'])
    window['-P2_AUTO_MOVE-'].Update(settings['-P2_AUTO_MOVE-'])

    if '-P2_MODEL_FOLDER-' in window.AllKeysDict:
        window['-P2_MODEL_FOLDER-'].Update(settings['-P2_MODEL_FOLDER-'])

    window['-P2_TRIALS-'].Update(settings['-P2_TRIALS-'])

    update_turn_indicator()

    update_thinkers()


def settings_dialog():
    dialog = st.create_settings_window(settings)
    while True:
        event, values = dialog.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Apply & Save':
            st.save_settings(settings, values)
            apply_settings()
            break
    dialog.close()


def game_over_message(game):
    if game.just_won():
        sg.popup_quick_message('Game over: ' + settings['-P' + str(1 + game.turn) + '_NAME-'] + ' has won!',
                               keep_on_top=True, line_width=200, background_color=POPUP_COLOR)
        return True
    return False


def execute_move(move):
    if move == "resign":
        game_over_message(game)
        return
    elif move == "swap":
        game.play_swap()
    else:
        game.play(move)
    board.create_move_objects(game, len(game.history) - 1)
    game_over_message(game)
    print(game.turn)


def board_event():

    if game_over_message(game):
        return
    move = board.getMove(game, values['-BOARD-'])
    print("move: ", move)
    if move:
        execute_move(move)


def row_turn_indicators():
    return [text_label('turn'),
            sg.Text(' ', key='-P1_TURNINDICATOR-',
                    text_color="yellow"), pad(13),
            sg.Text(' ', key='-P2_TURNINDICATOR-', text_color="yellow")]


def row_peg_icons():
    return [text_label('color'), get_peg_icon(16, '-P1_PEGSQUARE-'), get_peg_icon(16, "-P2_PEGSQUARE-")]


def row_player_names():
    return [text_label('name'),
            text_output('-P1_NAME-', 14), pad(1),
            text_output('-P2_NAME-', 14)]
    #text_field(settings['-P1_NAME-'], key='-P1_NAME-'),
    # text_field(settings['-P2_NAME-'], key='-P2_NAME-')]


def row_auto_moves():
    return [text_label('auto move'),
            sg.Checkbox(
        text="", enable_events=True, default=settings['-P1_AUTO_MOVE-'], key='-P1_AUTO_MOVE-', size=(7 + OFFSET, 1)),
        sg.Checkbox(
        text="", enable_events=True, default=settings['-P2_AUTO_MOVE-'], key='-P2_AUTO_MOVE-', size=(7 + OFFSET, 1))]


def row_model_folders():
    if same_models():
        return [text_label('model folder'),
                text_output('-P1_MODEL_FOLDER-', 33)]
    else:
        return [text_label('model folder'),
                text_output('-P1_MODEL_FOLDER-', 14), pad(1),
                text_output('-P2_MODEL_FOLDER-', 14)]


def row_trials():
    return [text_label('trials'),
            sg.Input(default_text=settings['-P1_TRIALS-'], size=(7, 1), enable_events=True,
                     key='-P1_TRIALS-'), pad(2 + OFFSET),
            sg.Input(default_text=settings['-P2_TRIALS-'], size=(7, 1), enable_events=True,
                     key='-P2_TRIALS-')]


def row_progress():
    return [text_label('progress'), sg.ProgressBar(
        1000, orientation='h', size=(21.5, 20), key='-PROGRESSBAR-')]


def row_time():
    return [text_label('time'), text_output('-ETA-', 33)]


def row_current_best():
    if same_models():
        return [text_label('current best'),
                text_output('-P1_CURRENT_BEST-', 33)]
    else:
        return [text_label('current best'),
                text_output('-P1_CURRENT_BEST-', 14), pad(1),
                text_output('-P2_CURRENT_BEST-', 14)]


def row_eval_bars():
    colors = (settings['-P1_COLOR-'], settings['-P2_COLOR-'])
    if same_models():
        return [text_label(''), sg.ProgressBar(2000, orientation='h', size=(21.5, 8), key='-P1_EVAL_BAR-',
                                               bar_color=colors)]
    else:
        return [text_label(''), sg.ProgressBar(2000, orientation='h', size=(9.3, 8), key='-P1_EVAL_BAR-', bar_color=colors), pad(1),
                sg.ProgressBar(2000, orientation='h', size=(
                    9.3, 8), key='-P2_EVAL_BAR-', bar_color=colors)
                ]


def row_eval_numbers():
    if same_models():
        return [text_label('eval'),
                text_output('-P1_EVAL_NUM-', 14)]
    else:
        return [text_label('eval'),
                text_output('-P1_EVAL_NUM-', 14), pad(1),
                text_output('-P2_EVAL_NUM-', 14)]


def row_history():
    return [text_label('history'), sg.Multiline(default_text='', font=("Courier", 10),
                                                background_color='lightgrey', autoscroll=True,
                                                key='-HISTORY-', disabled=True, size=(28, 6))]


def init_bots():
    sg.popup_quick_message('Loading model ... ', keep_on_top=True, line_width=200,
                           background_color=POPUP_COLOR)
    thinker[1] = st.load_model(1, settings)
    thinker[0] = st.load_model(2, settings)
    # warm-up bot1 before first move
    thinker[0].nm.eval_game(game)
    thinker[1].nm.eval_game(game)


def bot_move():
    tup = thinker[game.turn].pick_move(game)
    m = tup[0] if type(tup) == tuple else tup
    print("bot_move: ", game.turn, m)
    return m


def undo_move():
    global game
    if len(game.history) > 1:
        game.undo()
    elif len(game.history) == 1:
        game = twixt.Game(settings['-ALLOW_SWAP-'],
                          settings['-ALLOW_SCL-'])
    # switch of auto move
    if settings['-P' + str(2 - game.turn) + '_AUTO_MOVE-']:
        settings['-P' + str(2 - game.turn) + '_AUTO_MOVE-'] = False
        apply_settings()


def update_settings(event, values):
    settings[event] = values[event]

##### main window ##################################



# first, read settings file
settings = st.load_settings()
# sg.theme(settings['-THEME-'])

# then create a twixt board (draw it later)
board = board.TwixtBoard(settings)

# set the layout
menu_def = [['File', ['Settings...', 'Exit']],
            ['Help', 'About...']]
board_col = sg.Column([[board.graph]])

control_col = sg.Column([row_turn_indicators(),
                         row_peg_icons(),
                         row_player_names(),
                         row_auto_moves(),
                         row_trials(),
                         [sg.HSeparator()],
                         row_model_folders(),
                         row_eval_bars(),
                         row_eval_numbers(),
                         row_current_best(),
                         [sg.HSeparator()],
                         row_progress(),
                         row_time(),
                         [sg.HSeparator()],
                         row_history()],
                        vertical_alignment='top')

layout = [
    [sg.Menu(menu_def, tearoff=False)],
    [board_col, control_col],
    [
        sg.Button('Bot Move', size=(10, 1)),
        sg.Button('Stop Bot', size=(10, 1)),
        sg.Button('Undo Move', size=(10, 1)),
        sg.Button('Reset', size=(10, 1)),
        sg.Button('Exit', size=(10, 1))]
]

# create main window

window = sg.Window('twixtbot-ui', layout, margins=(25, 25))
window.Finalize()

# draw twixt board
game = twixt.Game(settings['-ALLOW_SWAP-'], settings['-ALLOW_SCL-'])


apply_settings()
window.refresh()

init_bots()


# Event Loop
while True:
    board.draw(game)
    update_turn_indicator()
    update_history()
    update_evals()
    if not game.just_won() and settings['-P' + str(2 - game.turn) + '_AUTO_MOVE-']:
        print("AUTO_MOVE:" + str(game.turn))
        move = bot_move()
        execute_move(move)
        board.draw(game)
        update_turn_indicator()
        update_history()
        update_evals()

    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Settings...':
        settings_dialog()
    elif event == '-BOARD-':
        board_event()
    elif event == 'Reset':
        game = twixt.Game(settings['-ALLOW_SWAP-'], settings['-ALLOW_SCL-'])
    elif event == 'Undo Move':
        undo_move()
    elif event in ['-P1_AUTO_MOVE-', '-P2_AUTO_MOVE-']:
        update_settings(event, values)

window.close()
