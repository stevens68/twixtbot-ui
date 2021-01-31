import PySimpleGUI as sg
import board
import settings as st
from backend import twixt
from os import path

# constants
POPUP_COLOR = "blue"
#TURNCHAR = '\u2605'
TURNCHAR = 'X'
TRIALS_MAX = 50000
TRIALS_RES = 100

# globals:

# bot1: thinker[1]
# bot2: thinker[2]
global thinker
thinker = [None, None]

global game
global window


def same_models():
    return path.abspath(settings['-P1_MODEL_FOLDER-']) == path.abspath(settings['-P2_MODEL_FOLDER-'])


def get_tooltip(player):
    p = str(player)
    text = "allow swap:\t" + \
        str(settings['-ALLOW_SWAP-']) + "   \n"
    text += "allow scl:\t" + \
        str(settings['-ALLOW_SCL-']) + "   \n"
    text += "eval before:\t" + \
        str(settings['-P' + p + '_EVAL_BEFORE-']) + "   \n"
    text += "temperature:\t" + \
        str(settings['-P' + p + '_TEMPERATURE-']) + "   \n"
    text += "add noise:\t" + str(settings['-P' + p + '_ADD_NOISE-']) + "   \n"
    text += "random rotation: \t" + \
        str(settings['-P' + p + '_RANDOM_ROTATION-']) + "   \n"
    text += "cpuct:\t\t" + str(settings['-P' + p + '_CPUCT-']) + "   "
    return text


def get_peg_icon(player):
    size = 16
    key = '-P' + str(player) + '_PEGSQUARE-'
    miniSquare = sg.Graph(canvas_size=(size, size),
                          graph_bottom_left=(0, 0),
                          graph_top_right=(size, size),
                          background_color=sg.theme_background_color(),
                          key=key,
                          pad=((6, 72 + st.OFFSET * 8), (0, 0)),
                          enable_events=False,
                          tooltip=get_tooltip(player))
    return miniSquare


def row_peg_icons():
    return [st.text_label('color'), get_peg_icon(1), get_peg_icon(2)]


def row_turn_indicators():
    return [st.text_label('turn'),
            sg.Input(size=(14, 1), disabled_readonly_background_color=st.RO_BACKGROUND, readonly=True,
                     key='-P1_TURNINDICATOR-', text_color="yellow"), st.pad(1),
            sg.Input(size=(14, 1), disabled_readonly_background_color=st.RO_BACKGROUND, readonly=True,
                     key='-P2_TURNINDICATOR-', text_color="yellow")]


def row_player_names():
    return [st.text_label('name'),
            st.text_output('-P1_NAME-', 14), st.pad(1),
            st.text_output('-P2_NAME-', 14)]


def row_auto_moves():
    return [st.text_label('auto move'),
            sg.Checkbox(
        text="", enable_events=True, default=settings['-P1_AUTO_MOVE-'], key='-P1_AUTO_MOVE-', size=(7 + st.OFFSET, 1)),
        sg.Checkbox(
        text="", enable_events=True, default=settings['-P2_AUTO_MOVE-'], key='-P2_AUTO_MOVE-', size=(7 + st.OFFSET, 1))]


def row_model_folders():
    if same_models():
        return [st.text_label('model folder'),
                st.text_output('-P1_MODEL_FOLDER-', 33)]
    else:
        return [st.text_label('model folder'),
                st.text_output('-P1_MODEL_FOLDER-', 14), st.pad(1),
                st.text_output('-P2_MODEL_FOLDER-', 14)]


def row_trials():

    return [st.text_label('trials'),
            sg.Slider(range=(0, TRIALS_MAX), default_value=settings['-P1_TRIALS-'], resolution=TRIALS_RES, orientation='horizontal',
                      enable_events=True, size=(11, 20), key='-P1_TRIALS-'),
            st.pad(2),
            sg.Slider(range=(0, TRIALS_MAX), default_value=settings['-P2_TRIALS-'], resolution=TRIALS_RES, orientation='horizontal',
                      enable_events=True, size=(11, 20), key='-P2_TRIALS-'),
            ]


def row_progress():
    return [st.text_label('progress'), sg.ProgressBar(
        1000, orientation='h', size=(21.5, 20), key='-PROGRESSBAR-')]


def row_time():
    return [st.text_label('time'), st.text_output('-ETA-', 33)]


def row_current_best():
    if same_models():
        return [st.text_label('current best'),
                st.text_output('-P1_CURRENT_BEST-', 33)]
    else:
        return [st.text_label('current best'),
                st.text_output('-P1_CURRENT_BEST-', 14), st.pad(1),
                st.text_output('-P2_CURRENT_BEST-', 14)]


def row_eval_bars():
    colors = (settings['-P1_COLOR-'], settings['-P2_COLOR-'])
    if same_models():
        return [st.text_label('eval'), sg.ProgressBar(2000, orientation='h', size=(21.5, 8), key='-P1_EVAL_BAR-',
                                                      bar_color=colors)]
    else:
        return [st.text_label(''), sg.ProgressBar(2000, orientation='h', size=(9.3, 8), key='-P1_EVAL_BAR-', bar_color=colors), st.pad(1),
                sg.ProgressBar(2000, orientation='h', size=(
                    9.3, 8), key='-P2_EVAL_BAR-', bar_color=colors)
                ]


def row_eval_numbers():
    if same_models():
        return [st.text_label(''),
                st.text_output('-P1_EVAL_NUM-', 14)]
    else:
        return [st.text_label('eval'),
                st.text_output('-P1_EVAL_NUM-', 14), st.pad(1),
                st.text_output('-P2_EVAL_NUM-', 14)]


def row_history():
    return [st.text_label('moves'), sg.Multiline(default_text='', font=("Courier", 10),
                                                 background_color='lightgrey', autoscroll=True,
                                                 key='-HISTORY-', disabled=True, size=(28, 6))]


def update_turn_indicators():

    turn = ["", ""]
    if game.result == twixt.RESIGN:
        turn[game.turn] = ""
        turn[1 - game.turn] = "resigned"
    elif game.just_won():
        turn[game.turn] = "won"
        turn[1 - game.turn] = ""
    elif game.turn == 1:
        turn = [TURNCHAR, '']
    else:
        turn = ['', TURNCHAR]

    window['-P1_TURNINDICATOR-'].Update(turn[0])
    window['-P2_TURNINDICATOR-'].Update(turn[1])


def update_history():

    text = ""
    for i, move in enumerate(game.history):
        text += "\n" if i > 0 and i % 2 == 0 else ""
        text += str(i + 1).rjust(2, ' ') + '. ' + str(move).upper()

        if move == twixt.SWAP:
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
            thinker[t].allow_swap = settings['-ALLOW_SWAP-']
            thinker[t].num_trials = int(settings['-P' + p + '_TRIALS-'])
            thinker[t].temperature = float(
                settings['-P' + p + '_TEMPERATURE-'])
            thinker[t].random_rotation = settings['-P' +
                                                  p + '_RANDOM_ROTATION-']
            thinker[t].add_noise = float(settings['-P' + p + '_ADD_NOISE-'])
            thinker[t].nm.cpuct = float(settings['-P' + p + '_CPUCT-'])


def update_game():
    game.allow_scl = settings['-ALLOW_SCL-']


def apply_settings():

    board.draw(game)

    # update setting controls in main window
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

    update_turn_indicators()
    update_thinkers()
    update_game()


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


def game_over():

    if game.just_won():
        message = 'Game over: ' + \
            settings['-P' + str(1 + game.turn) + '_NAME-'] + ' has won!'
        sg.popup_quick_message(message, keep_on_top=True,
                               line_width=200, background_color=POPUP_COLOR)
        return True
    elif game.result == twixt.RESIGN:
        message = 'Game over: ' + \
            settings['-P' + str(2 - game.turn) + '_NAME-'] + ' has resigned!'
        sg.popup_quick_message(message, keep_on_top=True,
                               line_width=200, background_color=POPUP_COLOR)
        return True
    return False


def execute_move(move):
    if move == twixt.RESIGN:
        game.result = twixt.RESIGN
        game_over()
        return
    elif move == twixt.SWAP:
        game.play_swap()
    else:
        game.play(move)
    board.create_move_objects(game, len(game.history) - 1)
    game_over()


def board_event():

    if game_over():
        return
    move = board.get_move(game, values['-BOARD-'])
    if move:
        execute_move(move)


def resign():

    if game_over():
        return
    execute_move(twixt.RESIGN)


def prepare_bots():
    sg.popup_quick_message('initializing bots ... ', keep_on_top=True, line_width=200,
                           background_color=POPUP_COLOR)
    thinker[1] = st.init_bot(1, settings)
    thinker[0] = st.init_bot(2, settings)
    # warm-up bot1 before first move
    thinker[0].nm.eval_game(game)
    thinker[1].nm.eval_game(game)


def bot_move():
    tup = thinker[game.turn].pick_move(game)
    m = tup[0] if type(tup) == tuple else tup
    print("bot_move: ", game.turn, m)
    return m


def undo_move():

    if game_over():
        return

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

control_col = sg.Column([row_peg_icons(),
                         row_player_names(),
                         row_turn_indicators(),
                         row_auto_moves(),
                         row_trials(),
                         st.row_separator('network'),
                         row_model_folders(),
                         row_eval_bars(),
                         row_eval_numbers(),
                         row_current_best(),
                         row_progress(),
                         row_time(),
                         st.row_separator('history'),
                         row_history()],
                        vertical_alignment='top')

layout = [
    [sg.Menu(menu_def, tearoff=False)],
    [board_col, control_col],
    [
        sg.Button('Bot', size=(10, 1)),
        sg.Button('Stop', size=(10, 1)),
        sg.Button('Cancel', size=(10, 1)),
        sg.Button('Undo', size=(10, 1)),
        sg.Button('Resign', size=(10, 1)),
        sg.Button('Reset', size=(10, 1))]
]

# create main window

window = sg.Window('twixtbot-ui', layout, margins=(25, 25))
window.Finalize()

# draw twixt board
game = twixt.Game(settings['-ALLOW_SWAP-'], settings['-ALLOW_SCL-'])


apply_settings()
window.refresh()

prepare_bots()


# Event Loop
while True:
    board.draw(game)
    update_turn_indicators()
    update_history()
    update_evals()
    if not game_over() and settings['-P' + str(2 - game.turn) + '_AUTO_MOVE-']:
        move = bot_move()
        execute_move(move)
        board.draw(game)
        update_turn_indicators()
        update_history()
        update_evals()

    event, values = window.read()
    print(event)  # don't print values; contains unprintable utf-8 char
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Settings...':
        settings_dialog()
    elif event == '-BOARD-':
        board_event()
    elif event == 'Resign':
        resign()
    elif event == 'Reset':
        game = twixt.Game(settings['-ALLOW_SWAP-'], settings['-ALLOW_SCL-'])
    elif event == 'Undo':
        undo_move()
    elif event in ['-P1_AUTO_MOVE-', '-P2_AUTO_MOVE-', '-P1_TRIALS-', '-P2_TRIALS-']:
        update_settings(event, values)
        update_thinkers()
    elif event == 'Bot':
        if not game_over():
            move = bot_move()
            execute_move(move)


window.close()
