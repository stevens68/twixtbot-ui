import PySimpleGUI as sg
import board
import settings as st
import twixt


OFFSET = 5
# helper


def text_label(text, key=None):
    return sg.Text(text + ':', justification='r', size=(15, 1), key=key)


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

    print("Evaluating:")
    score, best_moves = thinker[1].nm.eval_game(game)
    # get score from white's perspective
    score = round((2 * game.turn - 1) * score, 3)
    print("turn, score, best:", game.turn, score, best_moves)

    window['-P1_EVAL_NUM-'].Update(score)
    window['-P1_EVAL_BAR-'].Update(1000 * score + 1000)
    window['-P1_CURRENT_BEST-'].Update(str(best_moves))

    if settings['-P1_MODEL_FOLDER-'] != settings['-P2_MODEL_FOLDER-']:
        score, best_moves = thinker[0].nm.eval_game(game)
        # get score from white's perspective
        score = round((2 * game.turn - 1) * score, 3)
        print("turn, score, best:", game.turn, score, best_moves)

        window['-P2_EVAL_NUM-'].Update(score)
        window['-P2_EVAL_BAR-'].Update(1000 * score + 1000)
        window['-P2_CURRENT_BEST-'].Update(best_moves)


def apply_settings():

    board.draw(game)
    # sg.ChangeLookAndFeel(settings['-THEME-'])

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


def settings_dialog():
    dialog = st.create_settings_window(settings)
    while True:
        event, values = dialog.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == '-P1_LOAD-':
            thinker[1] = st.load_model(1, settings)
        elif event == '-P2_LOAD-':
            thinker[0] = st.load_model(2, settings)
        elif event == 'Apply & Save':
            st.save_settings(settings, values)
            apply_settings()
            break
    dialog.close()


def game_over_message(game):
    if game.just_won():
        sg.popup_quick_message('Game over: ' + settings['-P' + str(1 + game.turn) + '_NAME-'] + ' has won!',
                               keep_on_top=True, line_width=200, background_color=st.POPUP_COLOR)
        return True
    return False


def board_event():

    if game_over_message(game):
        return
    move = board.getMove(game, values['-BOARD-'])
    print("move: ", move)
    if move:
        if move == "swap":
            game.play_swap()
        else:
            game.play(move)
        board.create_move_objects(game, len(game.history) - 1)
        game_over_message(game)
        print(game.turn)


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
        text="", default=settings['-P1_AUTO_MOVE-'], key='-P1_AUTO_MOVE-', size=(7 + OFFSET, 1)),
        sg.Checkbox(
        text="", default=settings['-P2_AUTO_MOVE-'], key='-P2_AUTO_MOVE-', size=(7 + OFFSET, 1))]


def row_model_folders():
    if settings['-P1_MODEL_FOLDER-'] == settings['-P2_MODEL_FOLDER-']:
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
                     key='-P2_TRIALS-'), pad(2 + OFFSET)]


def row_progress():
    return [text_label('progress'), sg.ProgressBar(
        1000, orientation='h', size=(22, 20), key='-PROGRESSBAR-')]


def row_time():
    return [text_label('time'), text_output('-ETA-', 14)]


def row_current_best():
    if settings['-P1_MODEL_FOLDER-'] == settings['-P2_MODEL_FOLDER-']:
        return [text_label('current best'),
                text_output('-P1_CURRENT_BEST-', 33)]
    else:
        return [text_label('current best'),
                text_output('-P1_CURRENT_BEST-', 14), pad(1),
                text_output('-P2_CURRENT_BEST-', 14)]


def row_eval_bars():
    colors = (settings['-P1_COLOR-'], settings['-P2_COLOR-'])
    if settings['-P1_MODEL_FOLDER-'] == settings['-P2_MODEL_FOLDER-']:
        return [text_label('eval'), sg.ProgressBar(2000, orientation='h', size=(22, 10), key='-P1_EVAL_BAR-',
                                                   bar_color=colors)]
    else:
        return [text_label('eval'), sg.ProgressBar(2000, orientation='h', size=(10, 10), key='-P1_EVAL_BAR-', bar_color=colors),
                sg.ProgressBar(2000, orientation='h', size=(
                    10, 20), key='-P2_EVAL_BAR-', bar_color=colors)
                ]


def row_eval_numbers():
    if settings['-P1_MODEL_FOLDER-'] == settings['-P2_MODEL_FOLDER-']:
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
                         [sg.HSeparator()],
                         row_model_folders(),
                         row_trials(),
                         [sg.HSeparator()],
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
# window.refresh()


thinker = [None, None]

thinker[1] = st.load_model(1, settings)
thinker[0] = st.load_model(2, settings)


# Event Loop
while True:
    update_turn_indicator()
    update_history()
    update_evals()
    if not game.just_won() and settings['-P' + str(2 - game.turn) + '_AUTO_MOVE-']:
        print("AUTO_MOVE:" + str(game.turn))
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
        board.draw(game)
    elif event == 'Undo Move':
        if len(game.history) > 1:
            game.undo()
        elif len(game.history) == 1:
            game = twixt.Game(settings['-ALLOW_SWAP-'],
                              settings['-ALLOW_SCL-'])
        board.draw(game)


window.close()
