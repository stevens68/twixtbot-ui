import PySimpleGUI as sg
import board
import settings as st
import twixt


OFFSET = 5
# helper


def TextLabel(text, key=None):
    return sg.Text(text + ':', justification='r', size=(15, 1), key=key)


def TextField(text, key=None):
    return sg.Text(text, justification='l', size=(10 + OFFSET, 1), key=key)


def Pad(s):
    return sg.Text("", size=(s, 1))


def TextOutput(key, length):
    return sg.Input(size=(length, 1), disabled_readonly_background_color="lightgrey", readonly=True,
                    key=key)


def getPegIcon(size, key):
    miniSquare = sg.Graph(canvas_size=(size, size),
                          graph_bottom_left=(0, 0),
                          graph_top_right=(size, size),
                          background_color=sg.theme_background_color(),
                          key=key,
                          pad=((6, 72 + OFFSET * 8), (0, 0)),
                          enable_events=False)
    return miniSquare


def updateTurnIndicator(window, game):

    if game.turn == 1:
        turn = ['x', ' ']
    else:
        turn = [' ', 'x']

    window['-P1_TURNINDICATOR-'].Update(turn[0])
    window['-P2_TURNINDICATOR-'].Update(turn[1])


def applySettings(window, board, settings, game):

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
    window['-P2_MODEL_FOLDER-'].Update(settings['-P2_MODEL_FOLDER-'])
    window['-P2_TRIALS-'].Update(settings['-P2_TRIALS-'])

    updateTurnIndicator(window, game)


def settingsDialog():
    dialog = st.createSettingsWindow(settings)
    while True:
        event, values = dialog.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == '-P1_LOAD-':
            thinker[1] = st.loadModel(1, settings)
        elif event == '-P2_LOAD-':
            thinker[0] = st.loadModel(2, settings)
        elif event == 'Apply & Save':
            st.saveSettings(settings, values)
            applySettings(window, board, settings, game)
            break
    dialog.close()


def gameOverMessage(game):
    if game.just_won():
        sg.popup_quick_message('Game over: ' + settings['-P' + str(1 + game.turn) + '_NAME-'] + ' has won!',
                               keep_on_top=True, line_width=200, background_color=st.POPUP_COLOR)
        return True
    return False


def boardEvent():

    if gameOverMessage(game):
        return
    move = board.getMove(game, values['-BOARD-'])
    print("move: ", move)
    if move:
        if move == "swap":
            game.play_swap()
        else:
            game.play(move)
        board.create_move_objects(game, len(game.history) - 1)
        gameOverMessage(game)
        print(game.turn)

##### main window ##################################



# first, read settings file
settings = st.loadSettings()
# sg.theme(settings['-THEME-'])

# then create a twixt board (draw it later)
board = board.TwixtBoard(settings)

# set the layout
menu_def = [['File', ['Settings...', 'Exit']],
            ['Help', 'About...']]

board_col = sg.Column([[board.graph]])

control_col = sg.Column([[TextLabel('turn'),
                          sg.Text(' ', key='-P1_TURNINDICATOR-',
                                  text_color="yellow"), Pad(13),
                          sg.Text(' ', key='-P2_TURNINDICATOR-', text_color="yellow")],
                         [TextLabel('color'),
                          getPegIcon(16, '-P1_PEGSQUARE-'),
                          getPegIcon(16, "-P2_PEGSQUARE-")],
                         [TextLabel('name'),
                          TextField(settings['-P1_NAME-'],
                                    key='-P1_NAME-'),
                          TextField(settings['-P2_NAME-'], key='-P2_NAME-')],
                         [TextLabel('auto move'),
                          sg.Checkbox(
                              text="", default=settings['-P1_AUTO_MOVE-'], key='-P1_AUTO_MOVE-', size=(7 + OFFSET, 1)),
                          sg.Checkbox(
                              text="", default=settings['-P2_AUTO_MOVE-'], key='-P2_AUTO_MOVE-', size=(7 + OFFSET, 1))],
                         [sg.HSeparator()],
                         [TextLabel('model folder'),
                          sg.Input(default_text=settings['-P1_MODEL_FOLDER-'],
                                   disabled_readonly_background_color='lightgrey',
                                   key='-P1_MODEL_FOLDER-', disabled=True, size=(14, 1)), Pad(1),
                          sg.Input(default_text=settings['-P2_MODEL_FOLDER-'],
                                   disabled_readonly_background_color='lightgrey',
                                   key='-P2_MODEL_FOLDER-', disabled=True, size=(14, 1))],
                         [TextLabel('trials'),
                          sg.Input(default_text=settings['-P1_TRIALS-'], size=(7, 1), enable_events=True,
                                   key='-P1_TRIALS-'), Pad(2 + OFFSET),
                          sg.Input(default_text=settings['-P2_TRIALS-'], size=(7, 1), enable_events=True,
                                   key='-P2_TRIALS-'), Pad(2 + OFFSET)],
                         [sg.HSeparator()],
                         [TextLabel('progress'), sg.ProgressBar(
                             1000, orientation='h', size=(10, 20), key='-PROGRESSBAR-')],
                         [TextLabel('time'), TextOutput('-ETA-', 15)],
                         [TextLabel('current best'), TextOutput(
                             '-CURRENT_BEST-', 15)],
                         [sg.HSeparator()],
                         [TextLabel('eval')],
                         [sg.HSeparator()],
                         # [sg.Output(size=(13, 20), key='-HISTORY-')],
                         [sg.Text('history:', font="Any 10")]], vertical_alignment='top')

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

applySettings(window, board, settings, game)
# window.refresh()


thinker = [None, None]

#thinker[1] = st.loadModel(1, settings)
#thinker[0] = st.loadModel(2, settings)


# Event Loop
while True:
    updateTurnIndicator(window, game)
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Settings...':
        settingsDialog()
    elif event == '-BOARD-':
        boardEvent()
    elif event == 'Reset':
        game = twixt.Game(settings['-ALLOW_SWAP-'], settings['-ALLOW_SCL-'])
        board.draw(game)


window.close()
