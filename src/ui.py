import PySimpleGUI as sg
import board
import settings as st
import twixt


# constants
THINKER_ARGS = 'nnmplayer:trials=100,verbosity=0,model=model/pb,location=C:\temp\loc'

# helper


def TextLabel(text, key=None):
    return sg.Text(text + ':', justification='r', size=(10, 1), key=key)


def TextField(text, key=None):
    return sg.Text(text, justification='l', size=(10, 1), key=key)


def TextOutput(key, length):
    return sg.Input(size=(length, 1), disabled_readonly_background_color="lightgrey", readonly=True,
                    key=key)


def getPegIcon(size, key):
    miniSquare = sg.Graph(canvas_size=(size, size),
                          graph_bottom_left=(0, 0),
                          graph_top_right=(size, size),
                          background_color=sg.theme_background_color(),
                          key=key,
                          pad=((6, 72), (0, 0)),
                          enable_events=False)
    return miniSquare


def point_to_move(point):
    return chr(ord('a') + point[0]) + "%d" % (twixt.Game.SIZE - point[1])


def replayGame(game):
    for p in game.history:
        if isinstance(p, twixt.Point):
            game.play(point_to_move(p))
        else:
            # swap case
            game.play(p)


def applySettings(window, board, settings, game):

    board.draw()
    replayGame(game)

    window['-PLAYER1_NAME-'].Update(settings['-PLAYER1_NAME-'])
    window['-P1PEGSQUARE-'].erase()
    window['-P1PEGSQUARE-'].DrawCircle((7, 9), 6,
                                       settings['-PLAYER1_COLOR-'], settings['-PLAYER1_COLOR-'])
    window['-PLAYER1_TYPE-'].Update(settings['-PLAYER1_TYPE-'])

    window['-PLAYER2_NAME-'].Update(settings['-PLAYER2_NAME-'])
    window['-P2PEGSQUARE-'].erase()
    window['-P2PEGSQUARE-'].DrawCircle((7, 9), 6,
                                       settings['-PLAYER2_COLOR-'], settings['-PLAYER2_COLOR-'])
    window['-PLAYER2_TYPE-'].Update(settings['-PLAYER2_TYPE-'])

    if game.turn == 1:
        turn = ['*', '']
    else:
        turn = ['', '*']

    window['-P1TURNINDICATOR-'].Update(turn[0])
    window['-P2TURNINDICATOR-'].Update(turn[1])


##### main window ##################################



# first, read settings file
settings = st.loadSettings()

# then create a twixt board (draw it later)
board = board.TwixtBoard(settings)

# set the layout
menu_def = [['File', ['Settings...', 'Exit']],
            ['Help', 'About...']]

board_col = sg.Column([[board.graph]])

control_col = sg.Column([[TextLabel('turn'),
                          sg.Text('', key='-P1TURNINDICATOR-',
                                  text_color="yellow"),
                          sg.Text('', key='-P2TURNINDICATOR-', text_color="yellow")],
                         [TextLabel('color'),
                          getPegIcon(16, '-P1PEGSQUARE-'),
                          getPegIcon(16, "-P2PEGSQUARE-")],
                         [TextLabel('player'),
                          TextField(settings['-PLAYER1_NAME-'],
                                    key='-PLAYER1_NAME-'),
                          TextField(settings['-PLAYER2_NAME-'], key='-PLAYER2_NAME-')],
                         [TextLabel('type'),
                          TextField(settings['-PLAYER1_TYPE-'],
                                    key='-PLAYER1_TYPE-'),
                          TextField(settings['-PLAYER2_TYPE-'], key='-PLAYER2_TYPE-')],
                         [sg.HSeparator()],
                         [TextLabel('progress'), sg.ProgressBar(
                             1000, orientation='h', size=(10, 20), key='-PROGRESSBAR-')],
                         [TextLabel('trials'), TextOutput('-TRIALS-', 15)],
                         [TextLabel('eta'), TextOutput('-ETA-', 15)],
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
        sg.Button('New Game', size=(10, 1)),
        sg.Button('Stop Bot', size=(10, 1)),
        sg.Button('Undo Move', size=(10, 1)),
        sg.Button('Exit', size=(10, 1))]
]

# create main window
window = sg.Window('twixtbot-ui', layout, margins=(25, 25))
window.Finalize()

# draw twixt board
game = twixt.Game()

applySettings(window, board, settings, game)
window.refresh()


thinker = None
if settings['-LOAD_MODEL_AT_START-']:
    sg.popup_quick_message('Loading model in ' +
                           settings['-MODEL_FOLDER-'] + '...', keep_on_top=True)
    thinker = twixt.get_thinker(THINKER_ARGS, dict())


# Event Loop
while True:
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Settings...':
        settings_dialog = st.createSettingsWindow(settings)
        event, values = settings_dialog.read(close=True)
        if event == 'Apply & Save':
            st.saveSettings(settings, values)
            applySettings(window, board, settings, game)
            settings_dialog.close()
        elif event == 'Exit':
            settings_dialog.close()
    elif event == '-BOARD-':
        move = board.getMove(game, values['-BOARD-'])
        if move:
            print("move: ", move)
            game.play(move)
            board.create_move_objects(game, len(game.history) - 1)

window.close()
