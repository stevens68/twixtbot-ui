import PySimpleGUI as sg
import board
import settings as st

# helper


def TextLabel(text, key=None):
    return sg.Text(text + ':', justification='r', size=(10, 1), key=key)


def TextOutput(key, length):
    return sg.Input(size=(length, 1), disabled_readonly_background_color="lightgrey", readonly=True,
                    key=key)


def getPegIcon(size, key):
    miniSquare = sg.Graph(canvas_size=(size, size),
                          graph_bottom_left=(0, 0),
                          graph_top_right=(size, size),
                          background_color=sg.theme_background_color(),
                          key=key,
                          enable_events=False)
    return miniSquare


def applySettings(window, board, settings):
    board.draw()
    window['-PLAYER1_NAME-'].Update(settings['-PLAYER1_NAME-'] + ':')
    window['-P1PEGSQUARE-'].erase()
    window['-P1PEGSQUARE-'].DrawCircle((7, 9), 6,
                                       settings['-PLAYER1_COLOR-'], settings['-PLAYER1_COLOR-'])
    window['-PLAYER1_TYPE-'].Update(settings['-PLAYER1_TYPE-'])

    window['-PLAYER2_NAME-'].Update(settings['-PLAYER2_NAME-'] + ':')
    window['-P2PEGSQUARE-'].erase()
    window['-P2PEGSQUARE-'].DrawCircle((7, 9), 6,
                                       settings['-PLAYER2_COLOR-'], settings['-PLAYER2_COLOR-'])
    window['-PLAYER2_TYPE-'].Update(settings['-PLAYER2_TYPE-'])

##### main window ##################################


# first, read settings file
settings = st.loadSettings()

# then create a twixt board (draw it later)
board = board.TwixtBoard(settings)

# set the layout
menu_def = [['File', ['Settings...', 'Exit']],
            ['Help', 'About...']]

board_col = sg.Column([[board.graph]])

control_col = sg.Column([[TextLabel(settings['-PLAYER1_NAME-'], key='-PLAYER1_NAME-'),
                          getPegIcon(16, '-P1PEGSQUARE-'),
                          sg.Text('', size=(5, 1), key='-PLAYER1_TYPE-'),
                          sg.Text('', key='-P1TURNINDICATOR-')],
                         [TextLabel(settings['-PLAYER2_NAME-'], key='-PLAYER2_NAME-'),
                          getPegIcon(16, "-P2PEGSQUARE-"),
                          sg.Text('', size=(5, 1), key='-PLAYER2_TYPE-'),
                          sg.Text('', key='-P2TURNINDICATOR-')],
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
board.draw()
applySettings(window, board, settings)


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
            applySettings(window, board, settings)
            settings_dialog.close()
        elif event == 'Exit':
            settings_dialog.close()

window.close()
