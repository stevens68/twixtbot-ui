import PySimpleGUI as sg
import board
from os import path
from json import (load as jsonload, dump as jsondump)

# Constants
BOARDLEN = 648

PLAYER_CHOICE = ['human', 'bot']
SETTINGS_FILE = path.join(path.dirname(__file__), r'settings.cfg')
MODEL_DIR = path.normpath(path.join(path.dirname(__file__), '..\model\pb'))

##################### Settings #####################

SETTINGS = {
    'MODEL_FOLDER': [MODEL_DIR, 'folder'],
    'LOAD_MODEL_AT_START': [True, 'load at start'],
    'PLAYER1_DEFAULT': [PLAYER_CHOICE[0], 'player1 default'],
    'PLAYER2_DEFAULT': [PLAYER_CHOICE[1], 'player2 default'],
    'ALLOW_SWAP': [True, 'allow swap'],
    'SELF_CROSSING_LINKS': [False, 'self-crossing links'],
    'SHOW_LABELS': [True, 'show labels'],
    'SHOW_GUIDELINES': [False, 'show guidelines'],
    'THEME': ['DarkBlue3', 'theme']
}


def load_settings(settings_file):
    try:
        with open(settings_file, 'r') as f:
            settings = jsonload(f)
    except Exception:
        sg.popup('No settings file found.\nCreating settings.cfg with default settings.',
                 keep_on_top=True)

        settings = {}
        for key in SETTINGS:
            settings[key] = SETTINGS[key][0]
        save_settings(settings_file, settings, None)
    return settings


def save_settings(settings_file, settings, values):
    if values:      # if there are stuff specified by another window, fill in those values
        for key in SETTINGS:  # update window with the values read from settings file
            try:
                window_key = '-' + key + '-'
                settings[key] = values[window_key]
            except Exception as e:
                print(
                    f'Problem updating settings from window values. Key = {window_key}: {e}')

    with open(settings_file, 'w') as f:
        jsondump(settings, f)

    sg.popup('Settings saved')


def create_settings_window(settings):
    sg.theme(settings['THEME'])

    def TextLabel(text):
        return sg.Text(text + ':', justification='r', size=(15, 1))

    layout = [
        [sg.Frame('model',
                  [[TextLabel(SETTINGS['MODEL_FOLDER'][1]),
                    sg.Input(key='-MODEL_FOLDER-'),
                    sg.FolderBrowse(target='-MODEL_FOLDER-', initial_folder=MODEL_DIR)],
                   [TextLabel(SETTINGS['LOAD_MODEL_AT_START'][1]),
                    sg.Checkbox(text=None, default=True, key='-LOAD_MODEL_AT_START-')]
                   ])],
        [sg.Frame('game',
                  [[TextLabel(SETTINGS['PLAYER1_DEFAULT'][1]),
                    sg.Combo(PLAYER_CHOICE, PLAYER_CHOICE[0], size=(15, 1), key='-PLAYER1_DEFAULT-')],
                   [TextLabel(SETTINGS['PLAYER2_DEFAULT'][1]),
                    sg.Combo(PLAYER_CHOICE, PLAYER_CHOICE[1], size=(15, 1), key='-PLAYER2_DEFAULT-')],
                   [TextLabel(SETTINGS['ALLOW_SWAP'][1]),
                    sg.Checkbox(text=None, default=True, key='-ALLOW_SWAP-')],
                   [TextLabel(SETTINGS['SELF_CROSSING_LINKS'][1]),
                    sg.Checkbox(text=None, default=False, key='-SELF_CROSSING_LINKS-')]
                   ])],
        [sg.Frame('board',
                  [[TextLabel(SETTINGS['SHOW_LABELS'][1]),
                    sg.Checkbox(text=None, default=True, key='-SHOW_LABELS-')],
                   [TextLabel(SETTINGS['SHOW_GUIDELINES'][1]),
                    sg.Checkbox(text=None, default=True, key='-SHOW_GUIDELINES-')]
                   ])],
        [sg.Frame('window',
                  [[TextLabel(SETTINGS['THEME'][1]), sg.Combo(
                      sg.theme_list(), size=(15, 1), key='-THEME-')]
                   ])],
        [sg.Button('Save'), sg.Button('Exit')]]

    """
        'ALLOW_SWAP': [True, 'allow swap'],
        'ALLOW_SELF_CROSSING_LINKS': [False, 'allow self-crossing links'],
        'SHOW_LABELS': [True, 'show labels'],
        'SHOW_GUIDELINES': [False, 'show guidelines'],
        'THEME': ['DarkBlue3', 'theme']
    
    """
    window = sg.Window('Settings', layout, keep_on_top=True, finalize=True)

    for key in SETTINGS:   # update window with the values read from settings file
        try:
            window_key = '-' + key + '-'
            window[window_key].update(
                value=settings[key])

        except Exception as e:
            print(
                f'Problem updating PySimpleGUI window from settings. Key = {window_key}: {e}')

    return window

##### main windows ##################################


menu_def = [['File', ['Settings...', 'Exit']],
            ['Help', 'About...']]


# twixt board
graph = sg.Graph(canvas_size=(BOARDLEN, BOARDLEN),
                 graph_bottom_left=(0, 0),
                 graph_top_right=(BOARDLEN, BOARDLEN),
                 background_color='lightgrey',
                 key='board',
                 enable_events=True)

board_col = sg.Column([[graph]
                       ])

player1_col = sg.Column([[sg.Text('player 1')],
                         [sg.Combo(['human', 'bot'], 'human', (15, 1))]], vertical_alignment='top')
player2_col = sg.Column([[sg.Text('player 2')],
                         [sg.Combo(['human', 'bot'], 'bot', (15, 1))]], vertical_alignment='top')

player_col = [player1_col, player2_col]


control_col = sg.Column([[player1_col, player2_col]
                         #, [sg.Output(size=(13, 20), key='-HISTORY-')],
                         ], vertical_alignment='top')

layout = [
    [sg.Menu(menu_def, tearoff=False)],
    [board_col, control_col],
    [

        sg.Button('New Game'),
        sg.Button('Next Move', size=(10, 1)),
        sg.Button('Stop Bot', size=(10, 1)),
        sg.Button('Undo Move', size=(10, 1)),
        sg.Button('Evaluate', size=(10, 1)),
        sg.Button('Exit', size=(10, 1))]
]

window = sg.Window('twixtbot-ui', layout)
window.Finalize()
# draw twixt board
board = board.TwixtBoard(graph)

settings = load_settings(SETTINGS_FILE)
settings_dialog = create_settings_window(settings)

# Event Loop
while True:
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
window.close()
