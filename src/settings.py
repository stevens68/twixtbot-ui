import PySimpleGUI as sg
from os import path
from json import (load as jsonload, dump as jsondump)


BOARD_SIZE = [400, 500, 600, 700, 800]
TYPE_HUMAN = 'human'
TYPE_BOT = 'bot'

COLORS = ["black", "blue", "brown", "cyan", "green", "orange", "lightblue",
          "lightgreen", "lightred", "pink", "purple", "red", "yellow", "white"]
SETTINGS_FILE = path.join(path.dirname(__file__), r'settings.cfg')
MODEL_DIR = path.normpath(path.join(path.dirname(__file__), '..\model\pb'))


SETTINGS = {
    '-MODEL_FOLDER-': [MODEL_DIR, 'folder'],
    '-LOAD_MODEL_AT_START-': [True, 'load at start'],
    '-PLAYER1_TYPE-': [TYPE_HUMAN, 'type'],
    '-PLAYER1_NAME-': ['Player 1', 'name'],
    '-PLAYER1_COLOR-': ['red', 'color'],
    '-PLAYER2_TYPE-': [TYPE_BOT, 'type'],
    '-PLAYER2_NAME-': ['Player 2', 'name'],
    '-PLAYER2_COLOR-': ['black', 'color'],
    '-ALLOW_SWAP-': [True, 'allow swap'],
    '-SELF_CROSSING_LINKS-': [False, 'self-crossing links'],
    '-SHOW_LABELS-': [True, 'show labels'],
    '-SHOW_GUIDELINES-': [False, 'show guidelines'],
    '-THEME-': ['DarkBlue3', 'theme'],
    '-BOARD_SIZE-': [BOARD_SIZE[2], 'board size'],
}


def loadSettings():
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = jsonload(f)
    except Exception:
        sg.popup('No settings file found.\nCreating settings.cfg with default settings.',
                 keep_on_top=True)

        settings = {}
        for key in SETTINGS:
            settings[key] = SETTINGS[key][0]
        saveSettings(settings, None)
    return settings


def saveSettings(settings, values):

    old_board_size = None
    if values:      # if there are stuff specified by another window, fill in those values
        for key in SETTINGS:  # update window with the values read from settings file
            try:
                if key == '-BOARD_SIZE-':
                    old_board_size = settings[key]
                settings[key] = values[key]
            except Exception as e:
                print(
                    f'Problem updating settings from window values. Key = {key}: {e}')

    with open(SETTINGS_FILE, 'w') as f:
        jsondump(settings, f)

    settings['-BOARD_SIZE-'] = old_board_size
    #sg.popup('Settings saved')
    return settings


def createSettingsWindow(settings):
    sg.theme(settings['-THEME-'])

    def TextLabel(text):
        if len(text) > 0:
            return sg.Text(text + ':', justification='r', size=(15, 1))
        else:
            return sg.Text('', justification='r', size=(15, 1))

    tab_model = [[TextLabel(SETTINGS['-MODEL_FOLDER-'][1]), sg.Input(key='-MODEL_FOLDER-'),
                  sg.FolderBrowse(target='-MODEL_FOLDER-', initial_folder=MODEL_DIR)],
                 [TextLabel(SETTINGS['-LOAD_MODEL_AT_START-'][1]),
                  sg.Checkbox(text=None, default=True, key='-LOAD_MODEL_AT_START-')],
                 [TextLabel(''), sg.Button('Load', size=(10, 1))]
                 ]
    tab_game = [[TextLabel(SETTINGS['-ALLOW_SWAP-'][1]),
                 sg.Checkbox(text=None, default=True, key='-ALLOW_SWAP-')],
                [TextLabel(SETTINGS['-SELF_CROSSING_LINKS-'][1]),
                    sg.Checkbox(text=None, default=False, key='-SELF_CROSSING_LINKS-')]
                ]
    tab_player1 = [[TextLabel(SETTINGS['-PLAYER1_TYPE-'][1]),
                    sg.Combo([TYPE_HUMAN, TYPE_BOT], TYPE_HUMAN, size=(15, 1), key='-PLAYER1_TYPE-', readonly=True)],
                   [TextLabel(SETTINGS['-PLAYER1_COLOR-'][1]),
                    sg.Combo(COLORS, SETTINGS['-PLAYER1_COLOR-'][0], size=(15, 1), key='-PLAYER1_COLOR-', readonly=True)],
                   [TextLabel(SETTINGS['-PLAYER1_NAME-'][1]),
                    sg.Input(SETTINGS['-PLAYER1_NAME-'][1], size=(15, 1), key='-PLAYER1_NAME-')]
                   ]

    tab_player2 = [[TextLabel(SETTINGS['-PLAYER2_TYPE-'][1]),
                    sg.Combo([TYPE_HUMAN, TYPE_BOT], TYPE_BOT, size=(15, 1), key='-PLAYER2_TYPE-', readonly=True)],
                   [TextLabel(SETTINGS['-PLAYER2_COLOR-'][1]),
                    sg.Combo(COLORS, SETTINGS['-PLAYER2_COLOR-'][0], size=(15, 1), key='-PLAYER2_COLOR-', readonly=True)],
                   [TextLabel(SETTINGS['-PLAYER2_NAME-'][1]),
                    sg.Input(SETTINGS['-PLAYER2_NAME-'][1], size=(15, 1), key='-PLAYER2_NAME-')]
                   ]

    tab_appearence = [
        [TextLabel(SETTINGS['-THEME-'][1]), sg.Combo(
            sg.theme_list(), size=(15, 1), key='-THEME-')],
        [TextLabel(SETTINGS['-BOARD_SIZE-'][1]),
         sg.Combo(BOARD_SIZE, SETTINGS['-BOARD_SIZE-'][0],
                  size=(15, 1), key='-BOARD_SIZE-', readonly=True),
         sg.Text('requires restart')],
        [TextLabel(SETTINGS['-SHOW_LABELS-'][1]),
         sg.Checkbox(text=None, default=True, key='-SHOW_LABELS-')],
        [TextLabel(SETTINGS['-SHOW_GUIDELINES-'][1]),
         sg.Checkbox(text=None, default=True, key='-SHOW_GUIDELINES-')]
    ]

    layout = [
        [sg.TabGroup(
            [
                [
                    sg.Tab('game', tab_game),
                    sg.Tab('player 1', tab_player1),
                    sg.Tab('player 2', tab_player2),
                    sg.Tab('model', tab_model),
                    sg.Tab('appearence', tab_appearence)
                ]
            ]
        )],
        [sg.Button('Apply & Save', size=(12, 1)),
         sg.Button('Exit', size=(10, 1))]
    ]

    window = sg.Window('Settings', layout, keep_on_top=True,
                       finalize=True, margins=(15, 15))

    for key in SETTINGS:   # update window with the values read from settings file
        try:
            window[key].update(
                value=settings[key])

        except Exception as e:
            print(
                f'Problem updating PySimpleGUI window from settings. Key = {key}: {e}')

    return window
