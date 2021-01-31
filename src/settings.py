import PySimpleGUI as sg
from json import (load as jsonload, dump as jsondump)
from backend import twixt
from os import path
from backend import nnmplayer

OFFSET = 5
RO_BACKGROUND = "slate grey"
RO_TEXT = "white"


BOARD_SIZE = [400, 500, 600, 700, 800]
COLORS = ["black", "blue", "brown", "cyan", "green", "orange", "lightblue",
          "lightgreen", "lightred", "pink", "purple", "red", "yellow", "white"]
DELAY_SEC = [0, .1, .2, .3, .5, 1, 2, 3, 5, 10]
SETTINGS_FILE = path.join(path.dirname(__file__), r'config.json')
MODEL_DIR = path.normpath(path.join(path.dirname(__file__), '..\model\pb'))
TEMPERATURE = [0.0, 0.5, 1.0]


SETTINGS = {
    '-ALLOW_SWAP-': [True, 'allow swap'],
    '-ALLOW_SCL-': [False, 'allow self-crossing links'],

    # PLAYER 1

    '-P1_NAME-': ['Tom', 'name'],
    '-P1_COLOR-': ['red', 'color'],
    '-P1_AUTO_MOVE-': [False, 'auto move'],

    # MODEL 1
    '-P1_MODEL_FOLDER-': [MODEL_DIR, 'model folder'],
    '-P1_RANDOM_ROTATION-': [False, 'random rotation'],
    '-P1_TEMPERATURE-': [TEMPERATURE[1], 'temperature'],
    '-P1_EVAL_BEFORE-': [True, 'eval before turn'],

    # MCTS 1
    '-P1_TRIALS-': [0, 'trials'],
    '-P1_CPUCT-': [1.0, 'cpuct'],
    '-P1_ADD_NOISE-': [0.25, 'add noise'],

    # PLAYER 2

    '-P2_NAME-': ['Jerry', 'name'],
    '-P2_COLOR-': ['black', 'color'],
    '-P2_AUTO_MOVE-': [True, 'auto move'],

    # MODEL 2
    '-P2_MODEL_FOLDER-': [MODEL_DIR, 'model folder'],
    '-P2_RANDOM_ROTATION-': [False, 'random rotation'],
    '-P2_TEMPERATURE-': [TEMPERATURE[0], 'temperature'],
    '-P2_EVAL_BEFORE-': [True, 'eval before turn'],

    # MCTS 2
    '-P2_TRIALS-': [0, 'trials'],
    '-P2_CPUCT-': [1.0, 'cpuct'],
    '-P2_ADD_NOISE-': [0.25, 'add noise'],

    # APPEARANCE
    '-SHOW_LABELS-': [True, 'show labels'],
    '-SHOW_GUIDELINES-': [False, 'show guidelines'],
    '-BOARD_SIZE-': [BOARD_SIZE[3], 'board size (pixels)'],
}


def text_label(text, key=None):
    label = text + ':' if len(text) > 0 else text
    return sg.Text(label, justification='r', size=(18, 1), pad=(20, 0), key=key)


def text_field(text, key=None):
    return sg.Text(text, justification='l', size=(10 + OFFSET, 1), key=key)


def text_output(key, length):
    return sg.Input(size=(length, 1), disabled_readonly_background_color=RO_BACKGROUND, text_color=RO_TEXT, readonly=True,
                    key=key)


def pad(s):
    return sg.Text("", size=(s, 1))


def row_separator(text):
    return sg.Text(text, font=("Arial", 9, "italic")), sg.HSeparator(pad=(0, 20))


def init_bot(player, settings):
    p = str(player)

    args = {
        "allow_swap": settings['-ALLOW_SWAP-'],
        "model": settings['-P' + p + '_MODEL_FOLDER-'],
        "trials": settings['-P' + p + '_TRIALS-'],
        "temperature": settings['-P' + p + '_TEMPERATURE-'],
        "random_rotation": settings['-P' + p + '_RANDOM_ROTATION-'],
        "add_noise": settings['-P' + p + '_ADD_NOISE-'],
    }

    return nnmplayer.Player(**args)


def load_settings():
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = jsonload(f)
    except Exception:
        sg.popup('No settings file found.\nCreating ' + SETTINGS_FILE + ' with default settings.',
                 keep_on_top=True)

        settings = {}
        for key in SETTINGS:
            settings[key] = SETTINGS[key][0]
        save_settings(settings, None)
    return settings


def save_settings(settings, values):

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
        jsondump(settings, f, indent=4, sort_keys=True)

    settings['-BOARD_SIZE-'] = old_board_size or settings['-BOARD_SIZE-']
    #sg.popup('Settings saved')
    return settings


def row_allow_swap():
    key = '-ALLOW_SWAP-'
    return [text_label(SETTINGS[key][1]), sg.Checkbox(text=None, default=SETTINGS[key][0], key=key)]


def row_allow_scl():
    key = '-ALLOW_SCL-'
    return [text_label(SETTINGS[key][1]), sg.Checkbox(text=None, default=SETTINGS[key][0], key=key)]


def row_player_color(player):
    key = '-P' + str(player) + '_COLOR-'
    return [text_label(SETTINGS[key][1]), sg.Combo(COLORS, SETTINGS[key][0], size=(15, 1), key=key, readonly=True)]


def row_player_name(player):
    key = '-P' + str(player) + '_NAME-'
    return [text_label(SETTINGS[key][1]), sg.Input(SETTINGS[key][0], size=(15, 1), key=key)]


def row_auto_move(player):
    key = '-P' + str(player) + '_AUTO_MOVE-'
    return [text_label(SETTINGS[key][1]), sg.Checkbox(text=None, default=SETTINGS[key][0], key=key)]


def row_eval_before(player):
    key = '-P' + str(player) + '_EVAL_BEFORE-'
    return [text_label(SETTINGS[key][1]), sg.Checkbox(text=None, default=SETTINGS[key][0], key=key)]


def row_model_folder(player):
    key = '-P' + str(player) + '_MODEL_FOLDER-'
    return [text_label(SETTINGS[key][1]), sg.Input(key=key),
            sg.FolderBrowse(target=key, initial_folder=MODEL_DIR), sg.Text('requires restart')]


def row_trials(player):
    key = '-P' + str(player) + '_TRIALS-'
    return [text_label(SETTINGS[key][1]), sg.Input(SETTINGS[key][0], size=(10, 1), key=key)]


def row_temperature(player):
    key = '-P' + str(player) + '_TEMPERATURE-'
    return [text_label(SETTINGS[key][1]),
            sg.Combo(TEMPERATURE, SETTINGS[key], size=(5, 1), key=key, readonly=True)]


def row_add_noise(player):
    key = '-P' + str(player) + '_ADD_NOISE-'
    return [text_label(SETTINGS[key][1]),
            sg.Spin(values=[float((x + 1) / 100) for x in range(100)], initial_value=SETTINGS[key], key=key, size=(5, 0))]


def row_cpuct(player):
    key = '-P' + str(player) + '_CPUCT-'
    return [text_label(SETTINGS[key][1]),
            sg.Spin(values=[float((x + 1) / 100) for x in range(100)], initial_value=SETTINGS[key], key=key, size=(5, 0))]


def row_random_rotation(player):
    key = '-P' + str(player) + '_RANDOM_ROTATION-'
    return [text_label(SETTINGS[key][1]), sg.Checkbox(text=None, default=SETTINGS[key][0], key=key)]


def tab_player(player):
    return [[sg.Text("")],
            row_player_color(player),
            row_player_name(player),
            row_auto_move(player),
            row_eval_before(player),
            row_model_folder(player),
            row_trials(player),
            row_temperature(player),
            row_add_noise(player),
            row_random_rotation(player),
            row_cpuct(player),
            [sg.Text("")]
            ]


def create_settings_window(settings):

    def text_label(text):
        if len(text) > 0:
            return sg.Text(text + ':', justification='r', size=(18, 1))
        else:
            return sg.Text('', justification='r', size=(18, 1))

    tab_general = [[sg.Text("")],
                   row_allow_swap(),
                   row_allow_scl()
                   ]

    tab_player1 = tab_player(1)
    tab_player2 = tab_player(2)

    tab_appearence = [
        [sg.Text("")],
        [text_label(SETTINGS['-BOARD_SIZE-'][1]),
         sg.Combo(BOARD_SIZE, SETTINGS['-BOARD_SIZE-'][0], size=(15, 1), key='-BOARD_SIZE-', readonly=True), sg.Text('requires restart')],
        [text_label(SETTINGS['-SHOW_LABELS-'][1]),
         sg.Checkbox(text=None, default=True, key='-SHOW_LABELS-')],
        [text_label(SETTINGS['-SHOW_GUIDELINES-'][1]),
         sg.Checkbox(text=None, default=True, key='-SHOW_GUIDELINES-')]
    ]

    layout = [
        [sg.TabGroup(
            [
                [
                    sg.Tab('General', tab_general),
                    sg.Tab('Player 1', tab_player1, pad=(20, 0)),
                    sg.Tab('Player 2', tab_player2, pad=(10, 0)),
                    sg.Tab('Appearance', tab_appearence)
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
