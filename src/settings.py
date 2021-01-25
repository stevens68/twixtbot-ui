import PySimpleGUI as sg
from os import path
from json import (load as jsonload, dump as jsondump)


import twixt


BOARD_SIZE = [400, 500, 600, 700, 800]
COLORS = ["black", "blue", "brown", "cyan", "green", "orange", "lightblue",
          "lightgreen", "lightred", "pink", "purple", "red", "yellow", "white"]
DELAY_SEC = [0, .1, .2, .3, .5, 1, 2, 3, 5, 10]
SETTINGS_FILE = path.join(path.dirname(__file__), r'settings.cfg')
MODEL_DIR = path.normpath(path.join(path.dirname(__file__), '..\model\pb'))
TEMPERATURE = [0.0, 0.5, 1.0]
POPUP_COLOR = "orange"


SETTINGS = {
    '-ALLOW_SWAP-': [True, 'allow swap'],
    '-ALLOW_SCL-': [False, 'allow self-crossing links'],
    '-DELAY_SEC-': [DELAY_SEC[4], 'auto move delay [s]'],

    '-P1_NAME-': ['Tom', 'name'],
    '-P1_COLOR-': ['red', 'color'],
    '-P1_AUTO_MOVE-': [False, 'auto move'],
    '-P1_MODEL_FOLDER-': [MODEL_DIR, 'model folder'],
    '-P1_RANDOM_ROTATION-': [False, 'random rotation'],
    '-P1_TRIALS-': [0, 'trials'],
    '-P1_NOISE-': [False, 'add_noise'],
    '-P1_TEMPERATURE-': [TEMPERATURE[1], 'temperature'],
    '-P1_SMART_ROOT-': [False, 'smart root'],

    '-P2_NAME-': ['Jerry', 'name'],
    '-P2_COLOR-': ['black', 'color'],
    '-P2_AUTO_MOVE-': [True, 'auto move'],
    '-P2_MODEL_FOLDER-': [MODEL_DIR, 'model folder'],
    '-P2_RANDOM_ROTATION-': [False, 'random rotation'],
    '-P2_TRIALS-': [0, 'trials'],
    '-P2_NOISE-': [False, 'add_noise'],
    '-P2_TEMPERATURE-': [TEMPERATURE[1], 'temperature'],
    '-P2_SMART_ROOT-': [False, 'smart root'],

    '-SHOW_LABELS-': [True, 'show labels'],
    '-SHOW_GUIDELINES-': [False, 'show guidelines'],
    #'-THEME-': ['DarkBlue3', 'theme'],
    '-BOARD_SIZE-': [BOARD_SIZE[3], 'board size'],
}

# constants
THINKER_ARGS = 'nnmplayer:trials=100,verbosity=0,model=model/pb'


def load_model(player, settings):
    p = str(player)
    sg.popup_quick_message('Loading model ... ', keep_on_top=True, line_width=200,
                           background_color=POPUP_COLOR)
    thinkerArgs = 'nnmplayer:trials=' + \
        str(settings['-P' + p + '_TRIALS-'])
    thinkerArgs += ',verbosity=0,model=' + \
        settings['-P' + p + '_MODEL_FOLDER-']
    return twixt.get_thinker(thinkerArgs, dict())


def load_settings():
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = jsonload(f)
    except Exception:
        sg.popup('No settings file found.\nCreating settings.cfg with default settings.',
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
        jsondump(settings, f)

    settings['-BOARD_SIZE-'] = old_board_size or settings['-BOARD_SIZE-']
    #sg.popup('Settings saved')
    return settings


def create_settings_window(settings):

    def text_label(text):
        if len(text) > 0:
            return sg.Text(text + ':', justification='r', size=(18, 1))
        else:
            return sg.Text('', justification='r', size=(18, 1))

    tab_general = [[sg.Text("")], [text_label(SETTINGS['-ALLOW_SWAP-'][1]),
                                   sg.Checkbox(text=None, default=SETTINGS['-ALLOW_SWAP-'][0], key='-ALLOW_SWAP-')],
                   [text_label(SETTINGS['-ALLOW_SCL-'][1]),
                    sg.Checkbox(text=None, default=SETTINGS['-ALLOW_SCL-'][0], key='-ALLOW_SCL-')],
                   [text_label(SETTINGS['-DELAY_SEC-'][1]),
                    sg.Combo(DELAY_SEC, settings['-DELAY_SEC-'], key='-DELAY_SEC-', size=(4, 1))],
                   ]
    tab_player1 = [[sg.Text("")], [text_label(SETTINGS['-P1_COLOR-'][1]),
                                   sg.Combo(COLORS, SETTINGS['-P1_COLOR-'][0], size=(15, 1), key='-P1_COLOR-', readonly=True)],
                   [text_label(SETTINGS['-P1_NAME-'][1]),
                    sg.Input(SETTINGS['-P1_NAME-'][0], size=(15, 1), key='-P1_NAME-')],
                   [text_label(SETTINGS['-P1_AUTO_MOVE-'][1]),
                    sg.Checkbox(text=None, default=SETTINGS['-P1_AUTO_MOVE-'][0], key='-P1_AUTO_MOVE-')],
                   [text_label(SETTINGS['-P1_MODEL_FOLDER-'][1]), sg.Input(key='-P1_MODEL_FOLDER-'),
                    sg.FolderBrowse(target='-P1_MODEL_FOLDER-',
                                    initial_folder=MODEL_DIR),
                    sg.Button('Load', key='-P1_LOAD-')],
                   [text_label(SETTINGS['-P1_TRIALS-'][1]),
                    sg.Input(SETTINGS['-P1_TRIALS-'][0], size=(10, 1), key='-P1_TRIALS-')],
                   [text_label('temperature'),
                    sg.Combo(TEMPERATURE, settings['-P1_TEMPERATURE-'], size=(5, 1), key='-P1_TEMPERATURE-', readonly=True)],
                   [text_label('noise'),
                    sg.Input(default_text=settings['-P1_NOISE-'], size=(7, 1), enable_events=True, key='-P1_NOISE-')],
                   [text_label(SETTINGS['-P1_RANDOM_ROTATION-'][1]),
                    sg.Checkbox(text=None, default=SETTINGS['-P1_RANDOM_ROTATION-'][0], key='-P1_RANDOM_ROTATION-')],
                   [text_label('smart root'),
                    sg.Checkbox(
                       text="", default=settings['-P1_SMART_ROOT-'], key='-P1_SMART_ROOT-', size=(7, 1))]
                   ]

    tab_player2 = [[sg.Text("")], [text_label(SETTINGS['-P2_COLOR-'][1]),
                                   sg.Combo(COLORS, SETTINGS['-P2_COLOR-'][0], size=(15, 1), key='-P2_COLOR-', readonly=True)],
                   [text_label(SETTINGS['-P2_NAME-'][1]),
                    sg.Input(SETTINGS['-P2_NAME-'][0], size=(15, 1), key='-P2_NAME-')],
                   [text_label(SETTINGS['-P2_AUTO_MOVE-'][1]),
                    sg.Checkbox(text=None, default=SETTINGS['-P2_AUTO_MOVE-'][0], key='-P2_AUTO_MOVE-')],
                   [text_label(SETTINGS['-P2_MODEL_FOLDER-'][1]), sg.Input(key='-P2_MODEL_FOLDER-'),
                    sg.FolderBrowse(target='-P2_MODEL_FOLDER-',
                                    initial_folder=MODEL_DIR),
                    sg.Button('Load', key='-P2_LOAD-')],
                   [text_label(SETTINGS['-P2_TRIALS-'][1]),
                    sg.Input(SETTINGS['-P2_TRIALS-'][0], size=(10, 1), key='-P2_TRIALS-')],
                   [text_label('temperature'),
                    sg.Combo(TEMPERATURE, settings['-P2_TEMPERATURE-'], size=(5, 1), key='-P2_TEMPERATURE-', readonly=True)],
                   [text_label('noise'),
                    sg.Input(default_text=settings['-P2_NOISE-'], size=(7, 1), enable_events=True, key='-P2_NOISE-')],
                   [text_label(SETTINGS['-P1_RANDOM_ROTATION-'][1]),
                    sg.Checkbox(text=None, default=SETTINGS['-P2_RANDOM_ROTATION-'][0], key='-P2_RANDOM_ROTATION-')],
                   [text_label('smart root'),
                    sg.Checkbox(
                       text="", default=settings['-P2_SMART_ROOT-'], key='-P2_SMART_ROOT-', size=(7, 1))]
                   ]

    tab_appearence = [  # [text_label(SETTINGS['-THEME-'][1]),
        # sg.Combo(sg.theme_list(), size=(15, 1), key='-THEME-'),
        # sg.Text('requires restart')],
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
                    sg.Tab('Player 1', tab_player1),
                    sg.Tab('Player 2', tab_player2),
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
