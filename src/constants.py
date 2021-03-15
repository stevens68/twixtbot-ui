import pathlib
from os import path


TURN_COLOR = "white"
PROGRESS_BAR_COLOR = "lightgreen"


EVAL_PLOT_COLOR = "purple"
VISITS_PLOT_COLOR = "blue"

PEG_HOLE_COLOR = "grey"
BOARD_LABEL_COLOR = "black"
BOARD_LABEL_FONT = ("Arial", 9)
GUIDELINE_COLOR = "#b3b3b3"
PLOT_LABEL_COLOR = "black"
PLOT_LABEL_FONT = ("Arial", 9)
OUTPUT_BACKGROUND_COLOR = "slate grey"
OUTPUT_TEXT_COLOR = "white"
COLOR_LIST = ["black", "blue", "cyan", "orange",
              "lightgreen", "purple", "red", "yellow", "white"]
FIELD_BACKGROUND_COLOR = "lightgrey"
HEATMAP_RADIUS_FACTOR = 0.5


TURN_CHAR = '\u2588'  # block char
TURN_HAS_WON = "has won"
TURN_RESIGNED = "resigned"

OFFSET = 5

# fonts
MOVES_FONT = ("Courier", 10)
SEPARATOR_FONT = ("Arial", 9, "italic")

# files and folders
SETINGS_FILE_NAME = "config.json"
SETTINGS_FILE = path.join(path.dirname(__file__), SETINGS_FILE_NAME)
MODEL_FOLDER = path.normpath(path.join(path.dirname(__file__), '../model/pb'))
SPINNER_IMAGE = str(pathlib.Path(__file__).parent.joinpath(
    r'../img/wheel.gif'))

# limits
TRIALS_MAX = 50000
TRIALS_RESOLUTION = 100

# events
ACCEPT_EVENT = "accepted"
CANCEL_EVENT = "cancelled"

# settings
BOARD_SIZE_LIST = [500, 600, 700, 800]
TEMPERATURE_LIST = [0.0, 0.5, 1.0]

# keys
K_COLOR = ['color', 'P1_COLOR', 'P2_COLOR', "red", "black"]
K_TURN_INDICATOR = ['turn', 'P1_TURN_INDICATOR',
                    'P2_TURN_INDICATOR', TURN_CHAR, ""]
K_NAME = ['name', 'P1_NAME', 'P2_NAME', "Tom", "Jerry"]
K_AUTO_MOVE = ['auto move', 'P1_AUTO_MOVE', 'P2_AUTO_MOVE', False, False]
K_TRIALS = ['MCTS trials', 'P1_TRIALS', 'P2_TRIALS', 0, 0]
K_ALLOW_SWAP = ['allow swap', 'ALLOW_SWAP', None, True]
K_ALLOW_SCL = ['allow self crossing links', 'ALLOW_SCL', None, False]
K_SMART_ACCEPT = ['smart accept', 'SMART_ACCEPT', None, True]

K_TEMPERATURE = ['temperature', 'P1_TEMPERATURE', 'P2_TEMPERATURE', 0.0, 0.0]
K_CPUCT = ['cpuct', 'P1_CPUCT', 'P2_CPUCT', 1.0, 1.0]
K_RANDOM_ROTATION = ['random rotation',
                     'P1_RANDOM_ROTATION', 'P2_RANDOM_ROTATION', False, False]
K_SMART_ROOT = ['smart root', 'P1_SMART_ROOT',
                'P2_SMART_ROOT', False, False]
K_ADD_NOISE = ['add noise', 'P1_ADD_NOISE', 'P2_ADD_NOISE', 0.0, 0.0]
K_BOARD_SIZE = ['board size (pixels)', 'BOARD_SIZE', None, 600]
K_SHOW_LABELS = ['show labels', 'SHOW_LABELS', None, True]
K_SHOW_GUIDELINES = ['show guidelines', 'SHOW_GUIDELINES', None, False]
K_MODEL_FOLDER = ['model folder', "P1_MODEL_FOLDER",
                  "P2_MODEL_FOLDER", "../model/pb", "../model/pb"]


# non-setting keys
K_EVAL_BAR = ['evaluation', 'EVAL_BAR', None, 0]
K_EVAL_NUM = ['', 'EVAL_NUM', None, ""]
K_EVAL_MOVES = ['P * 1000', 'EVAL_MOVES', None, ""]
K_EVAL_HIST = ['history', 'EVAL_HIST']
K_PROGRESS_BAR = ['progress', 'PROGRESS_BAR']
K_PROGRESS_NUM = ['', 'PROGRESS_NUMBERS']
K_SPINNER = ['', 'SPINNER']
K_VISITS = ['MCTS visits', 'VISITS']
K_MOVES = ['moves', 'MOVES']
K_BOARD = [None, 'BOARD']
K_THREAD = [None, 'THREAD']


SETTING_KEYS = [K_ALLOW_SWAP, K_ALLOW_SCL, K_SMART_ACCEPT,
                K_COLOR, K_NAME, K_AUTO_MOVE, K_TRIALS, K_MODEL_FOLDER,
                K_TEMPERATURE, K_CPUCT, K_ADD_NOISE, K_RANDOM_ROTATION,
                K_BOARD_SIZE,
                K_SHOW_LABELS, K_SHOW_GUIDELINES]


WINDOW_TITLE = 'twixtbot-ui'
SETTINGS_DIALOG_TITLE = "Settings"
ABOUT_DIALOG_TITLE = "About"

MSG_REQUIRES_RESTART = "restart required"
MSG_NO_CONFIG_FILE = 'No settings file found.\nCreating ' + \
    SETINGS_FILE_NAME + ' with default settings.'
MSG_ERROR_UPDATING_KEY = 'Problem updating settings from window values. Key = '
MSG_HEATMAP_CALCULATING = "Calculating heatmap (please be patient) ..."

ITEM_FILE = "File"
ITEM_OPEN_FILE = "Open File..."
ITEM_SETTINGS = "Settings..."
ITEM_EXIT = "Exit"
ITEM_HELP = "Help"
ITEM_ABOUT = "About..."

B_BOT_MOVE = "Bot Move"
B_ACCEPT = "Accept"
B_CANCEL = "Cancel"
B_UNDO = "Undo"
B_RESIGN = "Resign"
B_RESET = "Reset"
B_HEATMAP = "Heatmap"


B_APPLY_SAVE = 'Apply & Save'
B_RESET_DEFAULT = 'Reset to default'
B_EXIT = "Exit"

TAB_LABEL_GENERAL = "General"
TAB_LABEL_PLAYER1 = "Player 1"
TAB_LABEL_PLAYER2 = "Player 2"
TAB_LABEL_APPEARANCE = "Appearance"
