import pathlib
import logging
from os import path


TURN_COLOR = "white"
PROGRESS_BAR_COLOR = "lightgreen"


EVAL_PLOT_COLOR = "purple"
VISITS_PLOT_COLOR = "blue"

PEG_HOLE_COLOR = "grey"
BOARD_LABEL_COLOR = "black"
BOARD_LABEL_FONT = ("Arial", 9)
CURSOR_LABEL_BACKGROUND_COLOR = "white"
GUIDELINE_COLOR = "#b3b3b3"
PLOT_LABEL_COLOR = "black"
PLOT_LABEL_FONT = ("Arial", 9)
VISITS_LABEL_FONT = ("Arial", 8)
OUTPUT_BACKGROUND_COLOR = "slate grey"
OUTPUT_TEXT_COLOR = "white"
COLOR_LIST = ["black", "blue", "cyan", "orange",
              "lightgreen", "purple", "red", "yellow", "white"]
FIELD_BACKGROUND_COLOR = "lightgrey"
HIGHLIGHT_LAST_MOVE_COLOR = 'yellow'


HEATMAP_CIRCLE_COLOR = 'black'
HEATMAP_RADIUS_FACTOR = 2
HEATMAP_CIRCLE_FACTOR = 1.5
HEATMAP_LEGEND_STEPS = 10
HEATMAP_RGB_COLORS = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]  # min/mid/max

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

ROT_OFF = "off"
ROT_RAND = "random"
ROT_AVG = "average"
ROT_FLIP_HOR = "flip horizontally"
ROT_FLIP_VERT = "flip vertically"
ROT_FLIP_BOTH = "flip both"

ROTATION_LIST = [ROT_OFF, ROT_RAND, ROT_AVG, ROT_FLIP_HOR, ROT_FLIP_VERT, ROT_FLIP_BOTH]

# logging
LOG_LEVEL_LIST = list(map(logging.getLevelName, range(10, 60, 10)))

# keys - player 1/2
K_COLOR = ['color', 'P1_COLOR', 'P2_COLOR', "red", "black"]
K_NAME = ['name', 'P1_NAME', 'P2_NAME', "Tom", "Jerry"]
K_AUTO_MOVE = ['auto move', 'P1_AUTO_MOVE', 'P2_AUTO_MOVE', False, False]
K_MODEL_FOLDER = ['model folder', "P1_MODEL_FOLDER",
                  "P2_MODEL_FOLDER", "../model/pb", "../model/pb"]
K_RANDOM_ROTATION = ['random rotation',
                     'P1_RANDOM_ROTATION', 'P2_RANDOM_ROTATION', False, False]
K_TRIALS = ['trials', 'P1_TRIALS', 'P2_TRIALS', 0, 0]
K_SMART_ROOT = ['smart root', 'P1_SMART_ROOT',
                'P2_SMART_ROOT', False, False]
K_TEMPERATURE = ['temperature', 'P1_TEMPERATURE', 'P2_TEMPERATURE', 0.0, 0.0]
K_CPUCT = ['cpuct', 'P1_CPUCT', 'P2_CPUCT', 1.0, 1.0]
K_ROTATION = ['rotation', 'P1_ROTATION', 'P2_ROTATION', ROT_OFF, ROT_OFF]
K_ADD_NOISE = ['add noise', 'P1_ADD_NOISE', 'P2_ADD_NOISE', 0.0, 0.0]

# keys - general
K_ALLOW_SWAP = ['allow swap', 'ALLOW_SWAP', None, True]
K_ALLOW_SCL = ['allow self crossing links', 'ALLOW_SCL', None, False]
K_BOARD_SIZE = ['board size (pixels)', 'BOARD_SIZE', None, 600]
K_SHOW_LABELS = ['show labels', 'SHOW_LABELS', None, True]
K_SHOW_GUIDELINES = ['show guidelines', 'SHOW_GUIDELINES', None, False]
K_SHOW_CURSOR_LABEL = ['show cursor label', 'SHOW_CURSOR_LABEL', None, False]
K_HIGHLIGHT_LAST_MOVE = ['highlight last move',
                         'HIGHLIGHT_LAST_MOVE', None, False]
K_SMART_ACCEPT = ['smart accept', 'SMART_ACCEPT', None, True]
K_RESIGN_THRESHOLD = ['resign threshold', 'RESIGN_THRESHOLD', None, 0.95]

K_LOG_LEVEL = ['Log level', 'LOG_LEVEL', None,
               logging.getLevelName(logging.ERROR)]

# keys - non-setting
K_BOARD = [None, 'BOARD']
K_EVAL_BAR = ['', 'EVAL_BAR', None, 0]
K_TURN_INDICATOR = ['turn', 'P1_TURN_INDICATOR',
                    'P2_TURN_INDICATOR', TURN_CHAR, ""]
K_MOVES = ['moves', 'MOVES']
K_SHOW_EVALUATION = ['', 'SHOW_EVALUATION', None, True]
K_EVAL_NUM = ['', 'EVAL_NUM', None, ""]
K_EVAL_HIST = ['history', 'EVAL_HIST']
K_EVAL_MOVES = ['P * 1000', 'EVAL_MOVES', None, ""]
K_HEATMAP = ['heatmap', 'HEATMAP']

K_VISITS = ['visits', 'VISITS']
K_VISUALIZE_MCTS = ['visualize', 'VISUALIZE_MCTS', None, False]
K_PROGRESS_BAR = ['progress', 'PROGRESS_BAR']
K_PROGRESS_NUM = ['', 'PROGRESS_NUMBERS']
K_SPINNER = ['', 'SPINNER']

K_SPLASH_PROGRESS_BAR = ['SPLASH_PROGRESS_BAR']
K_SPLASH_STATUS_TEXT = ['SPLASH_STATUS_TEXT']
K_THREAD = [None, 'THREAD']


SETTING_KEYS = [K_ALLOW_SWAP, K_ALLOW_SCL, K_SMART_ACCEPT,
                K_COLOR, K_NAME, K_AUTO_MOVE, K_TRIALS, K_MODEL_FOLDER,
                K_TEMPERATURE, K_CPUCT, K_ADD_NOISE, K_ROTATION,
                K_BOARD_SIZE, K_LOG_LEVEL, K_SMART_ROOT, K_RESIGN_THRESHOLD,
                K_SHOW_LABELS, K_SHOW_GUIDELINES, K_SHOW_CURSOR_LABEL,
                K_HIGHLIGHT_LAST_MOVE]


WINDOW_TITLE = 'twixtbot-ui'
SETTINGS_DIALOG_TITLE = "Settings"
ABOUT_DIALOG_TITLE = "About"

MSG_REQUIRES_RESTART = "restart required"
MSG_NO_CONFIG_FILE = (f'No settings file found.\nCreating '
                      f'{SETINGS_FILE_NAME} with default settings.')
MSG_ERROR_UPDATING_KEY = ('Problem updating settings from window values. '
                          'key=%s, exc=%s')

ITEM_FILE = "&File"
ITEM_OPEN_FILE = "&Open File..."
ITEM_SAVE_FILE = "&Save File..."
ITEM_SETTINGS = "Se&ttings..."
ITEM_EXIT = "E&xit"
ITEM_HELP = "&Help"
ITEM_ABOUT = "&About..."

B_BOT_MOVE = "Bot Move"
B_ACCEPT = "Accept"
B_CANCEL = "Cancel"
B_UNDO = "Undo"
B_REDO = "Redo"
B_RESIGN = "Resign"
B_RESET = "Reset"
B_APPLY_SAVE = 'Apply & Save'
B_RESET_DEFAULT = 'Reset to default'
B_OK = "OK"

TAB_LABEL_GENERAL = "General"
TAB_LABEL_PLAYER1 = "Player 1"
TAB_LABEL_PLAYER2 = "Player 2"

# shortcut events for checkboxes
EVENT_SHORTCUT_HEATMAP = 'SHORTCUT_HEATMAP'
EVENT_SHORTCUT_SHOW_EVALUATION = 'SHORTCUT_SHOW_EVALUATION'
EVENT_SHORTCUT_VISUALIZE_MCTS = 'SHORTCUT_VISUALIZE_MCTS'
EVENT_SHORTCUT_AUTOMOVE_1 = 'SHORTCUT_AUTOMOVE_1'
EVENT_SHORTCUT_AUTOMOVE_2 = 'SHORTCUT_AUTOMOVE_2'
EVENT_SHORTCUT_TRIALS_1_PLUS = 'SHORTCUT_TRIALS_1_PLUS'
EVENT_SHORTCUT_TRIALS_1_MINUS = 'SHORTCUT_TRIALS_1_MINUS'
EVENT_SHORTCUT_TRIALS_2_PLUS = 'SHORTCUT_TRIALS_2_PLUS'
EVENT_SHORTCUT_TRIALS_2_MINUS = 'SHORTCUT_TRIALS_2_MINUS'

EVENT_EXIT = "Exit"

# Logging
LOGGER = 'twixtbot-ui'
LOG_FORMAT = ('[%(levelname)s] [%(asctime)s] [%(filename)s:(%(lineno)d] '
              '%(message)s')

# MCTS
MCTS_TRIAL_CHUNK = 20
