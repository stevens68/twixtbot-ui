import unittest
from unittest.mock import MagicMock
import src.layout


class DummyBoard:
    graph = "dummy_graph"


class DummyStgs:
    def get(self, key):
        return 100


src.layout.ct = MagicMock(
    K_COLOR=(None, "color1", "color2"),
    K_BOARD_SIZE=(None, "key", None, 10),
    K_EVAL_BAR=(None, "eval_bar"),
    ITEM_FILE="File",
    ITEM_OPEN_FILE="Open",
    ITEM_SAVE_FILE="Save",
    ITEM_SETTINGS="Settings",
    ITEM_EXIT="Exit",
    ITEM_HELP="Help",
    ITEM_ABOUT="About",
    B_BOT_MOVE="BotMove",
    B_ACCEPT="Accept",
    B_CANCEL="Cancel",
    B_UNDO="Undo",
    B_REDO="Redo",
    B_RESIGN="Resign",
    B_RESET="Reset",
)
src.layout.sg = MagicMock(
    Text=lambda *args, **kwargs: "text",
    Window=lambda *args, **kwargs: MagicMock(
        get_size=lambda: (10, 20),
        close=lambda: None,
        __getitem__=lambda self, k: MagicMock(get_size=lambda: (10, 20)),
    ),
    ProgressBar=lambda *args, **kwargs: "progress",
    Button=lambda *args, **kwargs: "button",
    Column=lambda *args, **kwargs: "column",
    Menu=lambda *args, **kwargs: "menu",
    Checkbox=lambda *args, **kwargs: "checkbox",
)


def row_colors():
    return "row_colors"


def row_names():
    return "row_names"


def row_turn_indicators():
    return "row_turn_indicators"


def row_auto_moves():
    return "row_auto_moves"


def row_moves():
    return "row_moves"


def row_separator(label, bold):
    return "row_separator"


def row_eval_show_num():
    return "row_eval_show_num"


def row_eval_hist():
    return "row_eval_hist"


def row_eval_moves():
    return "row_eval_moves"


def row_heatmap():
    return "row_heatmap"


def row_trials():
    return "row_trials"


def row_visits():
    return "row_visits"


def row_visualize_mcts():
    return "row_visualize_mcts"


def row_progress_bar():
    return "row_progress_bar"


def row_progress_nums():
    return "row_progress_nums"


src.layout.row_colors = row_colors
src.layout.row_names = row_names
src.layout.row_turn_indicators = row_turn_indicators
src.layout.row_auto_moves = row_auto_moves
src.layout.row_moves = row_moves
src.layout.row_separator = row_separator
src.layout.row_eval_show_num = row_eval_show_num
src.layout.row_eval_hist = row_eval_hist
src.layout.row_eval_moves = row_eval_moves
src.layout.row_heatmap = row_heatmap
src.layout.row_trials = row_trials
src.layout.row_visits = row_visits
src.layout.row_visualize_mcts = row_visualize_mcts
src.layout.row_progress_bar = row_progress_bar
src.layout.row_progress_nums = row_progress_nums


class TestMainWindowLayout(unittest.TestCase):
    def test_init_and_layout(self):
        layout = src.layout.MainWindowLayout(DummyBoard(), DummyStgs())
        self.assertIsInstance(layout, src.layout.MainWindowLayout)
        l = layout.get_layout()
        self.assertIsInstance(l, list)
        self.assertTrue(any("column" in str(row) for row in l))


if __name__ == "__main__":
    unittest.main()
