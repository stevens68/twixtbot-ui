import unittest
from unittest.mock import MagicMock
import src.layout

src.layout.ct = MagicMock(
    K_BOARD_SIZE=(None, 'key', None, 10),
    BOARD_SIZE_LIST=[10, 20, 30],
    MSG_REQUIRES_RESTART='restart',
    K_SHOW_LABELS=(None, 'show_labels', None, None, True),
    K_SHOW_GUIDELINES=(None, 'show_guidelines', None, None, True),
    K_SHOW_CURSOR_LABEL=(None, 'show_cursor_label', None, None, True),
    K_HIGHLIGHT_LAST_MOVE=(None, 'highlight_last_move', None, None, True),
    K_LOG_LEVEL=(None, 'log_level', None, None, 'INFO'),
    LOG_LEVEL_LIST=['INFO', 'DEBUG'],
    TAB_LABEL_GENERAL='General',
    TAB_LABEL_PLAYER1='Player1',
    TAB_LABEL_PLAYER2='Player2',
    B_APPLY_SAVE='Apply',
    B_RESET_DEFAULT='Reset',
    B_CANCEL='Cancel',
    B_OK='OK',
    K_SPLASH_PROGRESS_BAR=('splash_bar',),
    PROGRESS_BAR_COLOR='blue',
    K_SPLASH_STATUS_TEXT=('splash_status',),
    OUTPUT_TEXT_COLOR='black'
)
src.layout.sg = MagicMock(
    Text=lambda *args, **kwargs: 'text',
    Combo=lambda *args, **kwargs: 'combo',
    Checkbox=lambda *args, **kwargs: 'checkbox',
    Tab=lambda *args, **kwargs: 'tab',
    TabGroup=lambda *args, **kwargs: 'tabgroup',
    Button=lambda *args, **kwargs: 'button',
    ProgressBar=lambda *args, **kwargs: 'progress',
    theme_background_color=lambda: 'white'
)
src.layout.st_row_allow_swap = lambda: ['swap']
src.layout.st_row_allow_scl = lambda: ['scl']
src.layout.row_separator = lambda label: ['sep']
src.layout.st_label = lambda label: 'label'
src.layout.st_row_smart_accept = lambda: ['smart_accept']
src.layout.st_row_resign_threshold = lambda: ['resign_threshold']
src.layout.st_tab_player = lambda player: [['player_tab', player]]

class TestSettingsDialogLayout(unittest.TestCase):
    def test_init_and_layout(self):
        layout = src.layout.SettingsDialogLayout()
        self.assertIsInstance(layout, src.layout.SettingsDialogLayout)
        l = layout.get_layout()
        self.assertIsInstance(l, list)
        self.assertTrue(any('tabgroup' in str(row) for row in l))

class TestAboutDialogLayout(unittest.TestCase):
    def test_init_and_layout(self):
        layout = src.layout.AboutDialogLayout()
        self.assertIsInstance(layout, src.layout.AboutDialogLayout)
        l = layout.get_layout()
        self.assertIsInstance(l, list)
        self.assertTrue(any('button' in str(row) for row in l))

class TestSplashScreenLayout(unittest.TestCase):
    def test_init_and_layout(self):
        layout = src.layout.SplashScreenLayout()
        self.assertIsInstance(layout, src.layout.SplashScreenLayout)
        l = layout.get_layout()
        self.assertIsInstance(l, list)
        self.assertTrue(any('progress' in str(row) for row in l))

if __name__ == '__main__':
    unittest.main()

