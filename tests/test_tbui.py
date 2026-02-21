import unittest
import threading
from src.tbui import BotEvent, ProgressWindow, TwixtbotUI
import src.backend.nnmplayer


class DummyGame:
    result = None
    turn = 0
    history = []

    def just_won(self):
        return False

    pass


class DummyStgs:
    def get(self, key):
        return 120

    def get_tooltip(self, idx):
        return ""


class DummyCanvas:
    def bind(self, *args, **kwargs):
        pass


class DummyGraph:
    ParentContainer = None
    Key = "BOARD"
    Type = "DummyGraph"
    Font = None
    AutoSizeText = None
    TextColor = None
    Pad = None
    Size = None
    metadata = None
    TKCanvas = DummyCanvas()


class DummyBoard:
    graph = DummyGraph()

    def draw(self):
        pass


class DummyNM:
    def eval_game(self, game):
        return 0.0, [], [], []


class DummyPlayer:
    def __init__(self, **kwargs):
        self.nm = DummyNM()


src.backend.nnmplayer.Player = DummyPlayer


class DummyPlot:
    def update(self, values, maxval):
        pass


class TestBotEvent(unittest.TestCase):
    def test_context(self):
        be = BotEvent()
        self.assertIsNone(be.get_context())
        be.set_context("ctx")
        self.assertEqual(be.get_context(), "ctx")
        self.assertTrue(be.is_set())


class TestProgressWindow(unittest.TestCase):
    def test_init(self):
        pw = ProgressWindow()
        self.assertIsInstance(pw, ProgressWindow)


class TestTwixtbotUI(unittest.TestCase):
    def test_init(self):
        class PatchedTwixtbotUI(TwixtbotUI):
            def update_evals(self):
                pass

        ui = PatchedTwixtbotUI(DummyGame(), DummyStgs(), DummyBoard())
        self.assertIsInstance(ui, TwixtbotUI)
        self.assertIs(ui.board, ui.board)
        self.assertIs(ui.game, ui.game)
        self.assertIs(ui.stgs, ui.stgs)
        self.assertIsNone(ui.bot_event)
        self.assertIsInstance(ui.redo_moves, list)
        self.assertIsNone(ui.next_move)
        self.assertIsInstance(ui.logger, type(ui.logger))
        self.assertIsInstance(ui.ui_to_be_updated, threading.Event)
        self.assertIsNone(ui.thread)
        self.assertIsNone(ui.timer)


if __name__ == "__main__":
    unittest.main()
