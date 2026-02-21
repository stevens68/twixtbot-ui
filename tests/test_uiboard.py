import unittest
from unittest.mock import MagicMock
import src.uiboard
from collections import namedtuple

DummyMove = namedtuple("DummyMove", ["x", "y"])


class DummyGame:
    history = [DummyMove(1, 2), DummyMove(3, 4), DummyMove(5, 6)]
    DLINKS = []


class DummyStgs:
    def get(self, key):
        return 100


class DummyGraph:
    def erase(self):
        self.erased = True

    def DrawText(self, *args, **kwargs):
        return "text"

    def GetBoundingBox(self, item):
        return (0, 0), (10, 10)

    def DrawRectangle(self, *args, **kwargs):
        return "rect"

    def BringFigureToFront(self, item):
        self.brought = item

    def delete_figure(self, item):
        self.deleted = item

    def DrawCircle(self, *args, **kwargs):
        return "circle"


class DummyTwixtBoard:
    SIZE = 20

    @staticmethod
    def _move_to_point(move):
        return (1, 2)


src.uiboard.board = MagicMock(TwixtBoard=DummyTwixtBoard)
src.uiboard.sg = MagicMock(Graph=lambda **kwargs: DummyGraph())
src.uiboard.ct = MagicMock(
    K_BOARD_SIZE=(None, "key", None, 10),
    FIELD_BACKGROUND_COLOR="white",
    K_BOARD=("desc", "key"),
    BOARD_LABEL_COLOR="black",
    BOARD_LABEL_FONT="Arial",
    CURSOR_LABEL_BACKGROUND_COLOR="gray",
)
src.uiboard.twixt = MagicMock(Game=DummyTwixtBoard)


class TestUiBoard(unittest.TestCase):
    def test_init_and_draw(self):
        board = src.uiboard.UiBoard(DummyGame(), DummyStgs())
        board.graph = DummyGraph()
        board._draw_endlines = MagicMock()
        board._draw_labels = MagicMock()
        board._draw_guidelines = MagicMock()
        board._draw_pegholes = MagicMock()
        board.create_move_objects = MagicMock()
        board._draw_heatmap_legend = MagicMock()
        board._draw_heatmap = MagicMock()
        board.draw()
        board.draw(heatmap="hm")
        board.draw(complete=False)
        self.assertIsInstance(board, src.uiboard.UiBoard)

    def test_draw_cursor_label(self):
        board = src.uiboard.UiBoard(DummyGame(), DummyStgs())
        board.graph = DummyGraph()
        board.current_cursor_label = None
        board.rect_item = None
        board._point_to_coords = MagicMock(return_value=(5, 5))
        board.draw_cursor_label("a1")
        board.draw_cursor_label(None)
        self.assertIsInstance(board, src.uiboard.UiBoard)

    def test_create_move_objects(self):
        board = src.uiboard.UiBoard(DummyGame(), DummyStgs())
        board.graph = DummyGraph()
        board.game = DummyGame()
        board.create_move_objects(0)
        self.assertIsInstance(board, src.uiboard.UiBoard)


if __name__ == "__main__":
    unittest.main()
