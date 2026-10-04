"""Regression tests for MCTS/thread interaction in TwixtbotUI."""

import unittest
from unittest.mock import MagicMock, patch

from src import tbui
from src.backend import twixt


class DummyControl:
    """Minimal GUI control used by TwixtbotUI tests."""

    def __init__(self, value=True):
        self.value = value

    def get(self):
        return self.value

    def update(self, **kwargs):
        pass

    def Update(self, *args, **kwargs):
        pass


class DummyBoard:
    """Minimal board implementation for visualization tests."""

    def __init__(self):
        self.known_moves = set()
        self.graph = MagicMock()

    def _create_drawn_peg(self, move, color, last, visits):
        return ("peg", move, color, visits)

    def _create_visits_label(self, move, color, visits):
        return ("label", move, color, visits)

    def _create_drawn_link(self, move, other, color, visits):
        return ("link", move, other, color, visits)


def make_ui(game=None):
    """Create a TwixtbotUI instance without constructing the GUI."""
    ui = tbui.TwixtbotUI.__new__(tbui.TwixtbotUI)

    ui.game = game if game is not None else twixt.Game(False)
    ui.board = DummyBoard()
    ui.vis_path_objects = []
    ui.window = MagicMock()
    ui.logger = MagicMock()
    ui.bot_event = None
    ui.bots = [MagicMock(), MagicMock()]

    return ui


class TestDrawVisPath(unittest.TestCase):
    def test_visualization_does_not_mutate_live_game(self):
        """
        Regression test for issue #122.

        An MCTS visualization path may contain moves that are already part
        of the live game because the GUI event was queued while MCTS was
        running.

        Visualization must replay the path on a clone rather than on the
        live game.
        """
        game = twixt.Game(False)

        first_move = twixt.Point(1, 1)
        next_move = twixt.Point(2, 2)

        game.play(first_move)

        original_history = list(game.history)
        original_turn = game.turn
        original_pegs = [pegs.copy() for pegs in game.pegs]
        original_links = [links.copy() for links in game.links]

        ui = make_ui(game)
        ui.get_control = MagicMock(
            return_value=DummyControl(value=True)
        )

        path = [
            # This move is already present at the MCTS root.
            (first_move, 100),
            # This move is new and should be replayed on the clone.
            (next_move, 50),
        ]

        # Before the fix, this could raise InvalidMoveError because
        # first_move was played a second time on the live game.
        ui.draw_vis_path(path)

        # The live game must remain completely unchanged.
        self.assertEqual(game.history, original_history)
        self.assertEqual(game.turn, original_turn)

        for actual, expected in zip(game.pegs, original_pegs):
            self.assertTrue((actual == expected).all())

        for actual, expected in zip(game.links, original_links):
            self.assertTrue((actual == expected).all())

        # The already-existing root move is skipped.
        # The new move creates one peg and one label.
        self.assertEqual(len(ui.vis_path_objects), 2)

    def test_visualization_plays_new_path_moves_on_clone(self):
        """
        A path move not present in the root history is replayed on the
        cloned game, but never on the live game.
        """
        game = twixt.Game(False)

        root_move = twixt.Point(1, 1)
        visualization_move = twixt.Point(2, 2)

        game.play(root_move)

        ui = make_ui(game)
        ui.get_control = MagicMock(
            return_value=DummyControl(value=True)
        )

        path = [
            (root_move, 100),
            (visualization_move, 75),
        ]

        ui.draw_vis_path(path)

        # Only the real root move exists in the live game.
        self.assertEqual(game.history, [root_move])

        # The root move is skipped; the new move produces peg + label.
        self.assertEqual(len(ui.vis_path_objects), 2)

    def test_visualization_with_empty_path_does_not_change_game(self):
        """An empty visualization path is a no-op."""
        game = twixt.Game(False)
        move = twixt.Point(1, 1)
        game.play(move)

        ui = make_ui(game)
        ui.get_control = MagicMock(
            return_value=DummyControl(value=True)
        )

        ui.draw_vis_path([])

        self.assertEqual(game.history, [move])
        self.assertEqual(ui.vis_path_objects, [])

    def test_visualization_disabled_does_not_change_game(self):
        """Disabled visualization is a no-op."""
        game = twixt.Game(False)
        move = twixt.Point(1, 1)
        game.play(move)

        ui = make_ui(game)
        ui.get_control = MagicMock(
            return_value=DummyControl(value=False)
        )

        ui.draw_vis_path([(twixt.Point(2, 2), 10)])

        self.assertEqual(game.history, [move])
        self.assertEqual(ui.vis_path_objects, [])

    def test_clean_visualization_removes_previous_objects(self):
        """Existing visualization objects are removed before drawing."""
        ui = make_ui()

        old_objects = ["peg1", "label1", "link1"]
        ui.vis_path_objects = old_objects.copy()

        ui.clean_vis_path()

        self.assertEqual(ui.vis_path_objects, [])

        for obj in old_objects:
            ui.board.graph.delete_figure.assert_any_call(obj)


class TestCallBot(unittest.TestCase):
    def test_mcts_receives_game_clone_not_live_game(self):
        """
        The bot worker must never search the live GUI game.

        The GUI thread can continue changing the live game while MCTS is
        running, so the worker must receive an independent clone.
        """
        game = twixt.Game(False)
        game.play(twixt.Point(1, 1))

        ui = make_ui(game)

        search_game = game.clone()
        game.clone = MagicMock(return_value=search_game)

        response = {
            "moves": [twixt.Point(2, 2)],
            "Pscew": [1.0],
        }

        player = search_game.turn
        ui.bots[player].pick_move.return_value = response

        with patch.object(
            tbui,
            "_stochastic_choice",
            return_value=twixt.Point(2, 2),
        ):
            ui.execute_move = MagicMock()
            ui.call_bot()

        game.clone.assert_called_once_with()

        ui.bots[player].pick_move.assert_called_once()

        bot_game = ui.bots[player].pick_move.call_args.args[0]

        self.assertIs(bot_game, search_game)
        self.assertIsNot(bot_game, game)

        ui.execute_move.assert_called_once_with(
            twixt.Point(2, 2)
        )

    def test_cancelled_bot_resets_searching_players_mcts_root(self):
        """
        When the bot is cancelled, history_at_root is reset on the bot
        that searched the cloned game.

        The player is captured before the worker runs so a concurrent
        change of the live game turn cannot select the wrong bot.
        """
        game = twixt.Game(False)
        game.play(twixt.Point(1, 1))

        ui = make_ui(game)

        search_game = game.clone()
        ui.game.clone = MagicMock(return_value=search_game)

        player = search_game.turn
        bot = ui.bots[player]

        bot.pick_move.return_value = {
            "moves": [twixt.Point(2, 2)],
            "Pscew": [1.0],
        }

        event = MagicMock()
        event.is_set.return_value = True
        event.get_context.return_value = ct_cancel_event()

        ui.bot_event = event
        ui.execute_move = MagicMock()

        ui.call_bot()

        self.assertIsNone(bot.nm.history_at_root)
        ui.execute_move.assert_not_called()


class TestHandleThreadEvent(unittest.TestCase):
    def test_stale_mcts_event_is_ignored(self):
        """
        An MCTS event generated for an older game position must not be
        rendered after the live game has advanced.
        """
        game = twixt.Game(False)
        game.play(twixt.Point(1, 1))

        ui = make_ui(game)

        ui.update_progress = MagicMock()
        ui.draw_vis_path = MagicMock()

        stale_history = []

        values = {
            "history": stale_history,
            "max": 1000,
            "current": 500,
            "best_path": [
                (twixt.Point(2, 2), 100),
            ],
        }

        ui.handle_thread_event(values)

        ui.update_progress.assert_not_called()
        ui.draw_vis_path.assert_not_called()

    def test_current_mcts_event_is_processed(self):
        """An event for the current game position is processed."""
        game = twixt.Game(False)
        game.play(twixt.Point(1, 1))

        ui = make_ui(game)

        ui.update_progress = MagicMock()
        ui.draw_vis_path = MagicMock()

        values = {
            "history": list(game.history),
            "max": 1000,
            "current": 500,
            "best_path": [
                (twixt.Point(2, 2), 100),
            ],
        }

        ui.handle_thread_event(values)

        ui.update_progress.assert_called_once_with(values)
        ui.draw_vis_path.assert_called_once_with(
            values["best_path"]
        )

    def test_event_without_history_is_still_supported(self):
        """
        Events without a history field continue through normal handling.
        """
        ui = make_ui()

        ui.update_progress = MagicMock()
        ui.draw_vis_path = MagicMock()

        values = {
            "max": 0,
        }

        ui.handle_thread_event(values)

        ui.update_progress.assert_not_called()
        ui.draw_vis_path.assert_not_called()

    def test_stale_event_with_already_played_move_is_ignored(self):
        game = twixt.Game(False)

        move = twixt.Point(1, 1)
        game.play(move)

        ui = make_ui(game)
        ui.update_progress = MagicMock()
        ui.draw_vis_path = MagicMock()

        values = {
            "history": [],
            "max": 1000,
            "current": 500,
            "best_path": [
                (move, 100),
            ],
        }

        ui.handle_thread_event(values)

        ui.update_progress.assert_not_called()
        ui.draw_vis_path.assert_not_called()

    def test_visualization_replays_multiple_new_moves_on_clone(self):
        game = twixt.Game(False)

        root_move = twixt.Point(1, 1)
        move2 = twixt.Point(2, 2)
        move3 = twixt.Point(3, 3)

        game.play(root_move)

        ui = make_ui(game)
        ui.get_control = MagicMock(
            return_value=DummyControl(value=True)
        )

        path = [
            (root_move, 100),
            (move2, 75),
            (move3, 50),
        ]

        ui.draw_vis_path(path)

        self.assertEqual(game.history, [root_move])

        # Two new moves = two pegs + two labels.
        self.assertEqual(len(ui.vis_path_objects), 4)



    def test_visualization_uses_game_clone(self):
        game = twixt.Game(False)
        move = twixt.Point(1, 1)

        game.play(move)

        clone = game.clone()

        ui = make_ui(game)
        ui.get_control = MagicMock(
            return_value=DummyControl(value=True)
        )

        with patch.object(game, "clone", return_value=clone) as clone_mock:
            ui.draw_vis_path([
                (twixt.Point(2, 2), 10),
            ])

        clone_mock.assert_called_once_with()



def ct_cancel_event():
    """
    Return the real cancel-event constant.

    Kept in a helper so the test does not depend on a hard-coded string.
    """
    import src.constants as ct

    return ct.CANCEL_EVENT


if __name__ == "__main__":
    unittest.main()