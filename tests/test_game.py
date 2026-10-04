#! /usr/bin/env python

import unittest

import numpy as np

from src.backend.point import Point
from src.backend.twixt import Game, InvalidMoveError, SWAP


class TestGame(unittest.TestCase):

    def setUp(self):
        self.game = Game(allow_scl=False)

    def test_init(self):
        self.assertEqual(self.game.SIZE, 24)
        self.assertEqual(self.game.turn, Game.WHITE)
        self.assertEqual(len(self.game.pegs), 2)
        self.assertEqual(len(self.game.links), 8)
        self.assertIsInstance(self.game.history, list)
        self.assertIsInstance(self.game.open_pegs, list)
        self.assertIsInstance(self.game.reachable, list)
        self.assertIsInstance(self.game.reachable_history, list)

    def test_flip_turn(self):
        orig_turn = self.game.turn

        self.game._flip_turn()
        self.assertNotEqual(self.game.turn, orig_turn)

        self.game._flip_turn()
        self.assertEqual(self.game.turn, orig_turn)

    def test_play_updates_game_state(self):
        move = Point(5, 5)

        self.game.play(move)

        self.assertEqual(self.game.pegs[Game.WHITE][move], 1)
        self.assertEqual(self.game.pegs[Game.BLACK][move], 0)
        self.assertEqual(self.game.history, [move])
        self.assertEqual(self.game.turn, Game.BLACK)

        self.assertNotIn(move, self.game.open_pegs[Game.WHITE])
        self.assertNotIn(move, self.game.open_pegs[Game.BLACK])

    def test_play_accepts_string_move(self):
        self.game.play("f6")

        self.assertEqual(len(self.game.history), 1)
        self.assertIsInstance(self.game.history[0], Point)

    def test_play_rejects_out_of_bounds_move(self):
        with self.assertRaises(InvalidMoveError):
            self.game.play(Point(-1, 5))

        with self.assertRaises(InvalidMoveError):
            self.game.play(Point(Game.SIZE, 5))

        with self.assertRaises(InvalidMoveError):
            self.game.play(Point(5, -1))

        with self.assertRaises(InvalidMoveError):
            self.game.play(Point(5, Game.SIZE))

    def test_white_cannot_play_on_horizontal_boundary(self):
        with self.assertRaises(AssertionError):
            self.game.play(Point(0, 5))

        with self.assertRaises(AssertionError):
            self.game.play(Point(Game.SIZE - 1, 5))

    def test_black_cannot_play_on_horizontal_boundary(self):
        self.game.play(Point(5, 5))

        with self.assertRaises(AssertionError):
            self.game.play(Point(5, 0))

        with self.assertRaises(AssertionError):
            self.game.play(Point(5, Game.SIZE - 1))

    def test_play_rejects_occupied_point(self):
        move = Point(5, 5)

        self.game.play(move)

        # Give the turn back to White.
        self.game.play(Point(6, 6))

        with self.assertRaises(InvalidMoveError):
            self.game.play(move)

    def test_undo_restores_move(self):
        move = Point(5, 5)

        self.game.play(move)
        self.game.undo()

        self.assertEqual(self.game.history, [])
        self.assertEqual(self.game.turn, Game.WHITE)
        self.assertEqual(self.game.pegs[Game.WHITE][move], 0)
        self.assertEqual(self.game.pegs[Game.BLACK][move], 0)

        self.assertIn(move, self.game.open_pegs[Game.WHITE])
        self.assertIn(move, self.game.open_pegs[Game.BLACK])

    def test_undo_multiple_moves(self):
        moves = [
            Point(5, 5),
            Point(6, 6),
            Point(7, 7),
        ]

        for move in moves:
            self.game.play(move)

        self.game.undo()

        self.assertEqual(self.game.history, moves[:2])
        self.assertEqual(self.game.turn, Game.WHITE)

        self.game.undo()

        self.assertEqual(self.game.history, moves[:1])
        self.assertEqual(self.game.turn, Game.BLACK)

        self.game.undo()

        self.assertEqual(self.game.history, [])
        self.assertEqual(self.game.turn, Game.WHITE)

    def test_undo_empty_game_fails(self):
        with self.assertRaises(AssertionError):
            self.game.undo()

    def test_clone(self):
        moves = [
            Point(5, 5),
            Point(6, 6),
            Point(7, 7),
        ]

        for move in moves:
            self.game.play(move)

        clone = self.game.clone()

        self.assertIsNot(clone, self.game)
        self.assertEqual(clone.history, self.game.history)
        self.assertEqual(clone.turn, self.game.turn)
        self.assertEqual(clone.reachable, self.game.reachable)
        self.assertEqual(
            clone.reachable_history,
            self.game.reachable_history,
        )

        np.testing.assert_array_equal(clone.pegs, self.game.pegs)
        np.testing.assert_array_equal(clone.links, self.game.links)

    def test_clone_can_be_modified_without_affecting_original(self):
        first = Point(5, 5)
        second = Point(6, 6)

        self.game.play(first)

        clone = self.game.clone()
        clone.play(second)

        self.assertEqual(self.game.history, [first])
        self.assertEqual(clone.history, [first, second])

        self.assertEqual(self.game.pegs[Game.BLACK][second], 0)
        self.assertEqual(clone.pegs[Game.BLACK][second], 1)

    def test_clone_history_is_independent(self):
        self.game.play(Point(5, 5))

        clone = self.game.clone()
        clone.history.append(Point(10, 10))

        self.assertEqual(len(self.game.history), 1)
        self.assertEqual(len(clone.history), 2)

    def test_clone_open_pegs_are_independent(self):
        point = Point(5, 5)

        clone = self.game.clone()
        clone.open_pegs[Game.WHITE].remove(point)

        self.assertIn(point, self.game.open_pegs[Game.WHITE])
        self.assertNotIn(point, clone.open_pegs[Game.WHITE])

    def test_clone_reachable_sets_are_independent(self):
        point = Point(5, 5)

        clone = self.game.clone()
        clone.reachable[Game.WHITE].add(point)

        self.assertNotIn(point, self.game.reachable[Game.WHITE])
        self.assertIn(point, clone.reachable[Game.WHITE])

    def test_set_and_get_link(self):
        a = Point(5, 5)
        b = Point(7, 6)

        self.assertEqual(
            self.game.get_link(a, b, Game.WHITE),
            0,
        )

        self.game.set_link(a, b, Game.WHITE, 1)

        self.assertEqual(
            self.game.get_link(a, b, Game.WHITE),
            1,
        )

    def test_links_are_independent_by_color(self):
        a = Point(5, 5)
        b = Point(7, 6)

        self.game.set_link(a, b, Game.WHITE, 1)

        self.assertEqual(
            self.game.get_link(a, b, Game.WHITE),
            1,
        )
        self.assertEqual(
            self.game.get_link(a, b, Game.BLACK),
            0,
        )

    def test_play_creates_same_color_link(self):
        white1 = Point(5, 5)
        black = Point(10, 10)
        white2 = Point(7, 6)

        self.game.play(white1)
        self.game.play(black)
        self.game.play(white2)

        self.assertEqual(
            self.game.get_link(white1, white2, Game.WHITE),
            1,
        )

    def test_undo_removes_link_created_by_last_move(self):
        white1 = Point(5, 5)
        black = Point(10, 10)
        white2 = Point(7, 6)

        self.game.play(white1)
        self.game.play(black)
        self.game.play(white2)

        self.assertEqual(
            self.game.get_link(white1, white2, Game.WHITE),
            1,
        )

        self.game.undo()

        self.assertEqual(
            self.game.get_link(white1, white2, Game.WHITE),
            0,
        )

    def test_safe_get_peg_outside_board(self):
        self.assertEqual(
            self.game.safe_get_peg(Point(-1, 0), Game.WHITE),
            0,
        )

        self.assertEqual(
            self.game.safe_get_peg(
                Point(Game.SIZE, 0),
                Game.WHITE,
            ),
            0,
        )

    def test_safe_get_link_outside_board(self):
        a = Point(-1, 5)
        b = Point(5, 5)

        self.assertEqual(
            self.game.safe_get_link(a, b, Game.WHITE),
            0,
        )

        self.assertEqual(
            self.game.safe_get_link(b, a, Game.WHITE),
            0,
        )

    def test_inbounds(self):
        self.assertTrue(Game.inbounds(Point(0, 0)))
        self.assertTrue(
            Game.inbounds(
                Point(Game.SIZE - 1, Game.SIZE - 1)
            )
        )

        self.assertFalse(Game.inbounds(Point(-1, 0)))
        self.assertFalse(Game.inbounds(Point(0, -1)))
        self.assertFalse(
            Game.inbounds(Point(Game.SIZE, 0))
        )
        self.assertFalse(
            Game.inbounds(Point(0, Game.SIZE))
        )

    def test_inbounds_for_player(self):
        self.assertTrue(
            Game.inbounds_for_player(
                Point(1, 5),
                Game.WHITE,
            )
        )
        self.assertFalse(
            Game.inbounds_for_player(
                Point(0, 5),
                Game.WHITE,
            )
        )
        self.assertFalse(
            Game.inbounds_for_player(
                Point(Game.SIZE - 1, 5),
                Game.WHITE,
            )
        )

        self.assertTrue(
            Game.inbounds_for_player(
                Point(5, 1),
                Game.BLACK,
            )
        )
        self.assertFalse(
            Game.inbounds_for_player(
                Point(5, 0),
                Game.BLACK,
            )
        )
        self.assertFalse(
            Game.inbounds_for_player(
                Point(5, Game.SIZE - 1),
                Game.BLACK,
            )
        )

        self.assertFalse(
            Game.inbounds_for_player(Point(5, 5), 99)
        )

    def test_turn_to_player_uses_explicit_turn(self):
        self.assertEqual(
            self.game.turn_to_player(Game.WHITE),
            1,
        )
        self.assertEqual(
            self.game.turn_to_player(Game.BLACK),
            2,
        )

    def test_turn_to_player_uses_current_turn_when_omitted(self):
        self.assertEqual(
            self.game.turn_to_player(),
            2 - self.game.turn,
        )

        self.game.play(Point(5, 5))

        self.assertEqual(
            self.game.turn_to_player(),
            2 - self.game.turn,
        )

    def test_swap(self):
        first = Point(5, 7)
        swapped = Point(first.y, first.x)

        self.game.play(first)
        self.assertEqual(self.game.turn, Game.BLACK)

        self.game.play(SWAP)

        self.assertEqual(
            self.game.history,
            [first, SWAP],
        )
        self.assertEqual(
            self.game.turn,
            Game.WHITE,
        )

        self.assertEqual(
            self.game.pegs[Game.WHITE][first],
            0,
        )
        self.assertEqual(
            self.game.pegs[Game.BLACK][swapped],
            1,
        )

    def test_undo_swap(self):
        first = Point(5, 7)
        swapped = Point(first.y, first.x)

        self.game.play(first)
        self.game.play(SWAP)
        self.game.undo()

        self.assertEqual(
            self.game.history,
            [first],
        )
        self.assertEqual(
            self.game.turn,
            Game.BLACK,
        )
        self.assertEqual(
            self.game.pegs[Game.WHITE][first],
            1,
        )
        self.assertEqual(
            self.game.pegs[Game.BLACK][swapped],
            0,
        )

    def test_swap_requires_first_move(self):
        with self.assertRaises(AssertionError):
            self.game.play(SWAP)

    def test_initial_position_has_no_winner(self):
        self.assertFalse(self.game.is_winning(Game.WHITE))
        self.assertFalse(self.game.is_winning(Game.BLACK))
        self.assertFalse(self.game.just_won())

    def test_reachable_history_tracks_moves(self):
        self.game.play(Point(5, 5))
        self.game.play(Point(6, 6))

        self.assertEqual(
            len(self.game.reachable_history),
            2,
        )

        self.game.undo()

        self.assertEqual(
            len(self.game.reachable_history),
            1,
        )

        self.game.undo()

        self.assertEqual(
            self.game.reachable_history,
            [],
        )


if __name__ == "__main__":
    unittest.main()
