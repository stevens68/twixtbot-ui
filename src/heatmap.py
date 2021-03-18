# -*- coding: utf-8 -*-
import numpy
import backend.twixt as twixt


class Heatmap:
    def __init__(self, game, bot):
        self.game = game
        self.bot = bot
        self.policy_moves = None
        self.scores = numpy.full((self.game.SIZE, self.game.SIZE),
                                 numpy.nan,
                                 numpy.single)

        self.calculate()

    def calculate(self):
        # Get current win probability as base score
        base_sc, self.policy_moves, _ = self.bot.nm.eval_game(self.game)

        # Unpack policy moves from (move, P)-list
        self.policy_moves = [str(m) for m, p in self.policy_moves]

        # Loop over all legal moves and calculate and store score in heatmap
        for move in self.game.slow_legal_plays():
            if move != twixt.SWAP:
                self.game.play(move)
                sc, moves, _ = self.bot.nm.eval_game(self.game)
                self.scores[move] = base_sc - sc
                self.game.undo()

        # normalize heatmap
        scalar = max(-numpy.nanmin(self.scores), numpy.nanmax(self.scores))
        self.scores /= scalar
