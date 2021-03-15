# -*- coding: utf-8 -*-
import numpy
import backend.twixt as twixt


class Heatmap:
    def __init__(self, game, bot):
        self.game = game
        self.bot = bot
        self.scores = numpy.full((self.game.SIZE, self.game.SIZE),
                                 numpy.nan,
                                 numpy.single)

        self.calculate()

    def _get_score(self):
        sc, _ = self.bot.nm.eval_game(self.game)
        return sc

    def calculate(self):
        base_sc = self._get_score()
        slp = self.game.slow_legal_plays()
        if twixt.SWAP in slp:
            slp.remove(twixt.SWAP)
        for move in slp:
            self.game.play(move)
            self.scores[move] = base_sc - self._get_score()
            self.game.undo()

        # normalize
        scalar = max(-numpy.nanmin(self.scores), numpy.nanmax(self.scores))
        self.scores /= scalar
