#! /usr/bin/env python
import numpy
import random
import logging
import constants as ct
import backend.naf as naf
import backend.nneval as nneval
import backend.nnmcts as nnmcts
import backend.swapmodel as swapmodel
import backend.twixt as twixt


class Player:

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(ct.LOGGER)
        self.model = kwargs.get('model', None)
        self.num_trials = int(kwargs.get('trials', 100))
        self.temperature = float(kwargs.get('temperature', 0))
        self.random_rotation = int(kwargs.get('random_rotation', 0))

        self.smart_root = int(kwargs.get('smart_root', 0))
        self.allow_swap = int(kwargs.get('allow_swap', 1))
        self.add_noise = float(kwargs.get('add_noise', 0))
        self.cpuct = float(kwargs.get('cpuct', 1))
        self.board = kwargs.get('board', None)
        self.evaluator = kwargs.get('evaluator', None)

        if self.temperature not in (0.0, 0.5, 1.0):
            raise ValueError("Unsupported temperature")

        if self.model:
            # assert not self.socket
            if self.evaluator is None:
                self.evaluator = nneval.NNEvaluater(self.model)

            nneval_ = self.evaluator

            def nnfunc(game):

                nips = naf.NetInputs(game)
                if self.random_rotation:
                    rot = random.randint(0, 3)
                    nips.rotate(rot)
                else:
                    rot = 0
                pw, ml = nneval_.eval_one(nips)
                if len(pw) == 3:
                    pw = naf.three_to_one(pw)
                if len(pw) == 1 and len(pw[0] == 3):
                    pw = naf.three_to_one(pw[0])
                ml = naf.rotate_policy_array(ml, rot)
                if len(ml) == 1:
                    ml = ml[0]
                return pw, ml
        else:
            raise Exception("Specify model or resource")

        self.nm = nnmcts.NeuralMCTS(
            nnfunc,
            add_noise=self.add_noise,
            smart_root=self.smart_root,
            cpuct=self.cpuct,
            board=self.board,
            visualize_mcts=False
        )

    def pick_move(self, game, window=None, event=None):
        if self.allow_swap and len(game.history) < 2:

            if len(game.history) == 0:
                self.report = "swapmodel"
                m = swapmodel.choose_first_move()
                return self.nm.create_response(game, "done", moves=[m])
            elif swapmodel.want_swap(game.history[0]):
                self.report = "swapmodel"
                m = twixt.SWAP
                return self.nm.create_response(game, "done", moves=[m])
            # else:
            #   didn't want to swap => compute move

        if self.num_trials == 0:
            # don't use MCTS but just evaluate and return best move
            _, moves, P = self.nm.eval_game(game)
            return self.nm.create_response(game, "done", 0,
                                           0, moves=moves,
                                           P=[int(round(p * 1000)) for p in P])

        N = self.nm.mcts(game, self.num_trials, window, event)

        self.report = self.nm.report

        # When a forcing win or forcing draw move is found, there's no policy
        # array returned
        if isinstance(N, (str, twixt.Point)):
            return self.nm.create_response(game, "done", self.num_trials,
                                           self.num_trials, True,
                                           P=[1000, 0, 0])

        if self.temperature == 0.0:
            mx = N.max()
            weights = numpy.where(N == mx, 1.0, 0.0)
        elif self.temperature == 1.0:
            weights = N
        elif self.temperature == 0.5:
            weights = N ** 2
        self.logger.debug("weights=%s", weights)

        # flake8 said index isn't used. Not sure why this code is here.
        # Just to be sure: commenting out the code instead of deleting:
        # index = numpy.random.choice(numpy.arange(
        #     len(weights)), p=weights / weights.sum())

        return self.nm.create_response(game, "done", self.num_trials,
                                       self.num_trials, False)
