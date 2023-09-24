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
        self.rotation = kwargs.get('rotation', None)

        self.smart_root = int(kwargs.get('smart_root', 0))
        self.allow_swap = int(kwargs.get('allow_swap', 1))
        self.add_noise = float(kwargs.get('add_noise', 0))
        self.cpuct = float(kwargs.get('cpuct', 1.0))
        self.level = float(kwargs.get('level', 1.0))
        self.board = kwargs.get('board', None)
        self.evaluator = kwargs.get('evaluator', None)

        if self.temperature not in (0.0, 0.5, 1.0):
            raise ValueError("Unsupported temperature")

        if self.model:
            # assert not self.socket
            if self.evaluator is None:
                self.evaluator = nneval.NNEvaluater(self.model)

            nneval_ = self.evaluator
            
            def get_pw_ml(n, r):
                p, m = nneval_.eval_one(n)
                if len(p) == 3:
                    p = naf.three_to_one(p)
                if len(p) == 1 and len(p[0] == 3):
                    p = naf.three_to_one(p[0])
                m = naf.rotate_policy_array(m, r)
                if len(m) == 1:
                    m = m[0]
                return p, m

            def nnfunc(game):

                rot_map = {
                    ct.ROT_OFF: 0,
                    ct.ROT_FLIP_HOR: 1,
                    ct.ROT_FLIP_VERT: 2,
                    ct.ROT_FLIP_BOTH: 3
                }
          
                nips = naf.NetInputs(game)

                if self.rotation == ct.ROT_RAND: 
                    rot = random.randint(0, 3)
                    nips.rotate(rot)
                    pw, ml = get_pw_ml(nips, rot)
                elif self.rotation in [ct.ROT_OFF, ct.ROT_FLIP_HOR, ct.ROT_FLIP_VERT, ct.ROT_FLIP_BOTH]: 
                    rot = rot_map[self.rotation]
                    nips.rotate(rot)
                    pw, ml = get_pw_ml(nips, rot)
                elif self.rotation in [ct.ROT_AVG, ct.ROT_BEST_EVALUATION, ct.ROT_BEST_P_VALUE]:
                    pwl, mll = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
                    pwl[0], mll[0] = get_pw_ml(nips, 0)
                    for i in range(3):
                        # i: 0 => nips rot 1 => nips is flipped hor  => 1 to flip ml hor
                        # i: 1 => nips rot 2 => nips is flipped both => 3 to flip ml both
                        # i: 2 => nips rot 1 => nips is flipped vert => 2 to flip ml vert
                        nips.rotate(i % 2 + 1)  
                        pwl[i + 1], mll[i + 1] = get_pw_ml(nips, (3 - i) % 3 + 1)
                    
                    if self.rotation == ct.ROT_AVG:
                        pw, ml = sum(pwl) / 4.0, sum(mll) / 4.0
                    elif self.rotation == ct.ROT_BEST_EVALUATION:
                        imax = pwl.index(max(pwl))
                        pw, ml = pwl[imax], mll[imax]
                    elif self.rotation == ct.ROT_BEST_P_VALUE:
                        p_max = [max(ml) for ml in mll]
                        imax = p_max.index(max(p_max))
                        pw, ml = pwl[imax], mll[imax]
                else:
                    self.logger.error("invalid rotation value: %s", self.rotation)
                    
                return pw, ml
            
        else:
            raise Exception("Specify model or resource")

        self.nm = nnmcts.NeuralMCTS(
            nnfunc,
            add_noise=self.add_noise,
            smart_root=self.smart_root,
            cpuct=self.cpuct,
            board=self.board,
            level=self.level,
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
            _, moves, P, Pscew = self.nm.eval_game(game)
            return self.nm.create_response(game, "done", 0,
                                           0, moves=moves,
                                           P=P, Pscew=Pscew)

        N = self.nm.mcts(game, self.num_trials, window, event)

        self.report = self.nm.report

        # When a forcing win or forcing draw move is found, there's no policy
        # array returned
        if isinstance(N, (str, game.Point)):
            return self.nm.create_response(game, "done", self.num_trials,
                                           self.num_trials, True)

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
