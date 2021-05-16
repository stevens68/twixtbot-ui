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
                    ct.ROT_RAND: random.randint(0, 3),
                    ct.ROT_FLIP_HOR: 1,
                    ct.ROT_FLIP_VERT: 2,
                    ct.ROT_FLIP_BOTH: 3
                }
          
                nips = naf.NetInputs(game)
                # self.logger.error("ROR: %s", self.rotation)

                if self.rotation == ct.ROT_AVG:
                    pw, ml = get_pw_ml(nips, 0)
                    for i in range(3):
                        # 0 => 1 flip => nips is flipped hor => 1 to map ml   
                        # 1 => 2 flip => nips is flipped both => 3 to map ml
                        # 2 => 1 flip => nips is flipped vert => 2 to map ml
                        nips.rotate(i % 2 + 1)  
                        pw_rot, ml_rot = get_pw_ml(nips, (3 - i) % 3 + 1)
                        pw += pw_rot
                        ml += ml_rot
                        x = ml.argmax()
                    pw /= 4
                    ml /= 4
                else: 
                    rot = rot_map[self.rotation]
                    nips.rotate(rot)
                    pw, ml = get_pw_ml(nips, rot)

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
                                 0, moves=moves, P=[int(round(p * 1000)) for p in P])

        N = self.nm.mcts(game, self.num_trials, window, event)

        self.report = self.nm.report

        # When a forcing win or forcing draw move is found, there's no policy
        # array returned
        if isinstance(N, (str, twixt.Point)):
            return self.nm.create_response(game, "done", self.num_trials,
                                 self.num_trials, True, P=[1000, 0, 0])

        if self.temperature == 0.0:
            mx = N.max()
            weights = numpy.where(N == mx, 1.0, 0.0)
        elif self.temperature == 1.0:
            weights = N
        elif self.temperature == 0.5:
            weights = N ** 2
        self.logger.info("weights=%s", weights)
        index = numpy.random.choice(numpy.arange(
            len(weights)), p=weights / weights.sum())
        
        return self.nm.create_response(game, "done", self.num_trials,
                             self.num_trials, False)
