#! /usr/bin/env python
import math
import numpy
import logging

import backend.naf as naf
import backend.twixt as twixt
import constants as ct
from backend.point import Point


class EvalNode:

    def __init__(self):

        k = twixt.Game.SIZE * (twixt.Game.SIZE - 2)
        self.N = numpy.zeros(k)
        self.Q = numpy.zeros(k)
        self.P = numpy.zeros(k)
        self.proven = False
        self.score = None
        self.winning_move = None
        self.drawing_move = None
        self.subnodes = [None] * k


class NeuralMCTS:

    def __init__(self, sap, **kwargs):
        """ sap = score and policy function, takes a game as input """
        self.cpuct = kwargs.pop("cpuct", 1.0)
        self.add_noise = kwargs.pop("add_noise", 0.25)
        self.level = kwargs.pop("level", None)
        self.smart_root = kwargs.pop("smart_root", 0)
        self.smart_init = kwargs.pop("smart_init", 0)
        self.board = kwargs.pop("board", None)
        self.visualize_mcts = kwargs.pop("visualize_mcts", None)

        if kwargs:
            raise TypeError('Unexpected kwargs provided: %s' %
                            list(kwargs.keys()))
        self.sap = sap
        self.root = None
        self.history_at_root = None
        self.logger = logging.getLogger(ct.LOGGER)

    def expand_leaf(self, game):
        """ Create a brand new leaf node for the current game state
            and return it. """

        leaf = EvalNode()

        if game.just_won():
            leaf.proven = True
            leaf.score = -1
            leaf.LMnz = "just_won"
            return leaf

        leaf.LM = naf.legal_move_policy_array(game)
        leaf.LMnz = leaf.LM.nonzero()

        if not leaf.LM.any():
            # assert False
            leaf.proven = True
            leaf.score = 0
            leaf.LMnz = "just_drew"
            return leaf

        ####

        poseval, movelogits = self.sap(game)
        leaf.score = poseval
        if self.smart_init:
            leaf.Q[:] = poseval
        maxlogit = movelogits[leaf.LMnz].max()
        el = numpy.exp(movelogits - maxlogit)
        divisor = el[leaf.LMnz].sum()
        leaf.P = el / divisor
        # stevens68: set P to 0 for non-legal moves. Otherwise
        # illegal move might become best move in upcoming draws (move #100+)
        leaf.P[numpy.where(leaf.LM == 0)[0]] = 0
        # leaf.P *= leaf.LM

        self.logger.debug("LMnz: %s", leaf.LMnz)
        self.logger.debug("raw P: %s", leaf.P)

        if self.add_noise:
            leaf.P[leaf.LMnz] *= (1.0 - self.add_noise)
            leaf.P[leaf.LMnz] += self.add_noise * \
                numpy.random.dirichlet(0.03 * numpy.ones(len(leaf.LMnz[0])))

        self.logger.debug("after noise P: %s", leaf.P)
        return leaf
        # end expand_leaf()

    def visit_node(self, game, node, top=False, trials=None):
        """ Visit a node, return the evaluation from the
            point of view of the player currently to play. """

        assert not game.just_won()
        if not node.LM.any():
            self.proven = True
            if node.drawing_move:
                self.score = 0
            else:
                # all moves lose.  very sad.
                self.score = -1
            return self.score

        if top and self.smart_root:
            vnz = node.N[node.LMnz]
            maxn = vnz.max()
            winnables = (node.N > maxn - trials) & node.LM
            num_winnables = numpy.count_nonzero(winnables)
            assert num_winnables > 0, (maxn, trials, numpy.array_str(
                node.N), numpy.array_str(node.LM))
            if num_winnables == 1:
                index = winnables.argmax()
            else:
                maxes = (node.N == maxn)
                num_maxes = numpy.count_nonzero(maxes)
                assert num_maxes > 0
                if (num_maxes == 1 and
                        maxn - vnz[numpy.argsort(vnz)[-2:]][0] > 1):
                    # visit diff to second best is >1
                    winnables[maxes.argmax()] = 0

                nsum = node.N.sum()  # don't need to filter since all are 0
                stv = math.sqrt(nsum + 1.0)
                U = node.Q + self.cpuct * node.P * stv / (1.0 + node.N)

                wnz = numpy.nonzero(winnables)
                nz_index = U[wnz].argmax()
                index = wnz[0][nz_index]
        else:
            # At least one node worth visiting.  Figure out which one to
            # visit...
            nsum = node.N.sum()  # don't need to filter since all are 0
            stv = math.sqrt(nsum + 1.0)
            U = node.Q + self.cpuct * node.P * stv / (1.0 + node.N)

            nz_index = U[node.LMnz].argmax()
            index = node.LMnz[0][nz_index]

        move = naf.policy_index_point(game.turn, index)

        if top:
            self.logger.debug("selecting index=%d move=%s Q=%.3f P=%.5f N=%d",
                              index, str(move), node.Q[index],
                              node.P[index], node.N[index])

        subnode = node.subnodes[index]

        game.play(move)

        if subnode:
            subscore = -self.visit_node(game, subnode)
        else:
            subnode = self.expand_leaf(game)
            node.subnodes[index] = subnode
            subscore = -subnode.score

        game.undo()

        node.N[index] += 1
        if subnode.proven:
            node.Q[index] = subscore
            node.LM[index] = 0
            node.LMnz = node.LM.nonzero()
            if subscore == 1:
                node.proven = True
                node.winning_move = move
            elif subscore == 0:
                node.drawing_move = move
        else:
            node.Q[index] += (subscore - node.Q[index]) / node.N[index]

        return subscore

    def compute_root(self, game):

        if not self.history_at_root:
            self.root = None
            return

        i = 0
        while i < len(game.history) and i < len(self.history_at_root):
            if game.history[i] != self.history_at_root[i]:
                break
            i += 1

        if i < len(self.history_at_root):
            self.history_at_root = None
            self.root = None
            return

        while i < len(game.history) and self.root:
            move = game.history[i]
            if not isinstance(move, Point):
                self.history_at_root = None
                self.root = None
                return
            color = (game.turn + len(game.history) - i) % 2
            index = naf.policy_point_index(color, move)
            self.root = self.root.subnodes[index]
            self.history_at_root.append(move)

        # finish compute_root

    def top_moves_str(self, game):
        indices = numpy.argsort(self.root.P[self.root.LMnz])
        pts = [str(naf.policy_index_point(game, self.root.LMnz[0][index]))
               for index in indices[-3:]]

        return ":" + ",".join(pts)

    def _scew(self, P):
        prob = P
        q = self.level
        # stochastic
        avg = 1.0 / len(prob)
        p0 = prob[0]
        # scew P:  greedy <-- stochastic --> random uniform
        # if q == 0.0 => set new p to avg
        # if q == 0.5 => set new p to p (no change)
        # if q == 1.0 => set new p[0] to 1, p[n>0] = 0 (greedy)
        prob = [(-4*p+2*avg)*q*q + (4*p-3*avg)*q + avg for p in prob[1:]]
        prob[0] = (-4*p0+2*avg+2)*q*q + (4*p0-3*avg-1)*q + avg
        return prob

    def eval_game(self, game, maxbest=twixt.MAXBEST):

        self.compute_root(game)
        # assert self.root == None
        self.root = self.expand_leaf(game)
        top_ixs = numpy.argsort(self.root.P)[-maxbest:]
        moves = [naf.policy_index_point(game, ix) for ix in top_ixs][::-1]
        P = [self.root.P[ix] for ix in top_ixs][::-1]
        # scew P
        Pscew = self._scew(P)

        self.logger.debug("moves: %s, idx: %s", moves, top_ixs)
        return self.root.score, moves, P, Pscew

    def proven_result(self, game):
        if self.root.winning_move:
            self.report = "fwin" + self.top_moves_str(game)
            return self.root.winning_move
        elif self.root.drawing_move:
            self.report = "fdraw"
            return self.root.drawing_move
        else:
            self.report = "flose"
            return twixt.RESIGN

    def create_response(self, game, status,
                        num_trials=0, current_trials=0,
                        proven=False, moves=None, P=None, Pscew=None):

        resp = {
            "status": status,
            "current": current_trials,
            "max": num_trials,
            "proven": proven
        }

        if P is not None:
            resp["P"] = P.tolist() if type(P) != list else P
        else:
            resp["P"] = [1.0]

        if Pscew is not None:
            resp["Pscew"] = Pscew.tolist() if type(Pscew) != list else P
        else:
            resp["Pscew"] = [1.0]

        if not moves:
            indices = numpy.argsort(self.root.N)[::-1][:twixt.MAXBEST]
            resp["moves"] = [naf.policy_index_point(
                game.turn, i) for i in indices]
            resp["Y"] = [int(n) for n in self.root.N[indices].tolist()]
            resp["P"] = [p for p in self.root.P[indices].tolist()]
            resp["Pscew"] = self._scew(resp["P"])
            # resp["Q"] = self.root.Q[indices].tolist()
        else:
            resp["moves"] = moves

        return resp

    def send_message(self, window, response):
        window.write_event_value('THREAD', response)

    def mcts(self, game, trials, window, event):
        """ Using the neural net, compute the move visit count vector """

        self.compute_root(game)
        if self.root is None:
            self.root = self.expand_leaf(game)
            self.history_at_root = list(game.history)
            top_ixs = numpy.argsort(self.root.P)[-5:]
            if self.logger.level <= logging.INFO:
                # if... to avoid unnecessary conversion
                msg = f'eval={self.root.score:.3f} '
                for ix in top_ixs:
                    msg += (f'{naf.policy_index_point(game, ix)}: '
                            f'{int(self.root.P[ix] * 10000 + 0.5)}')
                self.logger.info(msg)

        if not self.root.proven:
            # for i in tqdm(range(trials), ncols=100, desc="processing",
            # file=sys.stdout):
            path = []
            for i in range(trials):
                assert not self.root.proven
                self.visit_node(game, self.root, True,
                                trials - i)

                if self.root.proven:
                    break

                if event is not None and event.is_set():
                    break

                if (i + 1) % ct.MCTS_TRIAL_CHUNK == 0:
                    resp = self.create_response(
                        game, "in-progress", trials, i + 1, False)
                    self.send_message(window, resp)
                    if self.visualize_mcts:
                        self.clean_path(path)
                        self.traverse(game, path, 0, self.root)

        if self.visualize_mcts:
            self.clean_path(path)

        if self.root.proven:
            return self.proven_result(game)

        self.logger.debug("N=%s", self.root.N)
        self.logger.debug("Q=%s", self.root.Q)

        self.report = "%6.3f" % (
            self.root.Q[numpy.argmax(self.root.N)]) + self.top_moves_str(game)
        return self.root.N

    def clean_path(self, path):
        # remove current best path
        for m in path:
            for obj in m.objects:
                self.board.graph.delete_figure(obj)
        del path[:]

    def traverse(self, game, path, level, node):

        k = numpy.argmax(node.N)
        n = node.N[k]
        if n > 0:
            sn = node.subnodes[k]
            move = naf.policy_index_point(game.turn % 2, k)
            game.play(move)

            self.board.create_move_objects(len(game.history)-1, n)
            path.append(self.board.history[-1])

            if sn is not None:
                self.traverse(game, path, level + 1, sn)

            game.undo()
            self.board.history.pop()
