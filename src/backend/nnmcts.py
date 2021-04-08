#! /usr/bin/env python
import math
import numpy

import backend.naf as naf
import backend.twixt as twixt



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
        self.verbosity = kwargs.pop("verbosity", 0)
        self.smart_root = kwargs.pop("smart_root", 0)
        self.smart_init = kwargs.pop("smart_init", 0)
        self.board = kwargs.pop("board", None)

        if kwargs:
            raise TypeError('Unexpected kwargs provided: %s' %
                            list(kwargs.keys()))
        self.sap = sap
        self.root = None
        self.history_at_root = None
        
        self.path = []

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

        if self.verbosity >= 3:
            print("LMnz:", leaf.LMnz)
            print("raw P:", leaf.P)

        if self.add_noise:
            leaf.P[leaf.LMnz] *= (1.0 - self.add_noise)
            leaf.P[leaf.LMnz] += self.add_noise * \
                numpy.random.dirichlet(0.03 * numpy.ones(len(leaf.LMnz[0])))

        if self.verbosity >= 3:
            print("after noise P:", leaf.P)
        return leaf
        # end expand_leaf()

    def visit_node(self, game, node, top=False, trials=None, trials_left=-1, window=None, path=None):
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
            maxn = node.N[node.LMnz].max()
            winnables = (node.N > maxn - trials_left) & node.LM
            num_winnables = numpy.count_nonzero(winnables)
            assert num_winnables > 0, (maxn, trials_left, numpy.array_str(
                node.N), numpy.array_str(node.LM))
            if num_winnables == 1:
                index = winnables.argmax()
            else:
                maxes = (node.N == maxn)
                num_maxes = numpy.count_nonzero(maxes)
                assert num_maxes > 0
                if num_maxes == 1:
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

        if top and self.verbosity >= 3 and U:
            for i in range(len(U)):
                _ = naf.policy_index_point(game.turn, i)

        if top and self.verbosity > 1:
            print("selecting index=%d move=%s Q=%.3f P=%.5f N=%d" %
                  (index, str(move), node.Q[index], node.P[index], node.N[index]))

        subnode = node.subnodes[index]

        game.play(move)
        path.append(move)
        idx = len(path)-1
        if len(self.path) < len(path):
            self.path.append(move)
        elif path[idx] != self.path[idx]:
            #print("   trunctating self.path at", idx)
            del self.path[idx:]
            self.path.append(move)
        
        # self.board.mcts_update(move)
        if subnode:
            subscore = -self.visit_node(game, subnode, path=path)
        else:
            #print(trials, "expanding", path)
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
            if not isinstance(move, twixt.Point):
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

    def eval_game(self, game, maxbest=twixt.MAXBEST):

        self.compute_root(game)
        # assert self.root == None
        self.root = self.expand_leaf(game)
        top_ixs = numpy.argsort(self.root.P)[-maxbest:]
        moves = [naf.policy_index_point(game, ix) for ix in top_ixs][::-1]
        P = [int(round(self.root.P[ix] * 1000)) for ix in top_ixs][::-1]
        # print("moves: ", moves, ", idx: ",  top_ixs)
        return self.root.score, moves, P

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

    def send_message(self, window, game, status, num_trials=0, current_trials=0, proven=False, moves=None, P=None):

        resp = {
            "status": status,
            "current": current_trials,
            "max": num_trials,
            "proven": proven
        }

        if P is not None:
            resp["P"] = P.tolist() if type(P) != list else P

        if not moves:
            indices = numpy.argsort(self.root.N)[::-1][:twixt.MAXBEST]
            resp["moves"] = [naf.policy_index_point(
                game.turn, i) for i in indices]
            resp["Y"] = [int(n) for n in self.root.N[indices].tolist()]
            resp["P"] = [int(round(p * 1000))
                         for p in self.root.P[indices].tolist()]
            # resp["Q"] = self.root.Q[indices].tolist()
        else:
            resp["moves"] = moves

        window.write_event_value('THREAD', resp)

    def mcts(self, game, trials, window, event):
        """ Using the neural net, compute the move visit count vector """

        self.compute_root(game)
        if self.root is None:
            self.root = self.expand_leaf(game)
            self.history_at_root = list(game.history)
            if self.verbosity >= 1:
                top_ixs = numpy.argsort(self.root.P)[-5:]
                print("eval=%.3f  %s" % (self.root.score, " ".join(["%s:%d" % (naf.policy_index_point(
                    game, ix), int(self.root.P[ix] * 10000 + 0.5)) for ix in top_ixs])))

        if not self.root.proven:
            # for i in tqdm(range(trials), ncols=100, desc="processing",
            # file=sys.stdout):
            for i in range(trials):
                path = []
                assert not self.root.proven
                self.visit_node(game, self.root, True,
                                trials - i, window=window, path=path)

                if self.root.proven:
                    break

                if event is not None and event.is_set():
                    break

                if (i + 1) % 20 == 0:
                    #self.traverse(0, self.root)
                    self.send_message(
                        window, game, "in-progress", trials, i + 1, False)

        if self.root.proven:
            return self.proven_result(game)

        if self.verbosity >= 2:
            print("N=", self.root.N)
            print("Q=", self.root.Q)

        self.report = "%6.3f" % (
            self.root.Q[numpy.argmax(self.root.N)]) + self.top_moves_str(game)
        return self.root.N

    def traverse(self, level, node):

        k = numpy.argmax(node.N)
        n = node.N[k]
        sn = node.subnodes[k]

        print("l:", level, "k:", k, "n:", n)

        if sn is not None:
            self.traverse(level + 1, sn)
