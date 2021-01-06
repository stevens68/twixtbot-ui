#! /usr/bin/env python
import math
import numpy
import random
import select

import naf
import smmpp
import swapmodel
import twixt

debug = False

class Node:
    def __init__(self):
        K = twixt.Game.SIZE * (twixt.Game.SIZE - 2)
        self.Ns = numpy.zeros(K, numpy.int_)
        self.Nf = numpy.zeros(K, numpy.int_)
        self.Q = numpy.zeros(K, numpy.float32)
        self.P = numpy.zeros(K, numpy.float32)

        self.proven = False
        self.score = None
        self.winning_index = None
        self.subnodes = [None]*K

class Player:
    def __init__(self, **kwargs):
        self._init_params(kwargs)
        self._init_vars()
        self._init_client()

    def _init_vars(self):
        self.game = None
        self.root = None
        self.history_at_root = None
        self.leaves_waiting = 0


    def _init_params(self, kwargs):
        self.location = kwargs.pop('location')
        self.trials = int(kwargs.pop('trials'))
        self.async_calls = int(kwargs.pop('async_calls', 8))
        self.quiet = int(kwargs.pop('quiet', 0))
        self.use_swap = int(kwargs.pop('use_swap', 1))
        kwargs.pop('resources')

        if kwargs:
            raise TypeError("unprocessed options", kwargs)

    # NN client handler code

    def _init_client(self):
        self.client = smmpp.Client(self.location, self.async_calls, quiet=self.quiet)

    def _block_until_reply(self):
        self.client.handle_read()

    def _process_incoming_data(self, request_block):
        if debug:
            print "_pid, lw=", self.leaves_waiting
        incoming_data = request_block
        if self.leaves_waiting >= self.async_calls:
            incoming_data = True
            assert self.leaves_waiting == self.async_calls
        elif self.leaves_waiting > 0:
            rl, _, _ = select.select([self.client.socket], [], [], 0)
            if rl:
                incoming_data = True

        if incoming_data:
            self._block_until_reply()

    # MCTS stuff

    def _preload_root(self):
        if not self.history_at_root:
            self.root = None
            return

        i = 0
        while i < len(self.game.history) and i < len(self.history_at_root):
            if self.game.history[i] != self.history_at_root[i]:
                break
            i += 1

        if i < len(self.history_at_root):
            self.root = None
            return

        while i < len(self.game.history) and self.root:
            move = self.game.history[i]
            if not isinstance(move, twixt.Point):
                self.root = None
                return

            color = (self.game.turn + len(self.game.history) - i) % 2
            index = naf.policy_point_index(color, move)
            self.root = self.root.subnodes[index]
            self.history_at_root.append(move)
        # end _preload_root

    def _expand_leaf(self, parent, parent_index):
        if debug and parent is not None:
            print "expand_leaf @",self._stack_string()
        leaf = Node()
        leaf.parent = parent
        leaf.parent_index = parent_index

        if self.game.just_won():
            leaf.proven = True
            leaf.winning_index = None
            leaf.score = -1
            self.finished_leaves.append(leaf)
            return leaf

        leaf.LM = naf.legal_move_policy_array(self.game)
        assert leaf.LM.any()
        leaf.LMnz = leaf.LM.nonzero()

        nips = naf.NetInputs(self.game)
        rot = random.randint(0, 3)
        nips.rotate(rot)
        outbytes = nips.to_expanded_bytes()

        def set_reply(reply):
            p0 = numpy.frombuffer(reply, dtype=numpy.float32)
            nml = twixt.Game.SIZE * (twixt.Game.SIZE-2)

            if p0.shape[0] == nml + 1:
                leaf.score = p0[0]
                movelogits = naf.rotate_policy_array(p0[1:], rot)
            elif p0.shape[0] == nml + 3:
                leaf.score = naf.three_to_one(p0[0:3])
                movelogits = naf.rotate_policy_array(p0[3:], rot)
            else:
                raise TypeError("Bad shape:", p0.shape)

            maxlogit = movelogits[leaf.LMnz].max()
            el = numpy.exp(movelogits - maxlogit)
            divisor = el[leaf.LMnz].sum()
            leaf.P = el / divisor
            self.finished_leaves.append(leaf)
            assert self.leaves_waiting > 0
            self.leaves_waiting -= 1

        self.leaves_waiting += 1
        self.num_evals += 1
        self.client.write_query(outbytes, set_reply)
        return leaf
        # end _expand_leaf()

    def _node_to_pv(self, node, color):
        descend_index = None
        if node.winning_index is not None:
            descend_index = node.winning_index
        else:
            nmax = node.Nf.max()
            if nmax < 2:
                return []
            Nfnz = (node.Nf == nmax).nonzero()
            i1 = node.Q[Nfnz].argmax()
            i2 = Nfnz[0][i1]
            descend_index = i2

        move = naf.policy_index_point(color, descend_index)
        sub = self._node_to_pv(node.subnodes[descend_index], 1-color)
        sub.insert(0, str(move))
        return sub

    def _principal_var_str(self):
        pv = self._node_to_pv(self.root, self.game.turn)
        return ":pv=" + ",".join(pv)

    def _top_moves_str(self):
        indices = numpy.argsort(self.root.P[self.root.LMnz])
        pts = [str(naf.policy_index_point(self.game, self.root.LMnz[0][index])) for index in indices[-3:]]
        return ":" + ",".join(pts)

    def _stack_string(self):
        if self.history_at_root is None:
            return "root"
        n = len(self.history_at_root)
        return ",".join(map(str,self.game.history[n:]))

    def _go_to_child(self, parent, child_index):
        if debug:
            print "go_to @",self._stack_string()
        move = naf.policy_index_point(self.game.turn, child_index)
        subnode = parent.subnodes[child_index]
        self.game.play(move)
        parent.Ns[child_index] += 1

        if subnode:
            gc_index = self._pick_expand_index(subnode)
            if gc_index == "wait":
                parent.Ns[child_index] -= 1
                r = 0
            elif gc_index == "lose":
                subnode.proven = True
                subnode.winning_index = None
                subnode.score = -1
                self.finished_leves.append(subnode)
                r = 1
            else:
                assert gc_index >= 0
                r = self._go_to_child(subnode, gc_index)
        else:
            parent.subnodes[child_index] = self._expand_leaf(parent, child_index)
            r = 1

        self.game.undo()
        return r

    def _process_finished_leaves(self):
        while self.finished_leaves:
            leaf = self.finished_leaves.pop()
            score = leaf.score
            c = leaf
            while c.parent is not None:
                score = -score
                p = c.parent
                index = c.parent_index

                p.Nf[index] += 1
                assert p.Ns[index] >= p.Nf[index]
                if c.proven:
                    p.Q[index] = score
                    p.LM[index] = 0
                    p.LMnz = p.LM.nonzero()
                    if score == 1:
                        p.proven = True
                        p.winning_index = index
                    elif numpy.count_nonzero(p.LM) == 0:
                        p.proven = True
                        p.winning_index = None
                        p.score = -1
                else:
                    if debug:
                        print "index=",index,"score=",score,"pqi=",p.Q[index],"pnf=",p.Nf[index]
                    p.Q[index] += (score - p.Q[index]) / p.Nf[index]

                c = p
            # end while c.parent
        # end _process_finished_leaves

    def _pick_expand_index(self, node):
        # Pick a move to expand not at the root.
        check = (node.LM != 0) & (node.Ns + node.Nf != 1)
        if not check.any():
            waiters = (node.Ns + node.Nf == 1)
            if waiters.any():
                #print "list of waiters:", numpy.where(node.Ns + node.Nf == 1)
                #print "num waiters", self.leaves_waiting
                return "wait"
            else:
                return "lose"

        nsum = node.Ns[check].sum()
        stv = math.sqrt(nsum + 1.0)
        U = node.Q + node.P*stv/(1.0 + node.Ns)
        nz_index = U[check].argmax()
        index = check.nonzero()[0][nz_index]
        return index

    def _pick_root_expand_index(self, trials_left):
        node = self.root
        assert not node.proven
        check = (node.LM != 0) & (node.Ns + node.Nf != 1)
        if numpy.count_nonzero(check) == 0:
            return -1
        maxn = node.Ns[check].max()
        can_win = check & (node.Ns > maxn - trials_left)
        num_can_win = numpy.count_nonzero(can_win)

        if num_can_win == 0:
            return -1
        if num_can_win == 1:
            return check.argmax()

        maxes = can_win & (node.Ns == maxn)
        num_maxes = numpy.count_nonzero(maxes)
        assert num_maxes > 0, (can_win, maxn, node.Ns, trials_left)
        if num_maxes == 1 and num_can_win > 1:
            can_win[maxes.argmax()] = False

        nsum = node.Ns[can_win].sum()
        stv = math.sqrt(nsum + 1.0)
        U = node.Q + node.P*stv/(1.0 + node.Ns)
        nz_index = U[can_win].argmax()
        return can_win.nonzero()[0][nz_index]
        
 
    def pick_move(self, game, ctrlWindow=None):
        if len(game.history) == 0:
            self.report = "swapmodel"
            return swapmodel.choose_first_move()
        elif len(game.history) == 1 and self.use_swap:
            if swapmodel.want_swap(game.history[0]):
                self.report = "swapmodel"
                return "swap"

        # If we get here, it is move 2+ and either we cannot or do not swap.
        self.game = game
        self._preload_root()
        self.finished_leaves = []
        self.leaves_waiting = 0
        self.num_evals = 0

        ctrlWindow.updateProgress(self.num_evals, self.trials, None)

        if self.root == None:
            self.root = self._expand_leaf(None, None)
            self.history_at_root = list(self.game.history)
            self._block_until_reply()
            #num_evals = 1

        # Let's start thinkin'!

        request_block = False
        while not self.root.proven and self.num_evals < self.trials:
            good_moves = numpy.where(self.root.Nf == self.root.Nf.max()) #MB
            #print self.num_evals, good_moves[0]         #MB
            self._process_incoming_data(request_block)
            request_block = False
            self._process_finished_leaves()

            if (ctrlWindow != None and (self.num_evals) % 20 == 0):
                ctrlWindow.updateProgress(self.num_evals, self.trials, good_moves)

            if self.root.proven:
                break

            ei = self._pick_root_expand_index(self.trials - self.num_evals)
            if ei >= 0:
                r = self._go_to_child(self.root, ei)
                if r == 0:
                    assert self.leaves_waiting > 0
                    request_block = True
            elif self.leaves_waiting == 0:
                # no moves worth evaluating; means we lost.
                self.root.proven = True
            else:
                request_block = True

        ctrlWindow.updateProgress(self.num_evals, self.trials, good_moves)

        # Okay, done thinkin'.  flush out any waiting kids
        while self.leaves_waiting > 0:
            self._block_until_reply()
        self._process_finished_leaves()

        if self.root.proven:
            if self.root.winning_index is not None:
                self.report = "fwin"
                return naf.policy_index_point(game, self.root.winning_index)
            else:
                self.report = "flose"
                return "resign"

        assert numpy.array_equal(self.root.Nf, self.root.Ns), (self.root.Nf.sum(), self.root.Ns.sum())
        good_moves = numpy.where(self.root.Nf == self.root.Nf.max())
        index = random.choice(good_moves[0])
        move = naf.policy_index_point(game, index)

        self.report = "%6.3f %s" % (self.root.Q[index], self._principal_var_str())
        return move
        

if __name__ == "__main__":
    p = Player(location="/tmp/foo", trials="1500",goof="33")
