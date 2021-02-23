
import backend.twixt as twixt
import constants as ct


def if_else0(condition, value):
    if condition:
        return value
    else:
        return 0


class TBWHistory:
    def __init__(self, move):
        self.move = move
        self.objects = []


class TwixtBoard:

    def __init__(self, stgs):

        self.size = twixt.Game.SIZE
        self.history = []
        self.known_moves = set()

        self.stgs = stgs

    def _point_to_move(self, point):
        if isinstance(point, twixt.Point):
            return chr(ord('a') + point[0]) + "%d" % (twixt.Game.SIZE - point[1])
        else:
            # swap case
            return point

    def _point_to_coords(self, point):
        return ((point[0] + self.offset_factor) * self.cell_width,
                ((self.size - point[1] - 1) + self.offset_factor) * self.cell_width)

    def _move_to_point(self, move):
        return twixt.Point(ord(move[0]) - ord('a'), int(move[1:]) - 1)

    def set_game(self, game):
        i = 0
        while i < len(game.history) and i < len(self.history):
            if game.history[i] != self.history[i].move:
                break
            i += 1

        # Get rid of now unneeded objects
        if i < len(self.history):
            for h in reversed(self.history[i:]):
                for o in h.objects:
                    self.graph.delete(o)
                self.known_moves.remove(h.move)
            self.history = self.history[:i]

        # Add new objects
        while i < len(game.history):
            self.create_move_objects(game, i)
            i += 1

    def set_naf(self, naf, rotate=False):
        for h in reversed(self.history):
            for o in h.objects:
                self.graph.delete(o)

        self.history = []
        self.known_moves = set()

        if not rotate:
            def xyp(x, y): return twixt.Point(x, y)

            def pp(p): return p

            def cp(c): return 1 - c
        else:
            def xyp(x, y): return twixt.Point(y, x)

            def cp(c): return c

            def pp(p): return twixt.Point(p.y, p.x)

        objs = []

        for x, y, i in zip(*naf[:, :, 8:].nonzero()):
            objs.append(self._create_drawn_peg(xyp(x, y), cp(i & 1)))

        for x, y, j in zip(*naf[:, :, :8].nonzero()):
            link = twixt.Game.describe_link(j, x, y)
            objs.append(self._create_drawn_link(
                pp(link.p1), pp(link.p2), cp(i & 1)))

        nho = TBWHistory("nninputs")
        self.history.append(nho)
        nho.objects = objs

    def set_nn_inputs(self, pegs, links, rotate=False):
        for h in reversed(self.history):
            for o in h.objects:
                self.graph.delete(o)

        self.history = []
        self.known_moves = set()

        if not rotate:
            def xyp(x, y): return twixt.Point(x, y)

            def pp(p): return p

            def cp(c): return 1 - c
        else:
            def xyp(x, y): return twixt.Point(y, x)

            def cp(c): return c

            def pp(p): return twixt.Point(p.y, p.x)

        objs = []

        for x, y, i in zip(*pegs.nonzero()):
            objs.append(self._create_drawn_peg(xyp(x, y), cp(i)))

        i_px_py = []
        # color = game.WHITE
        for vertical in [False, True]:
            for diff_sign in [False, True]:
                for as_me in [False, True]:
                    index = if_else0(vertical, twixt.Game.LINK_LONGY)
                    index += if_else0(diff_sign, twixt.Game.LINK_DIFFSIGN)
                    index += 1 if as_me else 0
                    pad_x = if_else0(vertical or diff_sign, 1)
                    pad_y = if_else0(not vertical or diff_sign, 1)

                    i_px_py.append((index, pad_x, pad_y))

        for x, y, j in zip(*links.nonzero()):
            index, pad_x, pad_y = i_px_py[j]
            lx = x - pad_x
            ly = y - pad_y
            desc = twixt.Game.describe_link(index, lx, ly)
            c1 = 2 - 2 * pegs[desc.p1.x, desc.p1.y, 0] - \
                pegs[desc.p1.x, desc.p1.y, 1]
            c2 = 2 - 2 * pegs[desc.p2.x, desc.p2.y, 0] - \
                pegs[desc.p2.x, desc.p2.y, 1]
            if c1 == 0 and c2 == 0:
                color = 0
            elif c1 == 1 and c2 == 1:
                color = 1
            else:
                color = 2
            objs.append(self._create_drawn_link(
                pp(desc.p1), pp(desc.p2), color))

        nho = TBWHistory("nninputs")
        self.history.append(nho)
        nho.objects = objs
        # end set_nn_inputs

    def _create_drawn_peg(self, point, coloridx):
        if coloridx == 1:
            color = self.stgs.get_setting(ct.K_COLOR[1])
        else:
            color = self.stgs.get_setting(ct.K_COLOR[2])

        peg = self.graph.DrawCircle(self._point_to_coords(
            point), self.peg_radius, color, color)
        return peg

    def create_move_objects(self, game, index):
        move = game.history[index]
        if not isinstance(move, twixt.Point):
            # swap case: flip first move
            c1 = self.history.pop().objects[0]
            self.graph.delete_figure(c1)
            self.known_moves.clear()
            m1 = game.history[0]
            move = twixt.Point(m1.y, m1.x)
            # game.use_swap = True

        color = (index + 1) & 1

        nho = TBWHistory(move)
        self.history.append(nho)

        nho.objects.append(self._create_drawn_peg(move, color))
        self.known_moves.add(move)

        for dlink in game.DLINKS:
            other = move + dlink
            if other in self.known_moves:
                if game.safe_get_link(move, other, color):
                    nho.objects.append(
                        self._create_drawn_link(move, other, color))

        # end create_move_objects()

    def _create_drawn_link(self, p1, p2, color):
        # carray = [gr.color_rgb(0,0,0), gr.color_rgb(150,150,150), gr.color_rgb(255,0,0)]
        carray = [self.stgs.get_setting(ct.K_COLOR[2]),
                  self.stgs.get_setting(ct.K_COLOR[1]),
                  self.stgs.get_setting(ct.K_COLOR[1])]

        line = self.graph.DrawLine(self._point_to_coords(
            p1), self._point_to_coords(p2), carray[color], 5)
        return line
