import math
import twixt


# constants
COLOR_PLAYER1 = "red"
COLOR_PLAYER2 = "black"
COLOR_BOARD = "lightgrey"
COLOR_LABELS = "black"
FONT_LABELS = "Arial"
SIZE_LABELS = 9
COLOR_PEGHOLES = "grey"
COLOR_GUIDELINES = "#b3b3b3"


# helper


def ifelse0(condition, value):
    if condition:
        return value
    else:
        return 0


class TBWHistory:
    def __init__(self, move):
        self.move = move
        self.objects = []


class TwixtBoard:

    def __init__(self, graph):
        self.graph = graph
        self.size = twixt.Game.SIZE
        self.cell_width = self.graph.Size[0] / (twixt.Game.SIZE + 4)
        self.peg_radius = self.cell_width / 4
        self.hole_radius = 2

        self.history = []
        self.known_moves = set()

        self.draw_endlines()
        self.draw_pegholes()
        self.draw_labels()
        self.draw_guidelines()

    def draw_labels(self):
        for i in range(self.size):
            row_label = "%d" % (self.size - i)
            # left row label
            self.graph.DrawText(row_label,
                                (1.5 * self.cell_width,  (i + 2.5) * self.cell_width), COLOR_LABELS, (FONT_LABELS, SIZE_LABELS))
            # right row label
            self.graph.DrawText(row_label,
                                ((self.size + 2.5) * self.cell_width,
                                 (i + 2.5) * self.cell_width), COLOR_LABELS, (FONT_LABELS, SIZE_LABELS))

        for i in range(self.size):
            col_label = chr(ord('A') + i)
            # top column label
            self.graph.DrawText(col_label,
                                ((i + 2.5) * self.cell_width,
                                 1.5 * self.cell_width), COLOR_LABELS, (FONT_LABELS, SIZE_LABELS))
            # bottom column label
            self.graph.DrawText(col_label,
                                ((i + 2.5) * self.cell_width,
                                 (self.size + 2.5) * self.cell_width), COLOR_LABELS, (FONT_LABELS, SIZE_LABELS))

    def draw_pegholes(self):
        offset = 2.5
        for x in range(self.size):
            for y in range(self.size):
                if x in (0, self.size - 1) and y in (0, self.size - 1):
                    continue

                circle = self.graph.DrawCircle(
                    ((x + offset) * self.cell_width,
                     (y + offset) * self.cell_width),
                    self.hole_radius, COLOR_PEGHOLES, COLOR_PEGHOLES)

    def draw_endlines(self):
        o = 2 * self.cell_width
        s = self.size - 1
        w = self.cell_width
        self.graph.DrawLine((o + 1 * w, o + 1 * w + w / 3),
                            (o + 1 * w, o + s * w - w / 3), COLOR_PLAYER2, 3)
        self.graph.DrawLine((o + s * w, o + 1 * w + w / 3),
                            (o + s * w, o + s * w - w / 3), COLOR_PLAYER2, 3)
        self.graph.DrawLine((o + 1 * w + w / 3, o + 1 * w),
                            (o + s * w - w / 3, o + 1 * w), COLOR_PLAYER1, 3)
        self.graph.DrawLine((o + 1 * w + w / 3, o + s * w),
                            (o + s * w - w / 3, o + s * w), COLOR_PLAYER1, 3)

    def draw_guidelines(self):
        offset = 2.5
        for p in [((1,  1), (15,  8)),
                  ((15,  8), (22, 22)),
                  ((22, 22), (8, 15)),
                  ((8, 15), (1, 1)),
                  ((1, 22), (15, 15)),
                  ((15, 15), (22,  1)),
                  ((22,  1), (8,  8)),
                  ((8,  8), (1, 22))]:
            self.graph.DrawLine(((p[0][0] + offset) * self.cell_width, (p[0][1] + offset) * self.cell_width),
                                ((p[1][0] + offset) * self.cell_width,
                                 (p[1][1] + offset) * self.cell_width),
                                COLOR_GUIDELINES, .5)

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
                    index = ifelse0(vertical, twixt.Game.LINK_LONGY)
                    index += ifelse0(diff_sign, twixt.Game.LINK_DIFFSIGN)
                    index += 1 if as_me else 0
                    pad_x = ifelse0(vertical or diff_sign, 1)
                    pad_y = ifelse0(not vertical or diff_sign, 1)

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

    def _create_drawn_peg(self, point, color):
        peg = self.graph.DrawCircle(self.center(
            point.x, point.y), self.peg_radius, color, color)
        return peg

    def create_move_objects(self, game, index):
        move = game.history[index]
        if (not isinstance(move, twixt.Point)):
            # swap case: flip first move
            c1 = self.history.pop().objects[0]
            self.graph.delete(c1)
            self.known_moves.clear()
            m1 = game.history[0]
            move = twixt.Point(m1.y, m1.x)
            game.use_swap = True

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
        #carray = [gr.color_rgb(0,0,0), gr.color_rgb(150,150,150), gr.color_rgb(255,0,0)]
        carray = [COLOR_PLAYER2, COLOR_PLAYER1, COLOR_PLAYER1]
        c1 = self.center(*p1)
        c2 = self.center(*p2)
        dx = c2.x - c1.x
        dy = c2.y - c1.y
        hypot = math.hypot(dx, dy)
        cos = dx / hypot
        sin = dy / hypot
        lx1 = int(c1.x + cos * self.peg_radius + 0.5)
        ly1 = int(c1.y + sin * self.peg_radius + 0.5)
        lx2 = int(c2.x - cos * self.peg_radius + 0.5)
        ly2 = int(c2.y - sin * self.peg_radius + 0.5)
        line = self.graph.DrawLine((lx1, ly1), (lx2, ly2), carray[color], 5)
        return line
