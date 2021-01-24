import PySimpleGUI as sg
import math
import twixt


# constants
LABEL_COLOR = "black"
LABEL_FONT = ("Arial", 9)
PEGHOLE_COLOR = "grey"
GUIDELINE_COLOR = "#b3b3b3"

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

    def __init__(self, settings):

        self.settings = settings
        self.size = twixt.Game.SIZE
        self.history = []
        self.known_moves = set()

        bs = self.settings['-BOARD_SIZE-']
        self.cell_width = bs / (twixt.Game.SIZE + 4)
        self.peg_radius = self.cell_width / 3.8
        self.hole_radius = self.cell_width / 10
        self.offset_factor = 2.5
        self.graph = sg.Graph(canvas_size=(bs, bs),
                              graph_bottom_left=(0, 0),
                              graph_top_right=(bs, bs),
                              background_color='lightgrey',
                              key='-BOARD-',
                              enable_events=True)

    def draw(self, game=None):

        self.graph.erase()
        self._draw_endlines()
        self._draw_pegholes()
        self._draw_labels()
        self._draw_guidelines()

        for idx in range(len(game.history)):
            self.create_move_objects(game, idx)

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

    def _draw_labels(self):
        if self.settings['-SHOW_LABELS-']:
            for i in range(self.size):
                row_label = "%d" % (self.size - i)
                # left row label
                self.graph.DrawText(row_label, ((self.offset_factor - 1) * self.cell_width,
                                                (i + self.offset_factor) * self.cell_width), LABEL_COLOR, LABEL_FONT)
                # right row label
                self.graph.DrawText(row_label, ((self.size + self.offset_factor) * self.cell_width,
                                                (i + self.offset_factor) * self.cell_width), LABEL_COLOR, LABEL_FONT)

            for i in range(self.size):
                col_label = chr(ord('A') + i)
                # top column label
                self.graph.DrawText(col_label,
                                    ((i + self.offset_factor) * self.cell_width,
                                     (self.offset_factor - 1) * self.cell_width), LABEL_COLOR, LABEL_FONT)
                # bottom column label
                self.graph.DrawText(col_label,
                                    ((i + self.offset_factor) * self.cell_width,
                                     (self.size + self.offset_factor) * self.cell_width), LABEL_COLOR, LABEL_FONT)

    def _draw_pegholes(self):
        for x in range(self.size):
            for y in range(self.size):
                if x in (0, self.size - 1) and y in (0, self.size - 1):
                    continue

                self.graph.DrawCircle(
                    ((x + self.offset_factor) * self.cell_width,
                     (y + self.offset_factor) * self.cell_width),
                    self.hole_radius, PEGHOLE_COLOR, PEGHOLE_COLOR)

    def _draw_endlines(self):
        o = 2 * self.cell_width
        s = self.size - 1
        w = self.cell_width
        self.graph.DrawLine((o + 1 * w, o + 1 * w + w / 3),
                            (o + 1 * w, o + s * w - w / 3), self.settings['-P2_COLOR-'], 3)
        self.graph.DrawLine((o + s * w, o + 1 * w + w / 3),
                            (o + s * w, o + s * w - w / 3), self.settings['-P2_COLOR-'], 3)
        self.graph.DrawLine((o + 1 * w + w / 3, o + 1 * w),
                            (o + s * w - w / 3, o + 1 * w), self.settings['-P1_COLOR-'], 3)
        self.graph.DrawLine((o + 1 * w + w / 3, o + s * w),
                            (o + s * w - w / 3, o + s * w), self.settings['-P1_COLOR-'], 3)

    def _draw_guidelines(self):
        if self.settings['-SHOW_GUIDELINES-']:
            for p in [((1,  1), (15,  8)),
                      ((15,  8), (22, 22)),
                      ((22, 22), (8, 15)),
                      ((8, 15), (1, 1)),
                      ((1, 22), (15, 15)),
                      ((15, 15), (22,  1)),
                      ((22,  1), (8,  8)),
                      ((8,  8), (1, 22))]:
                self.graph.DrawLine(((p[0][0] + self.offset_factor) * self.cell_width,
                                     (p[0][1] + self.offset_factor) * self.cell_width),
                                    ((p[1][0] + self.offset_factor) * self.cell_width,
                                     (p[1][1] + self.offset_factor) * self.cell_width),
                                    GUIDELINE_COLOR, .5)

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

    def _create_drawn_peg(self, point, coloridx):
        if coloridx == 1:
            color = self.settings['-P1_COLOR-']
        else:
            color = self.settings['-P2_COLOR-']

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
        carray = [self.settings['-P2_COLOR-'],
                  self.settings['-P1_COLOR-'],
                  self.settings['-P1_COLOR-']]

        line = self.graph.DrawLine(self._point_to_coords(
            p1), self._point_to_coords(p2), carray[color], 5)
        return line

    def validSpot(self, move, game):

        p = self._move_to_point(move)
        if len(game.history) >= 2 and game.history[1] == "swap":
            # game is swapped
            if p in game.history[2:]:
                # spot occupied after swap
                return False
            elif twixt.Point(p.y, p.x) == game.history[0]:
                # spot occupied on swap (flip)
                return False
            else:
                return True
        else:
            # game is not swapped
            if p in game.history:
                # spot occupied
                return False

        return True

    def getMove(self, game, coords):

        offset = self.offset_factor * self.cell_width
        x = (coords[0] - offset) / self.cell_width
        y = (coords[1] - offset) / self.cell_width

        # click distance tolerance
        maxgap = 0.3
        xgap = abs(x - round(x))
        ygap = abs(y - round(y))
        if xgap > maxgap or ygap > maxgap:
            # click not on hole
            return None

        x = round(x)
        y = round(y)
        move = chr(ord('a') + x) + "%d" % (self.size - y)
        if len(game.history) == 1 and self._move_to_point(
                move) == game.history[0] and game.allow_swap:
            return "swap"

        if x < 0 or x > self.size - 1 or y < 0 or y > self.size - 1:
            # overboard click
            return None
        elif (x == 0 or x == self.size - 1) and (y == 0 or y == self.size - 1):
            # corner click
            return None
        elif ((x == 0 or x == self.size - 1) and len(game.history) % 2 == 0):
            # white clicked on black's end line
            return None
        elif ((y == 0 or y == self.size - 1) and len(game.history) % 2 == 1):
            # black clicked white's end line
            return None

        if not self.validSpot(move, game):
            return None

        return move
