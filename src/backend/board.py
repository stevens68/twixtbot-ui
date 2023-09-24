import backend.twixt as twixt
import constants as ct
from PySimpleGUI.PySimpleGUI import TEXT_LOCATION_BOTTOM_LEFT
from backend.point import Point


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
        self.visit_offset = self.stgs.get(ct.K_BOARD_SIZE[1]) / 120

    def _point_to_coords(self, point):
        return ((point[0] + self.offset_factor) * self.cell_width,
                ((self.size - point[1] - 1) + self.offset_factor) *
                self.cell_width)

    def _move_to_point(self, move):
        return Point(ord(move[0]) - ord('a'), int(move[1:]) - 1)

    def _create_drawn_peg(self, point, coloridx,
                          highlight_last_move=False, visits=None):

        if coloridx == 1:
            color = self.stgs.get(ct.K_COLOR[1])
        else:
            color = self.stgs.get(ct.K_COLOR[2])

        if highlight_last_move:
            pr = self.peg_radius + 1
            lw = 2
            line_color = ct.HIGHLIGHT_LAST_MOVE_COLOR
        else:
            pr = self.peg_radius
            lw = 1
            line_color = color

        if visits is not None:
            fill_color = None
        else:
            fill_color = color

        peg = self.graph.DrawCircle(self._point_to_coords(
            point), pr, fill_color, line_color, lw)

        return peg

    def _create_visits_label(self, point, coloridx, visits):

        if coloridx == 1:
            color = self.stgs.get(ct.K_COLOR[1])
        else:
            color = self.stgs.get(ct.K_COLOR[2])

        (x, y) = self._point_to_coords(point)
        label = self.graph.DrawText(str(int(visits)), (x, y+self.visit_offset),
                                    color, font=ct.VISITS_LABEL_FONT,
                                    text_location=TEXT_LOCATION_BOTTOM_LEFT)

        return label

    def create_move_objects(self, game, index, visits=None):
        move = game.history[index]
        if not isinstance(move, Point):
            # swap case: flip first move
            if len(self.history) > 0:
                c1 = self.history.pop().objects[0]
                self.graph.delete_figure(c1)
            self.known_moves.clear()
            m1 = game.history[0]
            move = Point(m1.y, m1.x)
            # game.use_swap = True

        color = (index + 1) & 1

        nho = TBWHistory(move)
        self.history.append(nho)

        highlight_last_move = visits is None and self.stgs.get(
            ct.K_HIGHLIGHT_LAST_MOVE[1]) and index == len(game.history) - 1
        nho.objects.append(self._create_drawn_peg(
            move, color, highlight_last_move, visits))
        self.known_moves.add(move)
        if visits is not None:
            nho.objects.append(self._create_visits_label(move, color, visits))

        for dlink in game.DLINKS:
            other = move + dlink
            if other in self.known_moves:
                if game.safe_get_link(move, other, color):
                    nho.objects.append(
                        self._create_drawn_link(move, other, color, visits))

    def undo_last_move_objects(self):
        if len(self.history) > 0:
            m = self.history.pop()
            if m.objects is not None:
                for obj in m.objects:
                    self.graph.delete_figure(obj)

    def _create_drawn_link(self, p1, p2, color, visits=None):
        # carray = [gr.color_rgb(0,0,0),
        #           gr.color_rgb(150,150,150),
        #           gr.color_rgb(255,0,0)]
        carray = [self.stgs.get(ct.K_COLOR[2]),
                  self.stgs.get(ct.K_COLOR[1]),
                  self.stgs.get(ct.K_COLOR[1])]

        lw = 2 if visits is not None else 5

        line = self.graph.DrawLine(self._point_to_coords(
            p1), self._point_to_coords(p2), carray[color], lw)
        return line
