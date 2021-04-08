
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


    def _point_to_coords(self, point):
        return ((point[0] + self.offset_factor) * self.cell_width,
                ((self.size - point[1] - 1) + self.offset_factor) * self.cell_width)

    def _move_to_point(self, move):
        return twixt.Point(ord(move[0]) - ord('a'), int(move[1:]) - 1)



    def _create_drawn_peg(self, point, coloridx, highlight_last_move=False):
        if coloridx == 1:
            color = self.stgs.get(ct.K_COLOR[1])
        else:
            color = self.stgs.get(ct.K_COLOR[2])

        if highlight_last_move:
            peg = self.graph.DrawCircle(self._point_to_coords(
                point), self.peg_radius + 1, color, ct.HIGHLIGHT_LAST_MOVE_COLOR, 2)
        else:
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

        highlight_last_move = self.stgs.get(
            ct.K_HIGHLIGHT_LAST_MOVE[1]) and index == len(game.history) - 1
        nho.objects.append(self._create_drawn_peg(
            move, color, highlight_last_move))
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
        carray = [self.stgs.get(ct.K_COLOR[2]),
                  self.stgs.get(ct.K_COLOR[1]),
                  self.stgs.get(ct.K_COLOR[1])]

        line = self.graph.DrawLine(self._point_to_coords(
            p1), self._point_to_coords(p2), carray[color], 5)
        return line
