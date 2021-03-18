import numpy
import PySimpleGUI as sg
import backend.board as board
import backend.twixt as twixt
import constants as ct


class UiBoard(board.TwixtBoard):

    def __init__(self, game, stgs):
        super().__init__(stgs)

        self.game = game

        bp = stgs.get_setting(ct.K_BOARD_SIZE[1])
        self.cell_width = bp / (twixt.Game.SIZE + 4)
        self.peg_radius = self.cell_width / 3.8
        self.hole_radius = self.cell_width / 10
        self.offset_factor = 2.5
        self.graph = sg.Graph(canvas_size=(bp, bp),
                              graph_bottom_left=(0, 0),
                              graph_top_right=(bp, bp),
                              background_color=ct.FIELD_BACKGROUND_COLOR,
                              key=ct.K_BOARD[1],
                              enable_events=True)

    def draw(self, heatmap=None):
        self.graph.erase()
        self._draw_endlines()
        self._draw_pegholes(heatmap)
        self._draw_labels()
        self._draw_guidelines()

        if self.game.history is not None:
            for idx in range(len(self.game.history)):
                self.create_move_objects(idx)

    def _draw_labels(self):
        if self.stgs.get_setting(ct.K_SHOW_LABELS[1]):
            for i in range(self.size):
                row_label = "%d" % (self.size - i)
                # left row label
                self.graph.DrawText(row_label, ((self.offset_factor - 1) * self.cell_width,
                                                (i + self.offset_factor) * self.cell_width),
                                    ct.BOARD_LABEL_COLOR, ct.BOARD_LABEL_FONT)
                # right row label
                self.graph.DrawText(row_label, ((self.size + self.offset_factor) * self.cell_width,
                                                (i + self.offset_factor) * self.cell_width),
                                    ct.BOARD_LABEL_COLOR, ct.BOARD_LABEL_FONT)

            for i in range(self.size):
                col_label = chr(ord('A') + i)
                # top column label
                self.graph.DrawText(col_label,
                                    ((i + self.offset_factor) * self.cell_width,
                                     (self.offset_factor - 1) * self.cell_width),
                                    ct.BOARD_LABEL_COLOR, ct.BOARD_LABEL_FONT)
                # bottom column label
                self.graph.DrawText(col_label,
                                    ((i + self.offset_factor) * self.cell_width,
                                     (self.size + self.offset_factor) * self.cell_width),
                                    ct.BOARD_LABEL_COLOR, ct.BOARD_LABEL_FONT)

    def _score_to_rgbstring(self, sc):
        if sc > 0:
            return '#00FF' + f'{int(255 * (1 - sc)):02x}'
        return '#00' + f'{int(255 * (sc + 1)):02x}' + 'FF'

    def _draw_pegholes(self, heatmap=None):
        for x in range(self.size):
            for y in range(self.size):
                if x in (0, self.size - 1) and y in (0, self.size - 1):
                    continue

                color = border = ct.PEG_HOLE_COLOR
                radius = self.hole_radius
                if heatmap:
                    sc = heatmap.scores[x, twixt.Game.SIZE - y - 1]
                    if not numpy.isnan(sc):
                        color = self._score_to_rgbstring(sc)
                        radius = self.peg_radius * (1 + ct.HEATMAP_RADIUS_FACTOR * abs(sc)) / 2
                        
                        move = chr(ord('a') + x) + str(twixt.Game.SIZE - y)
                        if move in heatmap.policy_moves:
                            print(f'{move} in {heatmap.policy_moves}')
                            border = 'black'
                        else:
                            border = color

                self.graph.DrawCircle(
                    ((x + self.offset_factor) * self.cell_width,
                     (y + self.offset_factor) * self.cell_width),
                    radius, color, border)

    def _draw_endlines(self):
        o = 2 * self.cell_width
        s = self.size - 1
        w = self.cell_width
        self.graph.DrawLine((o + 1 * w, o + 1 * w + w / 3),
                            (o + 1 * w, o + s * w - w / 3), self.stgs.get_setting(ct.K_COLOR[2]), 3)
        self.graph.DrawLine((o + s * w, o + 1 * w + w / 3),
                            (o + s * w, o + s * w - w / 3), self.stgs.get_setting(ct.K_COLOR[2]), 3)
        self.graph.DrawLine((o + 1 * w + w / 3, o + 1 * w),
                            (o + s * w - w / 3, o + 1 * w), self.stgs.get_setting(ct.K_COLOR[1]), 3)
        self.graph.DrawLine((o + 1 * w + w / 3, o + s * w),
                            (o + s * w - w / 3, o + s * w), self.stgs.get_setting(ct.K_COLOR[1]), 3)

    def _draw_guidelines(self):
        if self.stgs.get_setting(ct.K_SHOW_GUIDELINES[1]):
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
                                    ct.GUIDELINE_COLOR, .5)

    def get_move(self, coords):

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
        if len(self.game.history) == 1 and self._move_to_point(
                move) == self.game.history[0] and self.stgs.get_setting(ct.K_ALLOW_SWAP[1]):
            return twixt.SWAP

        if x < 0 or x > self.size - 1 or y < 0 or y > self.size - 1:
            # overboard click
            return None
        elif (x == 0 or x == self.size - 1) and (y == 0 or y == self.size - 1):
            # corner click
            return None
        elif ((x == 0 or x == self.size - 1) and len(self.game.history) % 2 == 0):
            # white clicked on black's end line
            return None
        elif ((y == 0 or y == self.size - 1) and len(self.game.history) % 2 == 1):
            # black clicked white's end line
            return None

        if not self.valid_spot(move):
            return None

        return move

    def valid_spot(self, move):

        p = self._move_to_point(move)
        if len(self.game.history) >= 2 and self.game.history[1] == twixt.SWAP:
            # game is swapped
            if p in self.game.history[2:]:
                # spot occupied after swap
                return False
            elif twixt.Point(p.y, p.x) == self.game.history[0]:
                # spot occupied on swap (flip)
                return False
            else:
                return True
        else:
            # game is not swapped
            if p in self.game.history:
                # spot occupied
                return False

        return True

    def create_move_objects(self, index):
        return super().create_move_objects(self.game, index)
