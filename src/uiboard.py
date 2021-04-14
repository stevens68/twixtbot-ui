import itertools
import PySimpleGUI as sg
import backend.board as board
import backend.twixt as twixt
import constants as ct


class UiBoard(board.TwixtBoard):

    def __init__(self, game, stgs):
        super().__init__(stgs)

        self.game = game
        self.current_cursor_label = None
        self.rect_item = None

        bp = stgs.get(ct.K_BOARD_SIZE[1])
        self.cursor_label_factor = bp / ct.K_BOARD_SIZE[3]
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
        

    def draw(self, heatmap=None, complete=True):
        if complete or heatmap is not None:
            self.graph.erase()
            self._draw_endlines()
            self._draw_labels()
            self._draw_guidelines()
            self._draw_pegholes()
            if self.game.history is not None:
                for idx in range(len(self.game.history)):
                    self.create_move_objects(idx)
        else:
            gl = len(self.game.history)
                # erase move before last move
            if gl > 1:
                self.undo_last_move_objects()
                self.create_move_objects(gl - 2)
            self.create_move_objects(gl-1)

        if heatmap is not None:
            self._draw_heatmap_legend(heatmap)
            self._draw_heatmap(heatmap)

    def draw_cursor_label(self, move):
        if move is None and self.current_cursor_label is not None:
            # remove current
            self.graph.delete_figure(self.current_cursor_label)
            self.current_cursor_label = None
            self.graph.delete_figure(self.rect_item)
        elif move is not None and self.current_cursor_label is None:
            (x, y) = self._point_to_coords(self._move_to_point(move))
            coords = (x - 2 * self.cursor_label_factor,
                      y + 15 * self.cursor_label_factor)
            self.current_cursor_label = self.graph.DrawText(move.upper(), coords,
                                                            ct.BOARD_LABEL_COLOR, ct.BOARD_LABEL_FONT)
            tl, br = self.graph.GetBoundingBox(self.current_cursor_label)
            self.rect_item = self.graph.DrawRectangle(
                tl, br, line_color=ct.CURSOR_LABEL_BACKGROUND_COLOR, fill_color=ct.CURSOR_LABEL_BACKGROUND_COLOR, line_width=3)
            self.graph.BringFigureToFront(self.current_cursor_label)

    def _draw_labels(self):
        if self.stgs.get(ct.K_SHOW_LABELS[1]):
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

    def _draw_heatmap_legend(self, heatmap):
        # Draw the label for the heatmap
        self.graph.DrawText(
            ct.K_HEATMAP[0],
            (self.cell_width * self.offset_factor,
             (self.offset_factor - 2) * self.cell_width),
            ct.BOARD_LABEL_COLOR, ct.BOARD_LABEL_FONT)

        # Draw the heatmap
        for i, rgb_col in enumerate(heatmap.heatmap_legend()):
            self.graph.DrawRectangle(
                (self.cell_width * (self.offset_factor + i + 2), 5),
                (self.cell_width * (i + 3 + self.offset_factor), 15),
                rgb_col, rgb_col)

    def _draw_heatmap(self, heatmap=None):
        if not heatmap:
            return

        for move, rgb_color in heatmap.rgb_colors.items():
            # Draw a circle around those moves with a p value
            self.graph.DrawCircle(
                ((move.x + self.offset_factor) * self.cell_width,
                 (twixt.Game.SIZE - move.y - 1 + self.offset_factor) * self.cell_width),
                self.hole_radius *
                (1 + ct.HEATMAP_RADIUS_FACTOR * ct.HEATMAP_CIRCLE_FACTOR),
                None, ct.HEATMAP_CIRCLE_COLOR)

            radius = self.hole_radius
            radius *= (1 + ct.HEATMAP_RADIUS_FACTOR *
                       heatmap.p_values[move] ** 0.5)

            # Draw colored circle
            self.graph.DrawCircle(
                ((move.x + self.offset_factor) * self.cell_width,
                 (twixt.Game.SIZE - move.y - 1 + self.offset_factor) * self.cell_width),
                radius, rgb_color, rgb_color)

    def _draw_pegholes(self):
        for x, y in itertools.product(range(self.size), range(self.size)):
            # skip the 4 corners
            if x in (0, self.size - 1) and y in (0, self.size - 1):
                continue

            self.graph.DrawCircle(
                ((x + self.offset_factor) * self.cell_width,
                 (y + self.offset_factor) * self.cell_width),
                self.hole_radius, ct.PEG_HOLE_COLOR, ct.PEG_HOLE_COLOR)

    def _draw_endlines(self):
        o = 2 * self.cell_width
        s = self.size - 1
        w = self.cell_width
        self.graph.DrawLine((o + 1 * w, o + 1 * w + w / 3),
                            (o + 1 * w, o + s * w - w / 3), self.stgs.get(ct.K_COLOR[2]), 3)
        self.graph.DrawLine((o + s * w, o + 1 * w + w / 3),
                            (o + s * w, o + s * w - w / 3), self.stgs.get(ct.K_COLOR[2]), 3)
        self.graph.DrawLine((o + 1 * w + w / 3, o + 1 * w),
                            (o + s * w - w / 3, o + 1 * w), self.stgs.get(ct.K_COLOR[1]), 3)
        self.graph.DrawLine((o + 1 * w + w / 3, o + s * w),
                            (o + s * w - w / 3, o + s * w), self.stgs.get(ct.K_COLOR[1]), 3)

    def _draw_guidelines(self):
        if self.stgs.get(ct.K_SHOW_GUIDELINES[1]):
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

        # returns (legalmove, pegposition)
        # pegposition:
        #    has the move coordinates of the peghole the mouse points to,
        #    None if the mouse doesn't point to a valid peghole
        # legalmove:
        #    can be none if mouse doesn't point to a legal move
        #    can be swap if moise points to move #1
        #    equals pegposition otherwise

        offset = self.offset_factor * self.cell_width
        x = (coords[0] - offset) / self.cell_width
        y = (coords[1] - offset) / self.cell_width

        # click distance tolerance
        maxgap = 0.3
        xgap = abs(x - round(x))
        ygap = abs(y - round(y))
        if xgap > maxgap or ygap > maxgap:
            # click not on hole
            return None, None

        x = round(x)
        y = round(y)
        move = chr(ord('a') + x) + "%d" % (self.size - y)
        if len(self.game.history) == 1 and self._move_to_point(
                move) == self.game.history[0] and self.stgs.get(ct.K_ALLOW_SWAP[1]):
            return twixt.SWAP, move

        if x < 0 or x > self.size - 1 or y < 0 or y > self.size - 1:
            # overboard click
            return None, None
        elif (x == 0 or x == self.size - 1) and (y == 0 or y == self.size - 1):
            # corner click
            return None, None
        elif ((x == 0 or x == self.size - 1) and len(self.game.history) % 2 == 0):
            # white clicked on black's end line
            return None, move
        elif ((y == 0 or y == self.size - 1) and len(self.game.history) % 2 == 1):
            # black clicked white's end line
            return None, move

        if not self.valid_spot(move):
            return None, move

        return move, move

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

    def create_move_objects(self, index, visits=None):
        return super().create_move_objects(self.game, index, visits)
