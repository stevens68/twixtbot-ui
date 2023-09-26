#! /usr/bin/env python
import numpy
import logging
from collections import namedtuple
import constants as ct
from backend.point import Point
from backend.select_set import SelectSet
from backend.board import TwixtBoard

SWAP = "swap"
RESIGN = "resign"
DRAW = "draw"
MAXBEST = 24**24

LinkDescription = namedtuple('LinkDescription', 'p1 p2 owner')

class Game:
    SIZE = 24
    LINK_LONGY = 4
    LINK_DIFFSIGN = 2
    BLACK = 0
    WHITE = 1
    DLINKS            = [(-2, -1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1)]
    DLINKS_RIGHTFIRST = [(2, -1), (2, 1), (1, -2), (1, 2), (-1, -2), (-1, 2), (-2, -1), (-2, 1)]
    DLINKS_DOWNFIRST  = [(-1, 2), (1, 2), (-2, 1), (2, 1), (-2, -1), (2, -1), (-1, -2), (1, -2)]

    COLOR_NAME = ("BLACK", "WHITE")

    def __init__(self, allow_scl):
        self.logger = logging.getLogger(ct.LOGGER)
        self.allow_scl = allow_scl

        self.result = None
        self.history = []
        self.pegs = [numpy.zeros((Game.SIZE, Game.SIZE), numpy.int8)
                     for _ in range(2)]
        self.links = [numpy.zeros((Game.SIZE, Game.SIZE), numpy.int8)
                      for _ in range(8)]
        self.turn = Game.WHITE
        self.open_pegs = [SelectSet(), SelectSet()]
        self.reachable = [set(), set()]
        self.reachable_history = []

        for x in range(Game.SIZE):
            for y in range(Game.SIZE):
                p = Point(x, y)
                if x not in (0, Game.SIZE - 1):
                    self.open_pegs[Game.WHITE].add(p)
                if y not in (0, Game.SIZE - 1):
                    self.open_pegs[Game.BLACK].add(p)
        
        # end __init__
    

    def _flip_turn(self):
        self.turn = 1 - self.turn

    def clone(self):

        copy = Game(self.allow_scl)
        copy.history = list(self.history)
        copy.pegs = numpy.array(self.pegs)
        copy.links = numpy.array(self.links)
        copy.turn = self.turn
        copy.open_pegs = [x.clone() for x in self.open_pegs]
        copy.reachable = [set(x) for x in self.reachable]
        copy.reachable_history = list(self.reachable_history)
        return copy

    def turn_to_player(self, turn=None):
        if turn is not None or turn == 0:
            return 2 - turn
        else:
            return 2 - self.turn

    def just_won(self):

        return self.is_winning(1 - self.turn)

    def _is_a_draw(self):
        return not (self._has_end_to_end_path(0) or self._has_end_to_end_path(1))

    def is_winning(self, color):

        return "win" in self.reachable[color]

    def play_swap(self):

        assert len(self.history) == 1
        a = self.history[0]
        b = Point(a.y, a.x)
        self.pegs[Game.WHITE][a] = 0
        self.pegs[Game.BLACK][b] = 1
        self.history.append(SWAP)

        self.turn = Game.WHITE

        self.reachable[Game.BLACK] = set()
        for p in self.reachable[Game.WHITE]:
            q = Point(p.y, p.x)
            self.reachable[Game.BLACK].add(q)
        self.reachable[Game.WHITE] = set()
        self.reachable_history.append([])

        """
        self.open_pegs[0].add(a)
        self.open_pegs[1].add(a)
        self.open_pegs[0].remove(b)
        self.open_pegs[1].remove(b)
        """

    def undo_swap(self):

        assert len(self.history) == 2
        a = self.history[0]
        b = Point(a.y, a.x)
        self.pegs[Game.WHITE][a] = 1
        self.pegs[Game.BLACK][b] = 0
        self.history.pop()

        self.turn = Game.BLACK

        self.reachable[Game.WHITE] = set()
        for p in self.reachable[Game.BLACK]:
            q = Point(p.y, p.x)
            self.reachable[Game.WHITE].add(q)
        self.reachable[Game.BLACK] = set()
        self.reachable_history.pop()

    def play(self, move, check_draw=False):

        if move == SWAP:
            self.play_swap()
            return

        if type(move) == str:
            move = Point(move)

        assert Game.inbounds(move), (move)
        assert self.pegs[0][move] == 0, (0, self.pegs[0], move, self.history)
        assert self.pegs[1][move] == 0, (1, self.pegs[1], move, self.history)

        if self.turn == Game.WHITE:
            assert move.x != 0 and move.x != Game.SIZE - 1
        else:
            assert move.y != 0 and move.y != Game.SIZE - 1

        for dlink in Game.DLINKS:
            pt = move + dlink
            if (not Game.inbounds(pt)) or self.pegs[self.turn][pt] == 0:
                continue
            if self.any_crossing_links(move, pt, 1 - self.turn):
                continue
            if (not self.allow_scl and
                    self.any_crossing_links(move, pt, self.turn)):
                continue

            self.set_link(move, pt, self.turn, 1)

        self.pegs[self.turn][move] = 1

        self.history.append(move)
        self.reachable_history.append(self._update_add_reachable(move))
        
        self._flip_turn()

        self.open_pegs[0].remove(move)
        self.open_pegs[1].remove(move)

        if self._is_a_draw():
            self.result = DRAW

        # end play(self)

    def _update_add_reachable(self, move):

        added = []
        color = self.turn
        my_reachable = self.reachable[color]

        if move.x == self.SIZE - 1 or move.y == self.SIZE - 1:
            # reachable because at the far boundary
            added.append(move)
        else:
            # reachable because links to another reachable
            for dlink in Game.DLINKS:
                other = move + dlink
                if other in my_reachable and self.get_link(move, other, color):
                    added.append(move)
                    if move.x == 0 or move.y == 0:
                        added.append("win")
                        my_reachable.add(move)
                        my_reachable.add("win")
                        return added
                    break

        if not added:
            return added

        # fun may commence!
        my_reachable.add(move)
        unvisited = [move]

        while unvisited:
            chk = unvisited.pop()
            assert chk in my_reachable
            for dlink in Game.DLINKS:
                other = chk + dlink
                if (self.inbounds(other) and
                        other not in my_reachable and
                        self.get_peg(other, color) and
                        self.get_link(chk, other, color)):
                    added.append(other)
                    unvisited.append(other)
                    my_reachable.add(other)
                    if other.x == 0 or other.y == 0:
                        added.append("win")
                        my_reachable.add("win")
                        return added

        return added

    def get_peg(self, point, color):

        return self.pegs[color][point]

    def safe_get_peg(self, point, color):

        if not Game.inbounds(point):
            return 0
        else:
            return self.pegs[color][point]

    def safe_get_link(self, a, b, color):

        if self.inbounds(a) and self.inbounds(b):
            return self.get_link(a, b, color)
        else:
            return 0

    def get_link(self, a, b, color):

        ix1, ix2 = self.get_link_index(a, b, color)
        self.logger.debug("x1,x2: %s, %s", ix1, ix2)
        return self.links[ix1][ix2]

    def set_link(self, a, b, color, value):

        ix1, ix2 = self.get_link_index(a, b, color)
        self.links[ix1][ix2] = value

    def get_link_index(self, a, b, color):

        ix1 = color

        if (a.x + b.x) % 2 != 0:
            ix1 += Game.LINK_LONGY

        cx = int((a.x + b.x) // 2)
        cy = int((a.y + b.y) // 2)

        if (b.y - a.y) * (b.x - a.x) < 0:
            ix1 += Game.LINK_DIFFSIGN

        return ix1, (cx, cy)

    @staticmethod
    def describe_link(index, x, y):

        color = index & 1
        if index & Game.LINK_LONGY:
            xlo = x
            xhi = x + 1
            ylo = y - 1
            yhi = y + 1
        else:
            xlo = x - 1
            xhi = x + 1
            ylo = y
            yhi = y + 1

        if index & Game.LINK_DIFFSIGN:
            p1 = Point(xlo, yhi)
            p2 = Point(xhi, ylo)
        else:
            p1 = Point(xlo, ylo)
            p2 = Point(xhi, yhi)

        return LinkDescription(p1, p2, color)

    @staticmethod
    def link_to_points(link):

        if isinstance(link, LinkDescription):
            return link.p1, link.p2
        else:
            return link[0], link[1]

    @staticmethod
    def do_links_cross(linka, linkb):

        a0, a1 = Game.link_to_points(linka)
        b0, b1 = Game.link_to_points(linkb)

        if a0.x > a1.x:
            a0, a1 = a1, a0
        if b0.x > b1.x:
            b0, b1 = b1, b0

        slope_a = (a1.y - a0.y) / float(a1.x - a0.x)
        slope_b = (b1.y - b0.y) / float(b1.x - b0.x)

        # I don't care about exactly equal links.
        if abs(slope_a) == abs(slope_b):
            return False

        int_a = a0.y - slope_a * a0.x
        int_b = b0.y - slope_b * b0.x
        xmeet = (int_b - int_a) / (slope_a - slope_b)
        return a0.x < xmeet and a1.x > xmeet and b0.x < xmeet and b1.x > xmeet

    def any_crossing_links(self, a, b, color, value=None):

        # reverse parity crosses three times.
        debug = False

        delta = b - a
        dshort = Point((delta.x & 1) * delta.x, (delta.y & 1) * delta.y)
        dlong = Point((delta.x - dshort.x) / 2, (delta.y - dshort.y) / 2)
        if debug:
            self.logger.debug(
                "any_crossing_links. a=%s, b=%s, color=%s", a, b, color)
            self.logger.debug("delta=(%d,%d)", delta.x, delta.y)
            self.logger.debug("dlong=(%d,%d)", dlong.x, dlong.y)
            self.logger.debug("dshort=(%d,%d)", dshort.x, dshort.y)

        cross_links = [
            (-1, 1, 1, 0),
            (0, 1, 2, 0),
            (1, 1, 3, 0),

            (0, 1, 1, -1),
            (0, 2, 1, 0),
            (1, 1, 2, -1),
            (1, 2, 2, 0),

            (0, -1, 1, 1),
            (1, 0, 2, 2)
        ]

        found = False
        for cl in cross_links:
            c = a + dlong * cl[0] + dshort * cl[1]
            d = a + dlong * cl[2] + dshort * cl[3]
            if debug:
                self.logger.debug("checking %d,%d", c, d)
            if (self.inbounds(c) and self.inbounds(d) and
                    self.get_link(c, d, color)):
                found = True
                self.set_link(c, d, color, 1)
        return found

    def undo(self, check_draw=False):

        assert len(self.history) > 0
        uturn = 1 - self.turn
        umove = self.history[-1]
        if umove == SWAP:
            self.undo_swap()
            return

        assert self.pegs[uturn][umove] == 1
        self.pegs[uturn][umove] = 0

        for dlink in Game.DLINKS:
            lend = umove + dlink
            if Game.inbounds(lend):
                self.set_link(umove, lend, uturn, 0)

        self.history.pop()
        self.turn = uturn
        rh = self.reachable_history.pop()
        for p in rh:
            assert p in self.reachable[uturn], (p)
            self.reachable[uturn].remove(p)

        if umove.x not in (0, Game.SIZE - 1):
            self.open_pegs[Game.WHITE].add(umove)
        if umove.y not in (0, Game.SIZE - 1):
            self.open_pegs[Game.BLACK].add(umove)

        # end undo
             
    @staticmethod
    def inbounds(p):
        """ Tell us whether a given point is inside the numpy arrays;
            may still not be a valid place to play. """
        return p.x >= 0 and p.x < Game.SIZE and p.y >= 0 and p.y < Game.SIZE

    @staticmethod
    def inbounds_for_player(p, color):

        if color == Game.WHITE:
            return (p.x >= 1 and p.x < Game.SIZE - 1 and
                    p.y >= 0 and p.y < Game.SIZE)
        elif color == Game.BLACK:
            return (p.x >= 0 and p.x < Game.SIZE and
                    p.y >= 1 and p.y < Game.SIZE - 1)
        else:
            return False

    def __str__(self):

        topline = "   " + "   ".join([chr(ord('A') + i)
                                      for i in range(Game.SIZE)]) + "\n"
        out = topline
        for y in range(Game.SIZE):
            if y > 0:
                out += self._str_tween_row(y - 1)
            out += self._str_peg_row(y)
        out += topline
        return out

    def _str_peg_row(self, y):

        if y in (0, Game.SIZE - 1):
            zero_char = " "
            xb_char = " "
        else:
            zero_char = "."
            xb_char = "|"

        out = "%2d " % (y + 1)

        for x in range(Game.SIZE):
            if x > 0:
                sw = Point(x - 1, y + 1)
                nw = Point(x - 1, y - 1)
                se = Point(x, y + 1)
                ne = Point(x, y - 1)
                ne_link = self.get_link(sw, ne, 0) + self.get_link(sw, ne, 1)
                nw_link = self.get_link(se, nw, 0) + self.get_link(se, nw, 1)

                if ne_link and nw_link:
                    lc = "X"
                elif ne_link:
                    lc = "/"
                elif nw_link:
                    lc = "\\"
                elif x in (1, Game.SIZE - 1):
                    lc = xb_char
                else:
                    lc = " "

                out += " " + lc + " "

            if self.pegs[0][x, y]:
                out += "#"
            elif self.pegs[1][x, y]:
                out += "O"
            elif x in (0, Game.SIZE - 1):
                out += zero_char
            else:
                out += "."

        return out + " %-2d\n" % (y)

    def _str_tween_row(self, y):

        if y in (0, Game.SIZE - 2):
            nadachar = "-"
            xls = " +-"
            xrs = "-+ "
        else:
            nadachar = " "
            xls = " | "
            xrs = " | "

        out = "   "
        for x in range(Game.SIZE):
            if x == 1:
                out += xls
            elif x == Game.SIZE - 1:
                out += xrs
            elif x > 0:
                out += nadachar * 3

            nw = Point(x - 1, y)
            ne = Point(x + 1, y)
            sw = Point(x - 1, y + 1)
            se = Point(x + 1, y + 1)

            ne_link = self.get_link(sw, ne, 0) + self.get_link(sw, ne, 1)
            nw_link = self.get_link(se, nw, 0) + self.get_link(se, nw, 1)

            if ne_link and nw_link:
                lc = "X"
            elif ne_link:
                lc = "/"
            elif nw_link:
                lc = "\\"
            elif x in (1, Game.SIZE - 1):
                lc = nadachar
            else:
                lc = " "
            out += lc

        return out + "\n"

    def _has_end_to_end_path(self, turn):
        check_board = Game(True)
        visited = []
        link_range = Game.DLINKS_DOWNFIRST if turn == Game.WHITE else Game.DLINKS_RIGHTFIRST 
        for i in range(1, Game.SIZE-1):
            # (i,0) for WHITE. (0,i) for BLACK
            pStart = Point(i * turn, i * (1 - turn))
            check_board.pegs[turn][pStart] = 1
            if self._find_path_to_end(check_board, turn, pStart, visited, link_range):
                return True
        return False

    def _find_path_to_end(self, check_board, turn, p, visited, link_range):
        visited.append(p)
        for dlink in link_range:
            pt = p + dlink
            if self.inbounds_for_player(pt, turn) and pt not in visited and not self.pegs[1-turn][pt]:
                # pt is a valid move (empty or own color)
                if not self.any_crossing_links(p, pt, 1-turn) and (self.allow_scl or not self.any_crossing_links(p, pt, turn)):
                    # there are no links crossing link (p, pt)
                    check_board.pegs[turn][pt] = 1
                    check_board.set_link(p, pt, turn, 1)
                    # if pt is on opposite endline - we found a path
                    if (turn == Game.WHITE and pt.y == Game.SIZE-1) or (turn == Game.BLACK and pt.x == Game.SIZE-1):
                        return True
                    elif self._find_path_to_end(check_board, turn, pt, visited, link_range):
                        return True
        return False

