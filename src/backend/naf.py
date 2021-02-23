#! /usr/bin/env python
import math
import numpy
import sys

import backend.twixt as twixt


class NetInputs:
    HEADER = 'JTwx'
    HEADER_BYTES = len(HEADER)
    NUM_RECENTS = 4
    FRONT_BYTES = HEADER_BYTES + 2 * NUM_RECENTS
    EXPANDED_SIZE = FRONT_BYTES + 10 * twixt.Game.SIZE ** 2
    COMPACT_SIZE = FRONT_BYTES + 10 * twixt.Game.SIZE ** 2 / 8
    NAF_DIMS = (twixt.Game.SIZE, twixt.Game.SIZE, 11)

    def __init__(self, thing):

        self.naf = numpy.zeros(self.NAF_DIMS, dtype=numpy.uint8)
        if thing is None:
            return
        elif isinstance(thing, twixt.Game):
            self.init_from_game(thing)
        elif isinstance(thing, str):
            if len(thing) == self.EXPANDED_SIZE:
                self.init_from_expanded(thing)
            elif len(thing) == self.COMPACT_SIZE:
                self.init_from_compact(thing)
            else:
                raise TypeError("Wrong string length")
        else:
            raise TypeError("Only twixt.Game or binary formats usable")

    def init_from_game(self, game):

        if game.turn == game.WHITE:
            self.init_from_game_white(game)
        else:
            self.init_from_game_black(game)

    def init_from_game_white(self, game):

        for i in range(8):
            self.naf[:, :, i] = game.links[i]
        for j in range(2):
            self.naf[:, :, 8 + j] = game.pegs[j]

        self._init_recents_from_game(game, lambda x: x)

    def init_from_game_black(self, game):

        for color in range(2):
            acolor = 1 - color
            # LONGY changes but DIFFISGN and color do not.
            self.naf[:, :, 4 + color] = game.links[0 + acolor].T
            self.naf[:, :, 6 + color] = game.links[2 + acolor].T
            self.naf[:, :, 0 + color] = game.links[4 + acolor].T
            self.naf[:, :, 2 + color] = game.links[6 + acolor].T
            self.naf[:, :, 8 + color] = game.pegs[acolor].T

        self._init_recents_from_game(game, lambda p: p.flip())

    def _init_recents_from_game(self, game, flipper):

        rev_recents = []
        i = 1
        swap_mode = False
        while len(rev_recents) < self.NUM_RECENTS and i < len(game.history):
            h = game.history[-i]
            if h == 'swap':
                assert not swap_mode
                swap_mode = True
                i += 1
                continue
            if swap_mode:
                h = h.flip()
            rev_recents.append(flipper(h))
            i += 1

        self.recents = list(reversed(rev_recents))
        self._load_recents()

    def _load_recents(self):

        for p in self.recents:
            self.naf[p.x, p.y, 10] = 1

    def hflip(self):

        tmp = numpy.flip(self.naf, 0)

        S = twixt.Game.SIZE
        vix = twixt.Game.LINK_LONGY
        self.naf = numpy.zeros((S, S, 11), dtype=numpy.uint8)
        self.naf[:, :, 10] = tmp[:, :, 10]
        for color in range(2):
            self.naf[:, :, 8 + color] = tmp[:, :, 8 + color]

            for diffsign in range(2):
                dix = diffsign * twixt.Game.LINK_DIFFSIGN
                adix = (1 - diffsign) * twixt.Game.LINK_DIFFSIGN
                # non verticals are easy
                self.naf[:, :, color + dix] = tmp[:, :, color + adix]
                # verticals need to be shifted.
                self.naf[:-1, :, color + dix +
                         vix] = tmp[1:, :, color + adix + vix]

        self.recents = [twixt.Point(twixt.Game.SIZE - 1 - p.x, p.y)
                        for p in self.recents]

    def vflip(self):

        tmp = numpy.flip(self.naf, 1)

        S = twixt.Game.SIZE
        vix = twixt.Game.LINK_LONGY
        self.naf = numpy.zeros((S, S, 11), dtype=numpy.uint8)
        self.naf[:, :, 10] = tmp[:, :, 10]

        for color in range(2):
            self.naf[:, :, 8 + color] = tmp[:, :, 8 + color]

            for diffsign in range(2):
                dix = diffsign * twixt.Game.LINK_DIFFSIGN
                adix = (1 - diffsign) * twixt.Game.LINK_DIFFSIGN
                # verticals are easy
                self.naf[:, :, color + dix +
                         vix] = tmp[:, :, color + adix + vix]
                # horizontals need to be shifted.
                self.naf[:, :-1, color + dix] = tmp[:, 1:, color + adix]

        self.recents = [twixt.Point(p.x, twixt.Game.SIZE - 1 - p.y)
                        for p in self.recents]

    def rotate(self, r):

        assert r >= 0 and r < NUM_ROTATIONS
        if r & HFLIP_BIT:
            self.hflip()
        if r & VFLIP_BIT:
            self.vflip()

    def to_compact_bytes(self):

        return self._front_bytes() + numpy.packbits(self.naf[:, :, :10]).tobytes()

    def to_expanded_bytes(self):

        return self._front_bytes() + self.naf[:, :, :10].tobytes()

    def _front_bytes(self):

        b = self.HEADER
        for i in range(self.NUM_RECENTS):
            if i < len(self.recents):
                p = self.recents[i]
                b += chr(p.x)
                b += chr(p.y)
            else:
                b += '\xff\xff'
        assert len(b) == self.FRONT_BYTES
        return b

    def _init_front_bytes(self, b):

        if b[:self.HEADER_BYTES] != self.HEADER:
            raise ValueError("Header sanity error (%s)" %
                             (b[:self.HEADER_BYTES]))
        self.recents = []
        for i in range(self.NUM_RECENTS):
            x = ord(b[self.HEADER_BYTES + i * 2])
            y = ord(b[self.HEADER_BYTES + i * 2 + 1])
            if x == 255 and y == 255:
                continue
            if x >= twixt.Game.SIZE or y >= twixt.Game.SIZE:
                raise ValueError("Recent Point Error")
            self.recents.append(twixt.Point(x, y))
        self._load_recents()

    def init_from_compact(self, b):

        assert len(b) == self.COMPACT_SIZE
        self._init_front_bytes(b)
        p0 = numpy.frombuffer(b[self.FRONT_BYTES:], dtype=numpy.uint8)
        p1 = numpy.unpackbits(p0)
        p2 = p1.reshape((twixt.Game.SIZE, twixt.Game.SIZE, 10))
        assert abs(numpy.count_nonzero(
            p2[:, :, 8]) - numpy.count_nonzero(p2[:, :, 9])) <= 1
        self.naf[:, :, :10] = p2

    def init_from_expanded(self, b):

        assert len(b) == self.EXPANDED_SIZE
        self._init_front_bytes(b)
        p0 = numpy.frombuffer(b[self.FRONT_BYTES:], dtype=numpy.uint8)
        p1 = p0.reshape(twixt.Game.SIZE, twixt.Game.SIZE, 10)
        c8 = numpy.count_nonzero(p1[:, :, 8])
        c9 = numpy.count_nonzero(p1[:, :, 9])
        if abs(c8 - c9) > 1:
            print("Zut!! c8=%d, c9=%d" % (c8, c9), file=sys.stderr)
            print("p8:", file=sys.stderr)
            print(binary_array_string(p1[:, :, 8]), file=sys.stderr)
            print("p9:", file=sys.stderr)
            print(binary_array_string(p1[:, :, 9]), file=sys.stderr)

        assert abs(c8 - c9) <= 1
        self.naf[:, :, :10] = p1

    def equal_with(self, other):

        if not numpy.array_equal(self.naf, other.naf):
            return False
        if self.recents != other.recents:
            return False
        return True

    def to_input_arrays(self, use_recents=False):

        pegs = self.naf[:, :, 8:10]
        links = numpy.copy(self.naf[:, :, :8])
        for index in range(8):
            vertical = index & twixt.Game.LINK_LONGY
            diffsign = index & twixt.Game.LINK_DIFFSIGN

            if vertical or diffsign:
                links[1:, :, index] = links[:-1, :, index]
                links[0, :, index] = 0
            if (not vertical) or diffsign:
                links[:, 1:, index] = links[:, :-1, index]
                links[:, 0, index] = 0

        if use_recents:
            locs = numpy.zeros(
                (twixt.Game.SIZE, twixt.Game.SIZE, 3), dtype=numpy.float32)
            location_inputs(locs)
            locs[:, :, 2] = self.naf[:, :, 10]
            return pegs, links, locs
        else:
            return pegs, links, location_inputs()

# naf is the Numpy Array Format.  It is always "swapped" so that "white"
#  is on play.  The shape is (S,S,10) where S = twixt.Game.SIZE.  The 10
# planes are defined as:
# 0-7  links
# 8-9  pegs


def old_game_to_naf(game):
    S = twixt.Game.SIZE
    naf = numpy.zeros((S, S, 10), dtype=numpy.uint8)

    if game.turn == game.WHITE:
        return _game_to_naf_white(game, naf)
    else:
        return _game_to_naf_black(game, naf)


def _game_to_naf_white(game, naf):
    for i in range(8):
        naf[:, :, i] = game.links[i]
    for j in range(2):
        naf[:, :, 8 + j] = game.pegs[j]
    return naf


def _game_to_naf_black(game, naf):
    for color in range(2):
        acolor = 1 - color
        # LONGY changes but DIFFISGN and color do not.
        naf[:, :, 4 + color] = game.links[0 + acolor].T
        naf[:, :, 6 + color] = game.links[2 + acolor].T
        naf[:, :, 0 + color] = game.links[4 + acolor].T
        naf[:, :, 2 + color] = game.links[6 + acolor].T
        naf[:, :, 8 + color] = game.pegs[acolor].T
    return naf


def old_hflip_naf(naf):
    tmp = numpy.flip(naf, 0)

    S = twixt.Game.SIZE
    vix = twixt.Game.LINK_LONGY
    out = numpy.zeros((S, S, 10), dtype=numpy.uint8)
    for color in range(2):
        out[:, :, 8 + color] = tmp[:, :, 8 + color]

        for diffsign in range(2):
            dix = diffsign * twixt.Game.LINK_DIFFSIGN
            adix = (1 - diffsign) * twixt.Game.LINK_DIFFSIGN
            # non verticals are easy
            out[:, :, color + dix] = tmp[:, :, color + adix]
            # verticals need to be shifted.
            out[:-1, :, color + dix + vix] = tmp[1:, :, color + adix + vix]
    return out


def old_vflip_naf(naf):
    tmp = numpy.flip(naf, 1)

    S = twixt.Game.SIZE
    vix = twixt.Game.LINK_LONGY
    out = numpy.zeros((S, S, 10), dtype=numpy.uint8)

    for color in range(2):
        out[:, :, 8 + color] = tmp[:, :, 8 + color]

        for diffsign in range(2):
            dix = diffsign * twixt.Game.LINK_DIFFSIGN
            adix = (1 - diffsign) * twixt.Game.LINK_DIFFSIGN
            # verticals are easy
            out[:, :, color + dix + vix] = tmp[:, :, color + adix + vix]
            # horizontals need to be shifted.
            out[:, :-1, color + dix] = tmp[:, 1:, color + adix]
    return out


def old_naf_expanded_size():
    return 10 * twixt.Game.SIZE ** 2


def old_naf_compressed_size():
    return 10 * twixt.Game.SIZE ** 2 / 8


def old_naf_to_compressed(naf):
    return numpy.packbits(naf).tobytes()


def old_naf_to_expanded(naf):
    return naf.tobytes()


def old_naf_from_compressed(b):
    assert len(b) == old_naf_compressed_size()
    p0 = numpy.frombuffer(b, dtype=numpy.uint8)
    p1 = numpy.unpackbits(p0)
    p2 = p1.reshape((twixt.Game.SIZE, twixt.Game.SIZE, 10))
    assert abs(numpy.count_nonzero(
        p2[:, :, 8]) - numpy.count_nonzero(p2[:, :, 9])) <= 1
    return p2


def old_naf_from_expanded(b):
    assert len(b) == old_naf_expanded_size()
    p0 = numpy.frombuffer(b, dtype=numpy.uint8)
    p1 = p0.reshape(twixt.Game.SIZE, twixt.Game.SIZE, 10)
    c8 = numpy.count_nonzero(p1[:, :, 8])
    c9 = numpy.count_nonzero(p1[:, :, 9])
    if abs(c8 - c9) > 1:
        print("Zut!! c8=%d, c9=%d" % (c8, c9), file=sys.stderr)
        print("p8:", file=sys.stderr)
        print(binary_array_string(p1[:, :, 8]), file=sys.stderr)
        print("p9:", file=sys.stderr)
        print(binary_array_string(p1[:, :, 9]), file=sys.stderr)

    assert abs(c8 - c9) <= 1
    return p1


def old_naf_to_ninputs(naf):
    pegs = naf[:, :, 8:]
    links = numpy.copy(naf[:, :, :8])
    for index in range(8):
        vertical = index & twixt.Game.LINK_LONGY
        diffsign = index & twixt.Game.LINK_DIFFSIGN

        if vertical or diffsign:
            links[1:, :, index] = links[:-1, :, index]
            links[0, :, index] = 0
        if (not vertical) or diffsign:
            links[:, 1:, index] = links[:, :-1, index]
            links[:, 0, index] = 0

    return pegs, links


def hflip_policy_array(array):
    S = twixt.Game.SIZE
    rect = numpy.reshape(array, (S - 2, S))
    r2 = numpy.flip(rect, 0)
    return numpy.reshape(r2, (S * (S - 2),))


def vflip_policy_array(array):
    S = twixt.Game.SIZE
    rect = numpy.reshape(array, (S - 2, S))
    r2 = numpy.flip(rect, 1)
    return numpy.reshape(r2, (S * (S - 2),))


def policy_index_point(thing, index):
    if isinstance(thing, twixt.Game):
        color = thing.turn
    elif isinstance(thing, int):
        color = thing
    else:
        raise ValueError("Bad type for thing")
    assert color in (0, 1)

    major, minor = divmod(index, twixt.Game.SIZE)

    assert 0 <= major and major < twixt.Game.SIZE - 2
    assert 0 <= minor and minor < twixt.Game.SIZE

    if color == twixt.Game.WHITE:
        return twixt.Point(major + 1, minor)
    else:
        return twixt.Point(minor, major + 1)


def policy_point_index(thing, point):
    if isinstance(thing, twixt.Game):
        color = thing.turn
    elif isinstance(thing, int):
        color = thing
    else:
        raise ValueError("Bad type for thing")
    assert color in (0, 1)

    if color == twixt.Game.WHITE:
        major = point.x - 1
        minor = point.y
    else:
        major = point.y - 1
        minor = point.x

    assert 0 <= major and major < twixt.Game.SIZE - 2, (major, minor)
    assert 0 <= minor and minor < twixt.Game.SIZE, (major, minor)

    return major * twixt.Game.SIZE + minor


def legal_move_policy_array(game):
    if game.turn == game.WHITE:
        pegsum = (game.pegs[0][1:game.SIZE - 1, :] +
                  game.pegs[1][1:game.SIZE - 1, :]).flatten()
    else:
        assert game.turn == game.BLACK
        pegsum = (game.pegs[0][:, 1:game.SIZE - 1] +
                  game.pegs[1][:, 1:game.SIZE - 1]).T.flatten()
    return 1 - pegsum


def binary_array_string(arr):
    if len(arr.shape) == 1:
        return "".join("." if x == 0 else "*" for x in arr)
    elif len(arr.shape) == 2:
        return "\n".join(binary_array_string(arr[:, y] for y in range(arr.shape[1])))
    else:
        raise ValueError("only one/two dimensional arrays handled")

######################


class LearningState:
    NUM_MOVES = twixt.Game.SIZE * (twixt.Game.SIZE - 2)
    NAF_BYTES = NetInputs.COMPACT_SIZE
    NUM_BYTES = NAF_BYTES + 2 * NUM_MOVES + 1

    def __init__(self, game):

        self.nips = NetInputs(game)
        self.N = None
        self.z = None

    def to_bytes(self):

        b = bytearray()
        b0 = self.nips.to_compact_bytes()
        b1 = self.N.astype(numpy.uint16).tobytes()
        assert len(b0) == LearningState.NAF_BYTES
        assert len(b1) == 2 * LearningState.NUM_MOVES

        b.extend(b0)
        b.extend(b1)
        b.append(int(self.z) + 1)
        assert len(b) == self.NUM_BYTES
        return bytes(b)

    @staticmethod
    def from_bytes(b, name=None):

        assert len(b) == LearningState.NUM_BYTES
        ls = LearningState(None)
        ls.nips = NetInputs(b[0:LearningState.NAF_BYTES])
        ls.N = numpy.frombuffer(
            b, dtype=numpy.uint16, count=LearningState.NUM_MOVES, offset=LearningState.NAF_BYTES)
        ls.z = ord(b[-1]) - 1
        ls.name = name
        return ls


def location_inputs(dest=None):
    S = twixt.Game.SIZE
    a = numpy.arange(0, 1, 1.0 / twixt.Game.SIZE, dtype=numpy.float32)
    b = numpy.tile(a, (S, 1))
    if dest is None:
        c = numpy.zeros((S, S, 2))
    else:
        c = dest
    c[:, :, 0] = b
    c[:, :, 1] = b.T
    return c


NUM_ROTATIONS = 4
HFLIP_BIT = 1
VFLIP_BIT = 2


def single_move_policy_array(thing, move):
    S = twixt.Game.SIZE
    a = numpy.zeros(S * (S - 2), dtype=numpy.uint16)
    a[policy_point_index(thing, move)] = 1
    return a


def rotate_policy_array(pa, r):
    assert r >= 0 and r < NUM_ROTATIONS
    x = pa
    if r & HFLIP_BIT:
        x = hflip_policy_array(x)
    if r & VFLIP_BIT:
        x = vflip_policy_array(x)
    return x

# ##


def three_to_one(three):
    """ Take a three-vector of logits and return a score between -1 and 1 """
    lL, lD, lW = three
    eL = math.exp(lL - lD)
    eW = math.exp(lW - lD)
    div = 1.0 + eL + eW
    pW = eW / div
    pL = eL / div
    return pW - pL


def one_to_three(one):
    """ Take a score -1, 0, or 1, and return the three vector of labels """
    return ((1, 0, 0), (0, 1, 0), (0, 0, 1))[one + 1]
