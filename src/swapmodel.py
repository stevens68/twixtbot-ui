#! /usr/bin/env python
import bisect
import math
import numpy
import random
import sys
import twixt


def _xy_predictors(xres, yres):
    return numpy.array([1.0, xres, yres, xres * yres])


def _point_predictors(p):
    x = p.x
    y = p.y
    S2 = twixt.Game.SIZE // 2
    if x >= S2:
        x = 2 * S2 - x - 1
    if y >= S2:
        y = 2 * S2 - y - 1
    assert x > 0 and x < S2 and y >= 0 and y < S2

    xres = x - 6.0
    yres = y - 5.5
    return _xy_predictors(xres, yres)

# i=0 beta=0.494481 t=56.1998
# i=1 beta=-0.00366079 t=-1.37993
# i=2 beta=0.0225597 t=9.21498
# i=3 beta=0.00114293 t=1.54838


# _betas = numpy.array([0.491727, -0.00391336, 0.0223454, 0.0014858])
_betas = numpy.array([0.494481, -0.00366079, 0.0225597, 0.00114293])


def _point_score(point):
    return numpy.dot(_betas, _point_predictors(point))


_halflife = 0.008


def want_swap(point):
    return _point_score(point) > 0.50


def points_and_locs():
    cum = 0.0
    locations = [0.0]
    points = []
    for x in range(1, twixt.Game.SIZE - 1):
        for y in range(twixt.Game.SIZE):
            point = twixt.Point(x, y)
            score = _point_score(point)
            weight = math.exp(math.log(0.5) * abs(score - 0.5) / _halflife)
            cum += weight
            locations.append(cum)
            points.append(point)

    return points, locations


def first_move_report():
    points, locations = points_and_locs()
    cum = locations[-1]
    for i, p in enumerate(points):
        if p.x >= twixt.Game.SIZE // 2 or p.y >= twixt.Game.SIZE // 2:
            continue
        pct = 4.0 * (locations[i + 1] - locations[i]) / cum
        print("%3s %5.2f" % (str(p), pct * 100))


def choose_first_move():
    points, locations = points_and_locs()
    cum = locations[-1]

    z = random.uniform(0, cum)
    i = bisect.bisect(locations, z)
    if i == len(locations):
        i -= 1
    return points[i]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(choose_first_move())
    elif len(sys.argv) == 2 and sys.argv[1] == "all":
        first_move_report()
    elif len(sys.argv) == 2:
        p = twixt.Point(sys.argv[1])
        print(_point_score(p), want_swap(p))
