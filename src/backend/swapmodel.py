#! /usr/bin/env python
import bisect
import math
import numpy
import random
import sys
import logging
from .. import constants as ct
from . import twixt

logger = logging.getLogger(ct.LOGGER)


def _xy_predictors(xres, yres):
    return numpy.array([1.0, xres, yres, xres * yres])


def _point_predictors(point_for_predictors):
    x = point_for_predictors.x
    y = point_for_predictors.y
    s2 = twixt.Game.SIZE // 2
    if x >= s2:
        x = 2 * s2 - x - 1
    if y >= s2:
        y = 2 * s2 - y - 1
    assert 0 < x < s2 and 0 <= y < s2

    xres = x - 6.0
    yres = y - 5.5
    return _xy_predictors(xres, yres)


# i=0 beta=0.494481 t=56.1998
# i=1 beta=-0.00366079 t=-1.37993
# i=2 beta=0.0225597 t=9.21498
# i=3 beta=0.00114293 t=1.54838


# _betas = numpy.array([0.491727, -0.00391336, 0.0223454, 0.0014858])
_betas = numpy.array([0.494481, -0.00366079, 0.0225597, 0.00114293])


def _point_score(point_for_score):
    return numpy.dot(_betas, _point_predictors(point_for_score))


_halflife = 0.008


def want_swap(point_for_swap):
    return _point_score(point_for_swap) > 0.50


def points_and_locs():
    cum = 0.0
    locations = [0.0]
    points = []
    for x in range(1, twixt.Game.SIZE - 1):
        for y in range(twixt.Game.SIZE):
            point_for_list = twixt.Point(x, y)
            score = _point_score(point_for_list)
            weight = math.exp(math.log(0.5) * abs(score - 0.5) / _halflife)
            cum += weight
            locations.append(cum)
            points.append(point_for_list)

    return points, locations


def first_move_report():
    points, locations = points_and_locs()
    cum = locations[-1]
    for i, point_for_report in enumerate(points):
        if point_for_report.x >= twixt.Game.SIZE // 2 or point_for_report.y >= twixt.Game.SIZE // 2:
            continue
        pct = 4.0 * (locations[i + 1] - locations[i]) / cum
        logger.info("%3s %5.2f" % (str(point_for_report), pct * 100))


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
        logger.info(choose_first_move())
    elif len(sys.argv) == 2 and sys.argv[1] == "all":
        first_move_report()
    elif len(sys.argv) == 2:
        point_from_arg = twixt.Point(sys.argv[1])
        logger.info(_point_score(point_from_arg), want_swap(point_from_arg))
