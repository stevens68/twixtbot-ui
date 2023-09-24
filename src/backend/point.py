import operator
from collections import namedtuple

class Point(namedtuple('Point', 'x y')):
    def __new__(cls, *args):

        if len(args) == 2:
            return tuple.__new__(cls, args)
        elif len(args) == 1:
            arg = args[0]
            if type(arg) == str:
                if arg[0] >= 'A' and arg[0] <= 'Z':
                    x = ord(arg[0]) - ord('A')
                else:
                    x = ord(arg[0]) - ord('a')
                y = int(arg[1:]) - 1
                return tuple.__new__(cls, (x, y))
            elif len(arg) == 2:
                return tuple.__new__(cls, (arg[0], arg[1]))

        raise ValueError("Bad initializer")

    def add_it(self, other):

        if isinstance(other, Point):
            return Point(operator.add(self.x, other.x),
                         operator.add(self.y, other.y))
        elif len(other) == 2:
            return Point(operator.add(self.x, other[0]),
                         operator.add(self.y, other[1]))
        else:
            raise ValueError("Cannot add")

    __add__ = add_it

    def subtract_it(self, other):

        if isinstance(other, Point):
            return Point(operator.sub(self.x, other.x),
                         operator.sub(self.y, other.y))
        elif len(other) == 2:
            return Point(operator.sub(self.x, other[0]),
                         operator.sub(self.y, other[1]))
        else:
            raise ValueError("Cannot subtract")

    __sub__ = subtract_it

    def radd_it(self, other):

        if len(other) == 2:
            return Point(operator.add(other[0], self.x),
                         operator.add(other[1], self.y))
        else:
            raise ValueError("Cannot add")

    __radd__ = radd_it

    def rsubtract_it(self, other):

        if len(other) == 2:
            return Point(operator.sub(other[0], self.x),
                         operator.sub(other[1], self.y))
        else:
            raise ValueError("Cannot subtract")

    __rsub__ = rsubtract_it

    def __mul__(self, other):

        return Point(self.x * other, self.y * other)

    __rmul__ = __mul__

    def __str__(self):

        return chr(self.x + ord('a')) + str(self.y + 1)

    def __repr__(self):

        return self.__str__()

    def flip(self):

        return Point(self.y, self.x)
