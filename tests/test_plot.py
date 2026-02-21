import matplotlib

matplotlib.use("Agg")
import unittest
from src.plot import ThreeBarPlot, EvalHistPlot


class DummyCanvas:
    pass


class DummyStgs:
    def get(self, key):
        return "#000000"


class DummyAgg:
    def draw(self):
        pass


class DummySubPlot:
    def clear(self):
        pass

    def invert_yaxis(self):
        pass

    def set_xlim(self, xmin=None, xmax=None):
        pass

    def barh(self, ind, y, color=None, tick_label=None):
        pass

    def text(self, x, y, s, color=None, fontfamily=None, fontsize=None):
        pass

    def bar(self, keys, values, color=None):
        pass


# Patch prepare to return dummy objects
import src.plot

src.plot.prepare = lambda canvas: (DummySubPlot(), DummyAgg())


class TestThreeBarPlot(unittest.TestCase):
    def test_init_and_update(self):
        plot = ThreeBarPlot(DummyCanvas(), bar_color="#ff0000")
        values = {"moves": ["a", "b", "c"], "Y": [1, 2, 3]}
        plot.update(values, xmax=10)
        plot.update(None, xmax=10)
        self.assertEqual(plot.bar_color, "#ff0000")


class PatchedEvalHistPlot(EvalHistPlot):
    def prepare(self, canvas):
        self.sub_plot, self.agg = DummySubPlot(), DummyAgg()


class TestEvalHistPlot(unittest.TestCase):
    def test_init_and_update(self):
        plot = PatchedEvalHistPlot(DummyCanvas(), DummyStgs())
        values = {"a": 1, "b": -1, "c": 0}
        plot.update(values)
        plot.update(None)
        self.assertIsInstance(plot.stgs, DummyStgs)


if __name__ == "__main__":
    unittest.main()
