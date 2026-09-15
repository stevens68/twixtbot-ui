import unittest
import re
from src.heatmap import Heatmap, p_to_rgb_string, heatmap_legend
import src.constants as ct
from src.backend.point import Point


class DummyGame:
    SIZE = 24


class DummyNM:
    def __init__(self, moves=None, p_vals=None):
        self.moves = moves if moves is not None else []
        self.p_vals = p_vals if p_vals is not None else []

    def eval_game(self, game, maxbest=None):
        return 0.5, self.moves, self.p_vals, []


class DummyBot:
    def __init__(self, moves=None, p_vals=None):
        self.nm = DummyNM(moves, p_vals)


class TestHeatmapFunctions(unittest.TestCase):
    def test_p_to_rgb_string_anchors(self):
        # Anchor points: 0.0, 0.5, 1.0
        hex_0 = p_to_rgb_string(0.0)
        hex_mid = p_to_rgb_string(0.5)
        hex_1 = p_to_rgb_string(1.0)

        # Expected anchor hex strings from ct.HEATMAP_RGB_COLORS
        expected_0 = "#" + "".join(f"{c:02x}" for c in ct.HEATMAP_RGB_COLORS[0])
        expected_mid = "#" + "".join(f"{c:02x}" for c in ct.HEATMAP_RGB_COLORS[1])
        expected_1 = "#" + "".join(f"{c:02x}" for c in ct.HEATMAP_RGB_COLORS[2])

        self.assertEqual(hex_0, expected_0)
        self.assertEqual(hex_mid, expected_mid)
        self.assertEqual(hex_1, expected_1)

    def test_p_to_rgb_string_format(self):
        hex_pattern = re.compile(r"^#[0-9a-f]{6}$")
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            rgb_str = p_to_rgb_string(p)
            self.assertTrue(hex_pattern.match(rgb_str), f"Invalid hex color format: {rgb_str}")

    def test_p_to_rgb_string_out_of_bounds(self):
        with self.assertRaises(AssertionError):
            p_to_rgb_string(-0.01)
        with self.assertRaises(AssertionError):
            p_to_rgb_string(1.01)

    def test_heatmap_legend(self):
        legend_default = heatmap_legend()
        self.assertEqual(len(legend_default), ct.HEATMAP_LEGEND_STEPS + 1)
        self.assertEqual(legend_default[0], p_to_rgb_string(0.0))
        self.assertEqual(legend_default[-1], p_to_rgb_string(1.0))

        custom_steps = 10
        legend_custom = heatmap_legend(num_steps=custom_steps)
        self.assertEqual(len(legend_custom), custom_steps + 1)


class TestHeatmap(unittest.TestCase):
    def test_init_missing_args(self):
        with self.assertRaises(ValueError):
            Heatmap()
        with self.assertRaises(ValueError):
            Heatmap(game=DummyGame())
        with self.assertRaises(ValueError):
            Heatmap(bot=DummyBot())

    def test_init_with_args(self):
        h = Heatmap(game=DummyGame(), bot=DummyBot())
        self.assertIsInstance(h, Heatmap)
        self.assertIsInstance(h.p_values, dict)
        self.assertIsInstance(h.rgb_colors, dict)
        self.assertIsInstance(h.policy_moves, list)

    def test_calculate_with_moves(self):
        m1 = Point(5, 5)
        m2 = Point(6, 7)
        m3 = Point(8, 9)
        moves = [m1, m2, m3]
        p_vals = [0.8, 0.4, 0.0]

        bot = DummyBot(moves=moves, p_vals=p_vals)
        game = DummyGame()
        h = Heatmap(game=game, bot=bot)

        # p_values are normalized relative to top move (p_val / p_val[0])
        # p_val is multiplied by 1000 and rounded: 800 and 400
        self.assertIn(m1, h.p_values)
        self.assertIn(m2, h.p_values)
        self.assertNotIn(m3, h.p_values)  # 0 probability breaks the loop

        self.assertAlmostEqual(h.p_values[m1], 1.0)
        self.assertAlmostEqual(h.p_values[m2], 0.5)

        self.assertIn(m1, h.rgb_colors)
        self.assertIn(m2, h.rgb_colors)
        self.assertEqual(h.rgb_colors[m1], p_to_rgb_string(1.0))
        self.assertEqual(h.rgb_colors[m2], p_to_rgb_string(0.5))


if __name__ == "__main__":
    unittest.main()
