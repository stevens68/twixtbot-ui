# -*- coding: utf-8 -*-
import constants as ct


def p_to_rgbstring(p):
    """ Converts a probability ([0..1]) value to an RGB color string

    The heatmap is based on 3 anchor points: minimum value (p=0), middle
    value (p=0.5) and maximum value (p=1). Each of those points is related
    to a color, which are defined in constants.py (HEATMAP_RGB_COLORS).
    Intermediate colors are interpolated between those anchor points.

    Args:
        p: probability value (range: 0..1, both inclusive)

    Returns:
        an RGB string with 3 hex values (e.g.: '#RRGGBB')
    """
    assert(p >= 0 and p <= 1)

    # Determine the colors to be mixed and calculate factor [0..1]
    if p > 0.5:
        f = 2 * p - 1
        col1, col2 = ct.HEATMAP_RGB_COLORS[1:3]
    else:
        f = 2 * p
        col1, col2 = ct.HEATMAP_RGB_COLORS[0:2]

    # Apply factor for col1 and col2
    rgb = [int((1 - f) * rgb1 + f * rgb2)
           for rgb1, rgb2 in zip(col1, col2)]

    # convert rgb list to string and return
    return '#' + ''.join(f'{c:02x}' for c in rgb)


class Heatmap:
    """ Contains data and functions to plot a heatmap on the Twixt board

    The constructor also calculates the heatmap. After the
    constructor is called, it isn't nessecary to call .calculate()

    **ALWAYS** instantiate Heatmap with game and bot.

    Args:
        twixt.Game: a twixt game (Alpha Zero Twixt game object)
        twixt.Player: a twixt bot (Alpha Zero Twixt bot)

    Attributes:
        p_values (dict of twixt.Point: str): normalized (0..1) policy values
        rgb_colors (dict of twixt.Point: str): rgb colors

    Raises:
        ValueError if the class is initialized without game or bot.
    """

    def __init__(self, game=None, bot=None):
        if game is None or bot is None:
            raise ValueError('Instantiate Heatmap with game and bot!')

        self.game = game
        self.bot = bot
        self.p_values = {}
        self.rgb_colors = {}
        self.calculate()

    def heatmap_legend(self, num_steps=ct.HEATMAP_LEGEND_STEPS):
        """ Returns a list of heatmap RGB values for a heatmap legend

        Args:
            num_steps: length of the returned list minus 1

        Returns:
            a list of RGB strings
        """
        return [p_to_rgbstring(p / num_steps) for p in range(num_steps + 1)]

    def calculate(self):
        """ Calculates the heatmap by evaluating the policy of the bot

        This method updates the p_values and rgb_colors in the Heatmap.
        """
        assert(self.game is not None)
        assert(self.bot is not None)

        self.p_values = {}
        self.rgb_colors = {}

        sc, self.policy_moves, p_val = self.bot.nm.eval_game(
            self.game, maxbest=self.game.SIZE**2)
        p_val = [int(round(p * 1000)) for p in p_val]
        for m, p in zip(self.policy_moves, p_val):
            if p == 0:
                # all the rest of the p-values will be 0; break loop
                break
            self.p_values[m] = p / p_val[0]
            self.rgb_colors[m] = p_to_rgbstring(p / p_val[0])
