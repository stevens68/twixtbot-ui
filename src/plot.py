
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

import constants as ct


class ThreeBarPlot():

    def __init__(self, canvas, bar_color):
        self.bar_color = bar_color
        self.prepare(canvas)

    def update(self, values=None, xmax=None):

        # clear the subplot
        ax1 = self.sub_plot
        ax1.clear()
        ax1.invert_yaxis()

        if values is not None:
            ind = np.arange(3)
            labels = [str(m).upper() for m in values["moves"]]
            if "Y" in values:
                ax1.set_xlim(xmin=0, xmax=xmax)
                ax1.barh(ind, values["Y"],
                         color=self.bar_color, tick_label=labels)
                offset = xmax * 0.02
                for i, v in enumerate(values["Y"]):
                    ax1.text(v + offset, i + 0.27, str(v),
                             color=ct.PLOT_LABEL_COLOR, fontfamily=ct.PLOT_LABEL_FONT[0], fontsize=ct.PLOT_LABEL_FONT[1])

        plt.subplots_adjust(left=None, bottom=None,
                            right=None, top=None, wspace=0, hspace=0)
        self.agg.draw()

    def prepare(self, canvas):

        fig, ax1 = plt.subplots(figsize=(2.4, 0.7))

        ax1.tick_params(axis='x', which='major',
                        labelcolor="black",  labeltop=False, labelbottom=False, top=False, bottom=False)
        ax1.tick_params(axis='y', which='major',
                        labelcolor="black",  labelleft=True, labelsize=8, pad=.8)

        ax1.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        agg = FigureCanvasTkAgg(fig, canvas)
        agg.get_tk_widget().pack()

        self.sub_plot = ax1
        self.agg = agg