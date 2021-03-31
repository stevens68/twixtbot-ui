
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


class EvalHistPlot():
    def __init__(self, canvas, stgs):
        self.sub_plot = None
        self.agg = None
        self.prepare(canvas)
        self.stgs = stgs

    def sc_to_color(self, sc):
        if sc > 0:
            return self.stgs.get(ct.K_COLOR[1])
        return self.stgs.get(ct.K_COLOR[2])

    def update(self, values=None):
        # clear the subplot
        ax1 = self.sub_plot
        ax1.clear()

        if values is not None:
            ax1.bar(values.keys(), values.values(),
                    color=list(map(self.sc_to_color, values.values())))

            xmax = max(10, len(values))
            plt.xlim(-1, xmax)
            plt.xticks(np.arange(0, xmax, xmax // 6))
            plt.ylim([-1, 1])

        plt.subplots_adjust(left=None, bottom=0.3,
                            right=None, top=0.9, wspace=0, hspace=0)
        self.agg.draw()

    def prepare(self, canvas):
        fig, ax1 = plt.subplots(figsize=(2.4, 0.7))

        ax1.tick_params(axis='x', which='major', labelcolor="black",
                        labelsize=8, pad=.8, top=False, bottom=False)
        ax1.tick_params(axis='y', which='major', labelcolor="black",
                        labelsize=8, pad=.8)
        ax1.autoscale(True, axis='x', tight=True)

        ax1.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        agg = FigureCanvasTkAgg(fig, canvas)
        agg.get_tk_widget().pack()

        self.sub_plot = ax1
        self.agg = agg
