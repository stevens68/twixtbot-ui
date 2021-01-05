#! /usr/bin/env  -- python
import math

import graphics as gr
import twixt
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
import time

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.figure import Figure


def ifelse0(condition, value):
    if condition:
        return value
    else:
        return 0


class TBWHistory:
    def __init__(self, move):
        self.move = move
        self.objects = []


class TwixtBoardWindow:
    CELL = 25

    SIZE = twixt.Game.SIZE
    PEG_RADIUS = CELL // 4
    HOLE_RADIUS = 2

    # Return me two diagonal points near board edge
    def twopoints(self, i):
        xbit = (i ^ (i >> 1)) & 1
        ybit = (i >> 1) & 1
        x0 = (1 + xbit * self.SIZE) * self.CELL
        y0 = (1 + ybit * self.SIZE) * self.CELL
        x1 = (2 + xbit * (self.SIZE - 2)) * self.CELL
        y1 = (2 + ybit * (self.SIZE - 2)) * self.CELL
        return [gr.Point(x0, y0), gr.Point(x1, y1)]

    def center(self, x, y):
        return gr.Point((x + 1) * self.CELL + self.CELL / 2, (y + 1) * self.CELL + self.CELL / 2)

    def __init__(self, name="TwixT board"):
        boardLen = self.CELL * (2 + self.SIZE)
        w, h = boardLen, boardLen
        x, y = 50, 50  # Screen position.
        self.win = gr.GraphWin(name, w, h, autoflush=False)
        self.win.master.geometry('%dx%d+%d+%d' % (w, h, x, y))

        self.history = []
        self.known_moves = set()

        self.eta = None
        self.lastBest = None
        # Regular background
        self.win.setBackground("lightgrey")

        # black/white end zones
        for i in range(4):
            a = self.twopoints(i)
            b = self.twopoints(i + 1)

            if i == 0:
                color = "red"
                p = gr.Point(a[1].getX() + self.CELL * .2, a[1].getY())
                q = gr.Point(b[1].getX() - self.CELL * .2, b[1].getY())
            elif i == 1:
                color = "black"
                p = gr.Point(b[1].getX(), b[1].getY() - self.CELL * .2)
                q = gr.Point(a[1].getX(), a[1].getY() + self.CELL * .2)
            elif i == 2:
                color = "red"
                p = gr.Point(a[1].getX() - self.CELL * .2, a[1].getY())
                q = gr.Point(b[1].getX() + self.CELL * .2, b[1].getY())
            elif i == 3:
                color = "black"
                p = gr.Point(b[1].getX(), b[1].getY() + self.CELL * .2)
                q = gr.Point(a[1].getX(), a[1].getY() - self.CELL * .2)

            line = gr.Line(p, q)
            line.setFill(color)
            line.setOutline(color)
            line.setWidth(3)
            line.draw(self.win)

        # peg holes
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                if x in (0, self.SIZE - 1) and y in (0, self.SIZE - 1):
                    continue

                c = gr.Circle(self.center(x, y), self.HOLE_RADIUS)
                c.setFill("grey")
                c.setOutline("grey")
                c.draw(self.win)

        # labels
        for i in range(self.SIZE):
            ctr = self.center(i, i)
            row_label = "%d" % (i + 1)
            txt = gr.Text(gr.Point(self.CELL / 2, ctr.y), row_label)
            txt.setSize(9)
            txt.draw(self.win)
            txt = txt.clone()
            txt.move(self.CELL * (self.SIZE + 1), 0)
            txt.draw(self.win)

            col_label = chr(ord('A') + i)
            txt = gr.Text(gr.Point(ctr.x, self.CELL / 2), col_label)
            txt.setSize(9)
            txt.draw(self.win)
            txt = txt.clone()
            txt.move(0, self.CELL * (self.SIZE + 1))
            txt.draw(self.win)

        gr.update()

    def set_game(self, game):
        i = 0
        while i < len(game.history) and i < len(self.history):
            if game.history[i] != self.history[i].move:
                break
            i += 1

        # Get rid of now unneeded objects
        if i < len(self.history):
            for h in reversed(self.history[i:]):
                for o in h.objects:
                    o.undraw()
                self.known_moves.remove(h.move)
            self.history = self.history[:i]

        # Add new objects
        while i < len(game.history):
            self.create_move_objects(game, i)
            i += 1

        gr.update()

    def set_naf(self, naf, rotate=False):
        for h in reversed(self.history):
            for o in h.objects:
                o.undraw()
        self.history = []
        self.known_moves = set()

        if not rotate:
            def xyp(x, y): return twixt.Point(x, y)

            def pp(p): return p

            def cp(c): return 1 - c
        else:
            def xyp(x, y): return twixt.Point(y, x)

            def cp(c): return c

            def pp(p): return twixt.Point(p.y, p.x)

        objs = []

        for x, y, i in zip(*naf[:, :, 8:].nonzero()):
            objs.append(self._create_drawn_peg(xyp(x, y), cp(i & 1)))

        for x, y, j in zip(*naf[:, :, :8].nonzero()):
            link = twixt.Game.describe_link(j, x, y)
            objs.append(self._create_drawn_link(
                pp(link.p1), pp(link.p2), cp(i & 1)))

        nho = TBWHistory("nninputs")
        self.history.append(nho)
        nho.objects = objs

    def set_nn_inputs(self, pegs, links, rotate=False):
        for h in reversed(self.history):
            for o in h.objects:
                o.undraw()
        self.history = []
        self.known_moves = set()

        if not rotate:
            def xyp(x, y): return twixt.Point(x, y)

            def pp(p): return p

            def cp(c): return 1 - c
        else:
            def xyp(x, y): return twixt.Point(y, x)

            def cp(c): return c

            def pp(p): return twixt.Point(p.y, p.x)

        objs = []

        for x, y, i in zip(*pegs.nonzero()):
            objs.append(self._create_drawn_peg(xyp(x, y), cp(i)))

        i_px_py = []
        # color = game.WHITE
        for vertical in [False, True]:
            for diff_sign in [False, True]:
                for as_me in [False, True]:
                    index = ifelse0(vertical, twixt.Game.LINK_LONGY)
                    index += ifelse0(diff_sign, twixt.Game.LINK_DIFFSIGN)
                    index += 1 if as_me else 0
                    pad_x = ifelse0(vertical or diff_sign, 1)
                    pad_y = ifelse0(not vertical or diff_sign, 1)

                    i_px_py.append((index, pad_x, pad_y))

        for x, y, j in zip(*links.nonzero()):
            index, pad_x, pad_y = i_px_py[j]
            lx = x - pad_x
            ly = y - pad_y
            desc = twixt.Game.describe_link(index, lx, ly)
            c1 = 2 - 2 * pegs[desc.p1.x, desc.p1.y, 0] - \
                pegs[desc.p1.x, desc.p1.y, 1]
            c2 = 2 - 2 * pegs[desc.p2.x, desc.p2.y, 0] - \
                pegs[desc.p2.x, desc.p2.y, 1]
            if c1 == 0 and c2 == 0:
                color = 0
            elif c1 == 1 and c2 == 1:
                color = 1
            else:
                color = 2
            objs.append(self._create_drawn_link(
                pp(desc.p1), pp(desc.p2), color))

        nho = TBWHistory("nninputs")
        self.history.append(nho)
        nho.objects = objs
        # end set_nn_inputs

    def _create_drawn_peg(self, point, color):
        peg = gr.Circle(self.center(point.x, point.y), self.PEG_RADIUS)
        if color == twixt.Game.WHITE:
            peg.setWidth(2)
            peg.setOutline("red")
            peg.setFill("red")
        else:
            peg.setOutline("black")
            peg.setFill("black")
        peg.draw(self.win)
        return peg

    def create_move_objects(self, game, index):
        move = game.history[index]
        #assert isinstance(move, twixt.Point), "Swap not handled yet"
        if (not isinstance(move, twixt.Point)):
            # swap: flip first move and flip color
            c1 = self.history.pop().objects[0]
            c1.undraw()
            self.known_moves.clear()
            m1 = game.history[0]
            move = twixt.Point(m1.y, m1.x)
            game.use_swap = True

        color = (index + 1) & 1

        nho = TBWHistory(move)
        self.history.append(nho)

        nho.objects.append(self._create_drawn_peg(move, color))
        self.known_moves.add(move)

        for dlink in game.DLINKS:
            other = move + dlink
            if other in self.known_moves:
                if game.safe_get_link(move, other, color):
                    nho.objects.append(
                        self._create_drawn_link(move, other, color))

        # end create_move_objects()

    def _create_drawn_link(self, p1, p2, color):
        #carray = [gr.color_rgb(0,0,0), gr.color_rgb(150,150,150), gr.color_rgb(255,0,0)]
        carray = ["black", "red", "red"]
        c1 = self.center(*p1)
        c2 = self.center(*p2)
        dx = c2.x - c1.x
        dy = c2.y - c1.y
        hypot = math.hypot(dx, dy)
        cos = dx / hypot
        sin = dy / hypot
        lx1 = int(c1.x + cos * self.PEG_RADIUS + 0.5)
        ly1 = int(c1.y + sin * self.PEG_RADIUS + 0.5)
        lx2 = int(c2.x - cos * self.PEG_RADIUS + 0.5)
        ly2 = int(c2.y - sin * self.PEG_RADIUS + 0.5)
        line = gr.Line(gr.Point(lx1, ly1), gr.Point(lx2, ly2))
        line.setWidth(5)
        line.setFill(carray[color])
        line.draw(self.win)
        return line


class TwixtControlWindow:

    def __init__(self, name="Control"):
        w, h = 450, 650
        x, y = 800, 50  # Screen position.

        self.win = tk.Tk()

        self.win.title("TwixT control")
        self.win.geometry('%dx%d+%d+%d' % (w, h, x, y))

        r = 0
        turnLabel = tk.Label(self.win, text="turn:")
        turnLabel.grid(row=r + 0, column=1, sticky="W", padx=5, pady=2)
        self.turnText = tk.Label(self.win, fg="red", text="RED")
        self.turnText.grid(row=r + 0, column=2, sticky="W", padx=5, pady=2)

        currentBestLabel = tk.Label(self.win, text="current best:")
        currentBestLabel.grid(row=r + 0, column=3, sticky="W", padx=5, pady=2)
        self.currentBestBox = scrolledtext.ScrolledText(
            self.win, height=3, width=16)
        self.currentBestBox.configure(font=("Consolas", 9))
        self.currentBestBox.tag_config("red", foreground="red")
        self.currentBestBox.grid(
            row=r + 1, column=3, rowspan=3, sticky="EWNS", padx=5, pady=(0, 5))

        r += 1
        trialsLabel = tk.Label(self.win, text="trials:")
        trialsLabel.grid(row=r + 0, column=1, sticky="W", padx=5, pady=2)

        self.trialsText = tk.Label(self.win, text="")
        self.trialsText.grid(row=r + 0, column=2, sticky="W", padx=5, pady=2)

        r += 1
        progressLabel = tk.Label(self.win, text="progress:")
        progressLabel.grid(row=r + 0, column=1, sticky="W", padx=5)

        self.progressBar = ttk.Progressbar(
            self.win, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progressBar.grid(row=r + 0, column=2, sticky="W", padx=5, pady=2)

        r += 1
        etaLabel = tk.Label(self.win, text="time remaining:")
        etaLabel.grid(row=r + 0, column=1, sticky="W", padx=5, pady=2)
        self.etaText = tk.Label(self.win, text="")
        self.etaText.grid(row=r + 0, column=2, sticky="W", padx=5, pady=2)

        r += 1
        ttk.Separator(self.win, orient=tk.HORIZONTAL).grid(
            column=1, row=r, columnspan=3, sticky='we')
        # history
        historyLabel = tk.Label(self.win, text="history:")
        historyLabel.grid(row=r + 1, column=1, sticky="W", padx=5)

        self.historyBox = scrolledtext.ScrolledText(
            self.win, height=13, width=12)
        self.historyBox.configure(font=("Consolas", 9))
        self.historyBox.tag_config("red", foreground="red")
        self.historyBox.grid(row=r + 2, column=1, rowspan=16,
                             sticky="EWNS", padx=5, pady=2)

        expectedLine = tk.Label(self.win, text="expected line:")
        expectedLine.grid(row=r + 2, column=2, sticky="W", padx=5, pady=2)

        self.expLineBox = tk.Text(self.win, height=1, width=30)
        self.expLineBox.configure(font=("Consolas", 9))
        self.expLineBox.tag_config("red", foreground="red")
        self.expLineBox.grid(row=r + 2, column=3,
                             columnspan=2, sticky="W", padx=5, pady=2)

        evaLabel = tk.Label(self.win, text="last evaluation:")
        evaLabel.grid(row=r + 3, column=2, columnspan=1,
                      sticky="W", padx=5, pady=2)

        self.evaBox = tk.Text(self.win, height=1, width=10)
        self.evaBox.configure(font=("Consolas", 9))
        self.evaBox.grid(row=r + 3, column=3, columnspan=1,
                         sticky="W", padx=5, pady=2)

        f = Figure(figsize=(4, 2), dpi=80)
        self.evaPlot = f.add_subplot(111)
        self.evaPlot.spines['bottom'].set_position('center')
        # self.trend.plot([0],[0.0])
        self.axes = f.gca()
        self.axes.xaxis.set_ticklabels([])

        self.axes.set_ylim([-1.1, 1.1])

        self.canvas = FigureCanvasTkAgg(f, self.win)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=r + 4, column=2, columnspan=3,
                                         sticky="EW", padx=5, pady=2)

    def reset(self, game):
        if (game.turn == twixt.Game.WHITE):
            self.turnText.config(text="RED", fg="red")
        else:
            self.turnText.config(text="BLACK", fg="black")

        self.progressBar['value'] = 0.0
        #self.expLineBox.delete(1.0, tk.END)
        #self.evaBox.delete(1.0, tk.END)
        self.trialsText.config(text="")
        self.etaText.config(text="")
        self.eta = None
        self.lastBest = None
        self.currentBestBox.delete(1.0, tk.END)

    def update(self, game, forecast, trendline, expLine):

        self.historyBox.delete(1.0, tk.END)
        hStr = ""
        for i, m in enumerate(game.history):
            if i % 2 == 1:
                hStr += str(m).upper().ljust(3) + "\n"
            else:
                hStr += str(i // 2 + 1).rjust(2) + ". " + \
                    str(m).upper().ljust(3) + " "

        self.historyBox.insert(tk.END, hStr)
        self.historyBox.yview(tk.END)

        for i in range(len(game.history)):
            if i % 2 == 0:
                self.historyBox.tag_add("red", str(
                    i // 2 + 1) + "." + str(4), str(i // 2 + 1) + "." + str(4 + 3))

        self.expLineBox.delete(1.0, tk.END)
        expList = expLine.upper().split(",")[1:]
        for i, m in enumerate(expList):
            curLen = len(self.expLineBox.get("1.0", tk.END))
            self.expLineBox.insert(tk.END, m + " ")
            if (i % 2 != game.turn):
                self.expLineBox.tag_add(
                    "red", "1." + str(curLen - 1), "1." + str(curLen - 1 + len(m)))

        self.evaBox.delete(1.0, tk.END)
        self.evaBox.insert(tk.END, forecast)

        self.evaPlot.plot(range(len(game.history)), trendline)
        self.axes.xaxis.set_ticks(range(len(trendline)))
        labels = []
        for _ in range(len(trendline)):
            labels.append("")

        self.axes.xaxis.set_ticklabels(labels)
        self.canvas.draw()
        # end _pick_root_expand_index

    def updateProgress(self, num_evals, trials, good_moves):
        self.progressBar['value'] = 100.0 * float(num_evals) / float(trials)
        self.trialsText.config(text=str(num_evals) + "/" + str(trials))
        self.etaText.config(text=self.getRemainingTime(num_evals, trials))
        if (good_moves):
            cb = " ".join(str(x) for x in good_moves[0].tolist()) + "\n"
            if (cb != self.lastBest):
                self.currentBestBox.insert(tk.END, cb)
                self.lastBest = cb
        self.win.update_idletasks()

    def getRemainingTime(self, num_evals, trials):

        if (self.eta == None):
            self.eta = dict()
            self.eta['lastTimestamp'] = time.time()
            self.eta['lastEvals'] = num_evals
            self.eta['lastMillis'] = []
            return ""

        t = time.time()
        tup = (num_evals - self.eta['lastEvals'],
               int(round((t - self.eta['lastTimestamp']) * 1000)))
        self.eta['lastMillis'].append(tup)
        self.eta['lastEvals'] = num_evals
        self.eta['lastTimestamp'] = t

        len_last_t = len(self.eta['lastMillis'])
        if len_last_t > 5:
            self.eta['lastMillis'].pop(0)

        mean_t = sum([tup[1] for tup in self.eta['lastMillis']]) // len_last_t
        mean_n = sum([tup[0] for tup in self.eta['lastMillis']]) // len_last_t
        remain_tot = (((trials - num_evals) * mean_t) / mean_n) // 1000
        remain_m = int(remain_tot / 60)
        remain_s = int(remain_tot % 60)

        return format(remain_m, '02d') + ":" + format(remain_s, '02d')


if __name__ == "__main__":
    test = TwixtBoardWindow()
    game = twixt.Game()
    game.play("b3")
    game.play("j10")
    game.play("c5")
    game.play("j11")
    game.play("d3")
    game.play("j12")
    game.play("b4")

    test.set_game(game)
    test.win.getMouse()
    game.undo()
    test.set_game(game)
    test.win.getMouse()
    test.win.close()
