
import PySimpleGUI as sg
import threading

import backend.twixt as twixt

import util.pmeter as pmeter

import constants as ct
import settings as st
import layout as lt
import files as fi
import plot as pt
import heatmap as hm
import uiboard

from tkinter import ttk


def popup_bot_in_progress():
    lt.popup("bot in progress. Click Accept or Cancel.")


class BotEvent(threading.Event):
    # used to stop or cancel a bot (thread)

    def __init__(self):
        super().__init__()

    def get_context(self):
        return self.context

    def set(self, context):
        super().set()
        self.context = context


class TwixtbotUI():

    def __init__(self, game, stgs, board):
        self.board = board
        self.game = game
        self.moves_score = {}
        self.stgs = stgs
        layout = lt.MainWindowLayout(board, stgs).get_layout()
        self.window = sg.Window(ct.WINDOW_TITLE, layout, margins=(25, 25))

        # create the objects to update the bar chart

        self.window.Finalize()

        canvas = self.window[ct.K_EVAL_MOVES[1]].TKCanvas
        self.eval_moves_plot = pt.ThreeBarPlot(canvas, ct.EVAL_PLOT_COLOR)

        canvas = self.window[ct.K_VISITS[1]].TKCanvas
        self.visit_plot = pt.ThreeBarPlot(canvas, ct.VISITS_PLOT_COLOR)
        self.visit_plot.update()

        canvas = self.window[ct.K_EVAL_HIST[1]].TKCanvas
        self.eval_hist_plot = pt.EvalHistPlot(canvas, stgs)

        self.update_settings_changed()
        self.prepare_bots()
        self.update_evals()

    def __del__(self):
        if hasattr(self, 'window'):
            self.window.close()
        del self.stgs
        del self.game
        del self.board

    def get_control(self, key, player=None):
        return self.window[key[1]] if player is None else self.window[key[player]]

    def get_current(self, key):
        return self.stgs.get_current(key, self.game)

    def set_current(self, key, value):
        return self.stgs.set_current(key, value, self.game)

    # update_ functions to update ui controls

    def update_tooltips(self):
        self.get_control(ct.K_AUTO_MOVE, 1).set_tooltip(
            self.stgs.get_tooltip(1))
        self.get_control(ct.K_AUTO_MOVE, 2).set_tooltip(
            self.stgs.get_tooltip(2))

    def update_turn_indicators(self):
        turn = ["", ""]
        if self.game.result == twixt.RESIGN:
            turn[self.game.turn] = ""
            turn[1 - self.game.turn] = ct.TURN_RESIGNED
        elif self.game.just_won():
            turn[self.game.turn] = ct.TURN_HAS_WON
            turn[1 - self.game.turn] = ""
        elif self.game.turn == 1:
            turn = [ct.TURN_CHAR, '']
        else:
            turn = ['', ct.TURN_CHAR]

        self.get_control(ct.K_TURN_INDICATOR, 1).Update(turn[0])
        self.get_control(ct.K_TURN_INDICATOR, 2).Update(turn[1])

    def update_history(self):
        text = ""
        for i, move in enumerate(self.game.history):
            text += "\n" if i > 0 and i % 2 == 0 else ""
            text += str(i + 1).rjust(2, ' ') + '. ' + str(move).upper()

            if move == twixt.SWAP:
                m1 = self.game.history[0]
                text += " " + chr(m1.y + ord('A')) + str(m1.x + 1)

            text += "\t\t" if i % 2 == 0 else ""

        self.get_control(ct.K_MOVES).Update(text)

    def calc_eval(self):
        score, moves, P = self.bots[self.game.turn].nm.eval_game(self.game)
        # get score from white's perspective
        self.next_move = score, moves, P
        sc = round((2 * self.game.turn - 1) * score, 3)
        # Add sc to dict of historical scores
        self.moves_score[len(self.game.history)] = sc

        return sc, moves, P

    def update_evals(self):
        if not self.game_over(False):
            sc, moves, P = self.calc_eval()

            self.get_control(ct.K_EVAL_NUM).Update(sc)
            self.get_control(ct.K_EVAL_BAR).Update(1000 * sc + 1000)

            # update chart
            values = {"moves": moves, "Y": P}
            self.eval_moves_plot.update(values, 1000)

        else:
            self.get_control(ct.K_EVAL_NUM).Update('')
            self.get_control(ct.K_EVAL_BAR).Update(0)

            self.next_move = None

        # clean visits
        self.visit_plot.update()
        self.eval_hist_plot.update(self.moves_score)

    def update_evalbar_colors(self):
        s = ttk.Style()
        ebs = self.window[ct.K_EVAL_BAR[1]].TKProgressBar.style_name
        s.configure(ebs, background=self.stgs.get_setting(ct.K_COLOR[1]))
        s.configure(ebs, troughcolor=self.stgs.get_setting(ct.K_COLOR[2]))

    def update_progress(self, values=None):
        if values is None:
            text = ""
            value = 0
            max_value = 0
        else:
            max_value = values["max"]
            value = values["current"]

            if self.stgs.get_setting(ct.K_SMART_ACCEPT[1]) and "Y" in values:
                diff = values["Y"][0] - values["Y"][1]
                if diff > max_value - value:
                    # 2nd best cannot catch up => accept
                    self.handle_accept_bot()

                # reduce max val
                while diff > values["max"] - max_value + 20 and max_value >= value + 20:
                    max_value -= 20

            text = str(value) + "/" + str(max_value) + "      " + \
                str(round(100 * value / max_value)) + "%      "
            if value > 0:
                self.timer.update(value + values["max"] - max_value)
                text += self.timer.getstatus()

        self.get_control(ct.K_PROGRESS_NUM).Update(text)
        self.get_control(ct.K_PROGRESS_BAR).UpdateBar(value, max_value)

    def update_after_move(self):
        self.board.draw()
        self.window.refresh()

        # reset progress
        self.update_progress()

        self.update_turn_indicators()
        self.update_history()
        self.update_evals()

    def update_settings_changed(self):

        self.board.draw()
        self.window.refresh()
        # update ui
        for p in [1, 2]:
            self.get_control(ct.K_NAME, p).Update(
                self.stgs.get_setting(ct.K_NAME[p]))
            self.get_control(ct.K_COLOR, p).erase()
            self.get_control(ct.K_COLOR, p).DrawCircle((7, 9), 6,
                                                       self.stgs.get_setting(
                                                           ct.K_COLOR[p]),
                                                       self.stgs.get_setting(ct.K_COLOR[p]))
            self.get_control(ct.K_AUTO_MOVE, p).Update(
                self.stgs.get_setting(ct.K_AUTO_MOVE[p]))
            self.get_control(ct.K_TRIALS, p).Update(
                self.stgs.get_setting(ct.K_TRIALS[p]))

        self.update_turn_indicators()
        self.update_tooltips()
        self.update_evalbar_colors()
        self.eval_hist_plot.update(self.moves_score)
        self.update_bots()
        self.update_game()

    def reset_game(self):
        self.game.__init__(self.stgs.get_setting(ct.K_ALLOW_SCL[1]))
        self.moves_score = {}
        # get eval of empty board to avoid gap at x=0 in plot in loaded games
        self.calc_eval()

    def update_game(self):
        self.game.allow_scl = self.stgs.get_setting(ct.K_ALLOW_SCL[1])

    # bot functions

    def update_bots(self):
        for t in [0, 1]:
            if hasattr(self, 'bots') and self.bots[t] is not None:
                p = self.game.turn_to_player(t)
                self.bots[t].allow_swap = self.stgs.get_setting(
                    ct.K_ALLOW_SWAP[1])
                self.bots[t].num_trials = int(
                    self.stgs.get_setting(ct.K_TRIALS[p]))
                self.bots[t].temperature = float(
                    self.stgs.get_setting(ct.K_TEMPERATURE[p]))
                self.bots[t].random_rotation = self.stgs.get_setting(
                    ct.K_RANDOM_ROTATION[p])
                self.bots[t].add_noise = float(
                    self.stgs.get_setting(ct.K_ADD_NOISE[p]))
                self.bots[t].nm.cpuct = float(
                    self.stgs.get_setting(ct.K_CPUCT[p]))

    def init_bot(self, player):

        args = {
            "allow_swap": self.stgs.get_setting(ct.K_ALLOW_SWAP[1]),
            "model": self.stgs.get_setting(ct.K_MODEL_FOLDER[player]),
            "trials": self.stgs.get_setting(ct.K_TRIALS[player]),
            "temperature": self.stgs.get_setting(ct.K_TEMPERATURE[player]),
            "random_rotation": self.stgs.get_setting(ct.K_RANDOM_ROTATION[player]),
            "add_noise": self.stgs.get_setting(ct.K_ADD_NOISE[player]),
            "cpuct": self.stgs.get_setting(ct.K_CPUCT[player])

        }

        import backend.nnmplayer as nnmplayer
        self.bots[2 - player] = nnmplayer.Player(**args)

    def prepare_bots(self):
        lt.popup('initializing bots ... ')

        self.bots = [None, None]
        self.init_bot(1)
        self.init_bot(2)
        # warm-up bots before first move
        self.bots[0].nm.eval_game(self.game)
        self.bots[1].nm.eval_game(self.game)

    def bot_move(self):
        if len(self.game.history) >= 2 and self.get_current(ct.K_TRIALS) == 0 and self.next_move is not None:
            # we already have the next move from evaluation
            self.bots[self.game.turn].nm.send_message(self.window, self.game, "done",
                                                      0, 0, moves=self.next_move[1], P=self.next_move[2])
        else:
            # mcts, or first/second move
            self.bots[self.game.turn].pick_move(
                self.game, self.window, self.event)

    def launch_bot(self):
        self.visit_plot.update()
        self.window[ct.K_SPINNER[1]].Update(visible=True)
        self.event = BotEvent()
        self.thread = threading.Thread(
            target=self.bot_move, args=(), daemon=True)

        self.timer = pmeter.ETA(self.get_current(ct.K_TRIALS), max_seconds=300)
        self.thread.start()

    # handle events
    def handle_heatmap(self):
        lt.popup(ct.MSG_HEATMAP_CALCULATING)
        self.board.draw(hm.Heatmap(self.game, self.bots[self.game.turn]))

    def handle_board_click(self, values):
        if self.game_over():
            return
        move = self.board.get_move(values[ct.K_BOARD[1]])
        if move is not None:
            # clear move statistics
            self.execute_move(move)
            self.update_after_move()

    def handle_open_file(self):
        players, moves = fi.get_game()
        if players is None:
            return

        # assign player names
        self.stgs.settings[ct.K_NAME[1]] = players[0]
        self.stgs.settings[ct.K_NAME[2]] = players[1]
        self.update_settings_changed()

        # reset game
        self.reset_game()

        # replay game
        try:
            lt.popup("loading game...")
            for m in moves:
                self.execute_move(m)
                self.calc_eval()
                # self.update_after_move()
        except:
            lt.popup("invalid move: " + str(m))

        self.update_after_move()

    def handle_resign(self):
        if self.game_over():
            return
        self.execute_move(twixt.RESIGN)

    def handle_undo(self):
        if self.game_over():
            return

        gl = len(self.game.history)

        if gl in self.moves_score:
            del self.moves_score[gl]

        if gl > 0 and gl != 2:
            self.game.undo()
        elif gl == 2:
            # move 2 might be a swap move => reset game and redo move #1
            move_one = self.game.history[0]
            self.reset_game()
            self.execute_move(move_one)

        # switch off auto move
        if self.get_current(ct.K_AUTO_MOVE):
            self.set_current(ct.K_AUTO_MOVE, False)
            self.get_control(
                ct.K_AUTO_MOVE, self.game.turn_to_player()).Update(False)

    def handle_accept_bot(self):
        self.event.set(ct.ACCEPT_EVENT)

    def handle_cancel_bot(self):
        self.event.set(ct.CANCEL_EVENT)
        # switch off auto move
        if self.get_current(ct.K_AUTO_MOVE):
            self.set_current(ct.K_AUTO_MOVE, False)
            self.get_control(
                ct.K_AUTO_MOVE, self.game.turn_to_player()).Update(False)

    def handle_thread_event(self, values):
        print("Bot response: " + str(values))
        if values["max"] != 0:
            self.update_progress(values)

        if "moves" in values and "current" in values and len(values["moves"]) > 1:
            self.visit_plot.update(values, values["max"])

        if values["status"] == "done":
            self.get_control(ct.K_SPINNER).Update(visible=False)
            if not self.event.is_set() or self.event.get_context() == ct.ACCEPT_EVENT:
                # bot has not been cancelled (but is finished or accepted)
                self.execute_move(values["moves"][0])
                self.update_after_move()
            else:
                # clear progress controls
                self.update_progress()
                # switch off auto move
                if self.get_current(ct.K_AUTO_MOVE):
                    self.set_current(ct.K_AUTO_MOVE, False)
                    self.get_control(
                        ct.K_AUTO_MOVE, self.game.turn_to_player()).Update(False)

    def handle_accept_and_cancel(self, event):
        if event == ct.B_ACCEPT:
            self.handle_accept_bot()
        elif event == ct.B_CANCEL:
            self.handle_cancel_bot()
        elif event in [ct.K_BOARD[1], ct.B_UNDO, ct.B_RESIGN, ct.B_RESET, ct.B_BOT_MOVE]:
            popup_bot_in_progress()

    def thread_is_alive(self):
        return hasattr(self, 'thread') and self.thread is not None and self.thread.is_alive()

    def game_over(self, display_message=True):
        if self.game.just_won():
            if display_message:
                lt.popup('Game over: ' + self.stgs.get_setting(
                    ct.K_NAME[3 - self.game.turn_to_player()]) + ' has won!')
            return True

        elif self.game.result == twixt.RESIGN:
            if display_message:
                lt.popup('Game over: ' +
                         self.get_current(ct.K_NAME) + ' has resigned!')
            return True

        return False

    def execute_move(self, move):
        if move == twixt.RESIGN:
            self.game.result = twixt.RESIGN
            self.game_over()
            return
        elif move == twixt.SWAP:
            self.game.play_swap()
        else:
            self.game.play(move)
        self.board.create_move_objects(len(self.game.history) - 1)
        self.game_over()

    def create_settings_window(self):

        sd = lt.SettingsDialogLayout()
        layout = sd.get_layout()
        settings_window = sg.Window(ct.SETTINGS_DIALOG_TITLE, layout, keep_on_top=True,
                                    finalize=True, margins=(15, 15))

        self.stgs.update_window(settings_window)

        return settings_window

    def settings_dialog(self):
        dialog = self.create_settings_window()
        while True:
            event, values = dialog.read()
            if event == sg.WIN_CLOSED or event == ct.B_EXIT:
                break
            elif event == ct.B_RESET_DEFAULT:
                self.stgs.reset_to_default()
                self.stgs.update_window(dialog)
            elif event == ct.B_APPLY_SAVE:
                self.stgs.save(values)
                break
        dialog.close()
        return event

    def create_about_window(self):

        ad = lt.AboutDialogLayout()
        layout = ad.get_layout()
        about_window = sg.Window(ct.ABOUT_DIALOG_TITLE, layout, keep_on_top=True,
                                 finalize=True, margins=(15, 15))
        return about_window

    def about_dialog(self):
        dialog = self.create_about_window()
        while True:
            event, values = dialog.read()
            if event == sg.WIN_CLOSED or event == ct.B_EXIT:
                break
        dialog.close()

    def get_event(self):
        if self.thread_is_alive():
            self.get_control(ct.K_SPINNER).UpdateAnimation(ct.SPINNER_IMAGE)
            # frequent read to update progress gif
            return self.window.read(timeout=200)
        else:
            # blocking read when no bot is processing
            return self.window.read()

    def handle_event(self, evet, values):
        if event == ct.ITEM_SETTINGS:
            if self.settings_dialog() == ct.B_APPLY_SAVE:
                self.update_settings_changed()
        elif event == ct.ITEM_ABOUT:
            self.about_dialog()
        elif event == ct.ITEM_OPEN_FILE:
            self.handle_open_file()
        elif st.key_like(event,  ['AUTO_MOVE', 'TRIALS']):
            # handle trials sliders and auto-move check boxes
            self.stgs.update(event, values)
            self.update_bots()
        elif event == ct.K_THREAD[1]:
            # handle event sent from bot
            self.handle_thread_event(values[ct.K_THREAD[1]])

        if self.thread_is_alive():
            # while bot is processing: handle Accept and Cancel
            self.handle_accept_and_cancel(event)
        else:
            # unless bot is processing: handle click on board and buttons other
            # than Accept, Cancel
            if event == ct.K_BOARD[1]:
                self.handle_board_click(values)
                # self.update_after_move()
            elif event == ct.B_HEATMAP:
                self.handle_heatmap()
            elif event == ct.B_BOT_MOVE:
                if not self.game_over():
                    # clear move statistics
                    self.visit_plot.update()
                    self.update_progress()
                    self.launch_bot()
            elif event == ct.B_UNDO:
                self.handle_undo()
                self.update_after_move()
            elif event == ct.B_RESIGN:
                self.handle_resign()
                self.update_turn_indicators()
            elif event == ct.B_RESET:
                self.reset_game()
                self.update_after_move()


# initialize settings from config.json
stgs = st.Settings()

# initialize game, pass "allow self crossing links" setting
game = twixt.Game(stgs.get_setting(ct.K_ALLOW_SCL[1]))

# initialize twixt board (draw it later)
board = uiboard.UiBoard(game, stgs)

# initialize ui
ui = TwixtbotUI(game, stgs, board)


# Event Loop
while True:
    if not ui.game_over(False) and ui.get_current(ct.K_AUTO_MOVE):
        # auto move case
        if not ui.thread_is_alive():
            ui.update_progress()
            ui.launch_bot()

    event, values = ui.get_event()

    if event == "__TIMEOUT__":
        continue
    elif event == sg.WIN_CLOSED or event == ct.B_EXIT:
        # exiting or closed
        break

    ui.handle_event(event, values)


del ui
