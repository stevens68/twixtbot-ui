#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import PySimpleGUI as sg
import threading
import time
import logging


class BotEvent(threading.Event):
    # used to stop or cancel a bot (thread)

    def __init__(self):
        super().__init__()
        self.context = None

    def get_context(self):
        return self.context

    def set(self, context):
        super().set()
        self.context = context


class ProgressWindow(sg.Window):

    def __init__(self):

        layout = lt.SplashScreenLayout().get_layout()
        super().__init__('twixtbot-ui', layout,
                         background_color=sg.theme_background_color(), keep_on_top=True,
                         finalize=True, margins=(15, 15))

    def update(self, text, progress):
        self.__getitem__(ct.K_SPLASH_PROGRESS_BAR[0]).UpdateBar(progress, 100)
        self.__getitem__(ct.K_SPLASH_STATUS_TEXT[0]).Update(text)
        self.refresh()


class TwixtbotUI():
    def __init__(self, game, stgs, board):
        # Show splash screen during init

        init_window = ProgressWindow()
        init_window.update('initializing GUI ...', 5)

        # init class properties
        self.board = board
        self.game = game
        self.moves_score = {}
        self.stgs = stgs
        self.bot_event = None
        self.redo_moves = []
        self.logger = logging.getLogger(ct.LOGGER)

        # Setup main GUI window
        layout = lt.MainWindowLayout(board, stgs).get_layout()
        self.window = sg.Window(ct.WINDOW_TITLE,
                                layout,
                                margins=(25, 25),
                                finalize=True)

        canvas = self.window[ct.K_EVAL_MOVES[1]].TKCanvas
        self.eval_moves_plot = pt.ThreeBarPlot(canvas, ct.EVAL_PLOT_COLOR)

        canvas = self.window[ct.K_VISITS[1]].TKCanvas
        self.visit_plot = pt.ThreeBarPlot(canvas, ct.VISITS_PLOT_COLOR)
        self.visit_plot.update()

        canvas = self.window[ct.K_EVAL_HIST[1]].TKCanvas
        self.eval_hist_plot = pt.EvalHistPlot(canvas, stgs)

        def motion(event):
            if stgs.get(ct.K_SHOW_CURSOR_LABEL[1]):
                coords = (event.x, stgs.get(
                    ct.K_BOARD_SIZE[1]) - event.y)
                _, move = board.get_move(coords)
                board.draw_cursor_label(move)

        self.window["BOARD"].TKCanvas.bind('<Motion>', motion)

        self.window.bind('<Alt-b>', ct.B_BOT_MOVE)
        self.window.bind('<Alt-a>', ct.B_ACCEPT)
        self.window.bind('<Alt-c>', ct.B_CANCEL)
        self.window.bind('<Alt-u>', ct.B_UNDO)
        self.window.bind('<Alt-d>', ct.B_REDO)
        self.window.bind('<Alt-g>', ct.B_RESIGN)
        self.window.bind('<Alt-r>', ct.B_RESET)
        self.window.bind('<Alt-e>', ct.EVENT_SHORTCUT_SHOW_EVALUATION)
        self.window.bind('<Alt-m>', ct.EVENT_SHORTCUT_HEATMAP)
        self.window.bind('<Alt-v>', ct.EVENT_SHORTCUT_VISUALIZE_MCTS)
        self.window.bind('<Alt-KeyPress-1>', ct.EVENT_SHORTCUT_AUTOMOVE_1)
        self.window.bind('<Alt-KeyPress-2>', ct.EVENT_SHORTCUT_AUTOMOVE_2)
        self.window.bind('<Alt-Right->', ct.EVENT_SHORTCUT_TRIALS_1_PLUS)
        self.window.bind('<Alt-Left->', ct.EVENT_SHORTCUT_TRIALS_1_MINUS)
        self.window.bind('<Alt-Shift-Right->', ct.EVENT_SHORTCUT_TRIALS_2_PLUS)
        self.window.bind('<Alt-Shift-Left->',
                         ct.EVENT_SHORTCUT_TRIALS_2_MINUS)

        # Apply settings
        init_window.update('refreshing settings ...', 10)
        self.update_settings_changed()

        # import
        init_window.update('importing modules ...', 30)
        import backend.nnmplayer as nnmplayer

        # Initialize and warm-up bots
        self.bots = [None, None]
        
        if self.stgs.same_models():
            init_window.update('initializing bots ...', 60)
            self.init_bot(1) # init bot[1]
            self.init_bot(2, self.bots[1].evaluator)
        else:
            init_window.update('initializing bot 1 ...', 50)
            self.init_bot(1)
            init_window.update('initializing bot 2 ...', 70)
            self.init_bot(2)

        init_window.update('warming up bots ...', 90)
        self.bots[0].nm.eval_game(self.game)
        self.bots[1].nm.eval_game(self.game)

        # Update evaluation graph
        self.update_evals()

        # Close and destroy splash window
        init_window.update('ready to play', 100)
        time.sleep(1)
        init_window.close()

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

    def clear_evals(self):
        self.get_control(ct.K_EVAL_NUM).Update('')
        self.get_control(ct.K_EVAL_BAR).Update(0)
        self.eval_moves_plot.update()
        self.eval_hist_plot.update()
        self.visit_plot.update()

    def update_evals(self):
        if not self.get_control(ct.K_SHOW_EVALUATION).get():
            self.clear_evals()
            self.next_move = None
            return

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
        s.configure(ebs, background=self.stgs.get(ct.K_COLOR[1]))
        s.configure(ebs, troughcolor=self.stgs.get(ct.K_COLOR[2]))

    def update_progress(self, values=None):
        if values is None:
            text = ""
            value = 0
            max_value = 0
        else:
            max_value = values["max"]
            value = values["current"]

            if self.stgs.get(ct.K_SMART_ACCEPT[1]) and "Y" in values:
                diff = values["Y"][0] - values["Y"][1]
                if diff > max_value - value:
                    # 2nd best cannot catch up => accept (if not already
                    # cancelled)
                    if self.bot_event.get_context() != ct.CANCEL_EVENT:
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

    def update_after_move(self, complete=True):
        if self.get_control(ct.K_HEATMAP).get():
            heatmap = hm.Heatmap(self.game, self.bots[self.game.turn])
        else:
            heatmap = None

        self.board.draw(heatmap, complete)

        self.window.refresh()

        # reset progress
        self.update_progress()

        self.update_turn_indicators()
        self.update_history()

        if self.get_control(ct.K_SHOW_EVALUATION).get():
            self.update_evals()

    def update_settings_changed(self):
        self.board.draw()
        self.window.refresh()
        # update ui
        for p in [1, 2]:
            self.get_control(ct.K_NAME, p).Update(
                self.stgs.get(ct.K_NAME[p]))
            self.get_control(ct.K_COLOR, p).erase()
            self.get_control(ct.K_COLOR, p).DrawCircle((7, 9), 6,
                                                       self.stgs.get(
                                                           ct.K_COLOR[p]),
                                                       self.stgs.get(ct.K_COLOR[p]))
            self.get_control(ct.K_AUTO_MOVE, p).Update(
                self.stgs.get(ct.K_AUTO_MOVE[p]))
            self.get_control(ct.K_TRIALS, p).Update(
                self.stgs.get(ct.K_TRIALS[p]))

        self.update_turn_indicators()
        self.update_tooltips()
        self.update_evalbar_colors()
        self.eval_hist_plot.update(self.moves_score)
        self.update_bots()
        self.update_game()
        self.update_logger()

    def reset_game(self):
        self.game.__init__(self.stgs.get(ct.K_ALLOW_SCL[1]))
        self.moves_score = {}
        # get eval of empty board to avoid gap at x=0 in plot in loaded games
        self.calc_eval()

    def update_game(self):
        self.game.allow_scl = self.stgs.get(ct.K_ALLOW_SCL[1])

    def update_logger(self):
        self.logger.setLevel(self.stgs.get(ct.K_LOG_LEVEL[1]))

    # bot functions

    def update_bots(self):
        for t in [0, 1]:
            if hasattr(self, 'bots') and self.bots[t] is not None:
                p = self.game.turn_to_player(t)
                self.bots[t].allow_swap = self.stgs.get(
                    ct.K_ALLOW_SWAP[1])
                self.bots[t].num_trials = int(
                    self.stgs.get(ct.K_TRIALS[p]))
                self.bots[t].temperature = float(
                    self.stgs.get(ct.K_TEMPERATURE[p]))
                self.bots[t].random_rotation = self.stgs.get(
                    ct.K_RANDOM_ROTATION[p])
                self.bots[t].add_noise = float(
                    self.stgs.get(ct.K_ADD_NOISE[p]))
                self.bots[t].nm.cpuct = float(
                    self.stgs.get(ct.K_CPUCT[p]))
                self.bots[t].nm.visualize_mcts = self.get_control(ct.K_VISUALIZE_MCTS).get()

    def init_bot(self, player, evaluator=None):

        args = {
            "allow_swap": self.stgs.get(ct.K_ALLOW_SWAP[1]),
            "model": self.stgs.get(ct.K_MODEL_FOLDER[player]),
            "trials": self.stgs.get(ct.K_TRIALS[player]),
            "temperature": self.stgs.get(ct.K_TEMPERATURE[player]),
            "random_rotation": self.stgs.get(ct.K_RANDOM_ROTATION[player]),
            "add_noise": self.stgs.get(ct.K_ADD_NOISE[player]),
            "cpuct": self.stgs.get(ct.K_CPUCT[player]),
            "board": self.board,
            "evaluator": evaluator 
        }

        import backend.nnmplayer as nnmplayer
        self.bots[2 - player] = nnmplayer.Player(**args)

    def bot_move(self):
        if len(self.game.history) >= 2 and self.get_current(ct.K_TRIALS) == 0 and self.next_move is not None:
            # we already have the next move from evaluation
            self.bots[self.game.turn].nm.send_message(self.window, self.game, "done",
                                                      0, 0, moves=self.next_move[1], P=self.next_move[2])
        else:
            # mcts, or first/second move
            self.bots[self.game.turn].pick_move(
                self.game, self.window, self.bot_event)

    def launch_bot(self):
        self.visit_plot.update()
        self.window[ct.K_SPINNER[1]].Update(visible=True)
        self.bot_event = BotEvent()
        self.thread = threading.Thread(
            target=self.bot_move, args=(), daemon=True)

        self.timer = pmeter.ETA(self.get_current(ct.K_TRIALS), max_seconds=300)
        self.thread.start()

    # handle events
    def handle_board_click(self, values):
        if self.game_over():
            return
        move, _ = self.board.get_move(values[ct.K_BOARD[1]])
        if move is not None:
            # clear move statistics
            self.execute_move(move)
            self.update_after_move(False)

    def handle_open_file(self):
        players, moves, x_lines = fi.get_game(self.stgs.get(ct.K_ALLOW_SCL[1]))
        if players is None:
            return

        # assign player names
        self.stgs.settings[ct.K_NAME[1]] = players[0]
        self.stgs.settings[ct.K_NAME[2]] = players[1]

        # adjust settings if needed
        if x_lines:
            self.stgs.set(ct.K_ALLOW_SCL[1], True)
        self.update_settings_changed()

        # reset game (which also handles changes crossing lines settings)
        self.reset_game()

        # replay game
        try:
            lt.popup("loading game...")
            for m in moves:
                self.execute_move(m)
                self.calc_eval()
        except Exception:
            lt.popup("invalid move: " + str(m))

        self.update_after_move()

    def handle_save_file(self):
        fi.save_game(
            [self.stgs.settings[ct.K_NAME[p]] for p in (1, 2)],
            self.game.history,
            self.game.SIZE,
            self.game is not None)

    def handle_resign(self):
        if self.game_over():
            return
        self.execute_move(twixt.RESIGN)

    def handle_undo(self):
        if self.game.result == twixt.RESIGN:
            self.game.result = None
            self.redo_moves.append(twixt.RESIGN)
            return

        gl = len(self.game.history)
        if gl in self.moves_score:
            del self.moves_score[gl]

        if gl > 0:
            self.redo_moves.append(self.game.history[-1])
            self.game.undo()

        # switch off auto move
        if self.get_current(ct.K_AUTO_MOVE):
            self.set_current(ct.K_AUTO_MOVE, False)
            self.get_control(
                ct.K_AUTO_MOVE, self.game.turn_to_player()).Update(False)

    def handle_redo(self):
        if len(self.redo_moves) > 0:
            self.execute_move(self.redo_moves.pop(), False)

    def handle_accept_bot(self):
        self.bot_event.set(ct.ACCEPT_EVENT)

    def handle_cancel_bot(self):
        self.bot_event.set(ct.CANCEL_EVENT)
        # switch off auto move
        if self.get_current(ct.K_AUTO_MOVE):
            self.set_current(ct.K_AUTO_MOVE, False)
            self.get_control(
                ct.K_AUTO_MOVE, self.game.turn_to_player()).Update(False)

    def handle_thread_event(self, values):
        self.logger.info("Bot response: %s", values)
        if values["max"] != 0:
            # mcts case
            self.update_progress(values)

        if self.get_control(ct.K_SHOW_EVALUATION).get() and "moves" in values and "current" in values and len(values["moves"]) > 1:
            self.visit_plot.update(values, max(1, values["max"]))

        if values["status"] == "done":
            self.get_control(ct.K_SPINNER).Update(visible=False)
            if not self.bot_event.is_set() or self.bot_event.get_context() == ct.ACCEPT_EVENT:
                # bot has not been cancelled (but is finished or accepted)
                self.execute_move(values["moves"][0])
                self.update_after_move(False)
            else:
                # bot has been cancelled clear progress controls and visits
                self.update_progress()
                # reset history_at_root resets tree and visit counts
                self.bots[self.game.turn].nm.history_at_root = None

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
        elif event in [ct.K_BOARD[1], ct.B_UNDO, ct.B_REDO, ct.B_RESIGN, ct.B_RESET, ct.B_BOT_MOVE, ct.K_VISUALIZE_MCTS[1], ct.K_HEATMAP[1],
                        ct.EVENT_SHORTCUT_VISUALIZE_MCTS, ct.EVENT_SHORTCUT_HEATMAP]:
            lt.popup("bot in progress. Click Accept or Cancel.")
            if event == ct.K_VISUALIZE_MCTS[1]:
                self.get_control(ct.K_VISUALIZE_MCTS).update(not self.get_control(ct.K_VISUALIZE_MCTS).get())
            elif event == ct.K_HEATMAP[1]:
                self.get_control(ct.K_HEATMAP).update(not self.get_control(ct.K_HEATMAP).get())
 
    def thread_is_alive(self):
        return hasattr(self, 'thread') and self.thread is not None and self.thread.is_alive()

    def game_over(self, display_message=True):
        if self.game.just_won():
            if display_message:
                lt.popup('Game over: ' + self.stgs.get(
                    ct.K_NAME[3 - self.game.turn_to_player()]) + ' has won!')
            return True

        elif self.game.result == twixt.RESIGN:
            if display_message:
                lt.popup('Game over: ' +
                         self.get_current(ct.K_NAME) + ' has resigned!')
            return True

        return False

    def execute_move(self, move, clear_redo_moves=True):
        if clear_redo_moves:
            self.redo_moves = []

        if move == twixt.RESIGN:
            self.game.result = twixt.RESIGN
            self.game_over()
            return
        elif move == twixt.SWAP:
            self.game.play_swap()
        else:
            self.game.play(move)
        #self.board.create_move_objects(len(self.game.history) - 1)
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

    def handle_menue_event(self, event, values):
        if event == ct.ITEM_SETTINGS.replace('&', ''):
            if self.settings_dialog() == ct.B_APPLY_SAVE:
                self.update_settings_changed()
            return True

        if event == ct.ITEM_ABOUT.replace('&', ''):
            self.about_dialog()
            return True

        if event == ct.ITEM_OPEN_FILE.replace('&', ''):
            self.handle_open_file()
            return True

        if event == ct.ITEM_SAVE_FILE.replace('&', ''):
            self.handle_save_file()
            return True

        return False

    def handle_button_event(self, event, values):
        if event == ct.B_BOT_MOVE:
            if not self.game_over():
                # clear move statistics
                self.visit_plot.update()
                self.update_progress()
                self.launch_bot()
            return True

        if event == ct.B_UNDO:
            self.handle_undo()
            self.update_after_move()
            return True

        if event == ct.B_REDO:
            self.handle_redo()
            self.update_after_move()
            return True

        if event == ct.B_RESIGN:
            self.handle_resign()
            self.update_turn_indicators()
            return True

        if event == ct.B_RESET:
            self.reset_game()
            self.update_after_move()
            return True

        return False

    def handle_shortcut_event(self, event, values):
        if event == ct.EVENT_SHORTCUT_HEATMAP:
            # toggle heatmap checkbox and redraw board
            self.get_control(ct.K_HEATMAP).Update(
                not self.get_control(ct.K_HEATMAP).get())
            self.update_after_move()
            return True

        if event == ct.EVENT_SHORTCUT_SHOW_EVALUATION:
            # toggle evaluation checkbox and redraw board
            self.get_control(ct.K_SHOW_EVALUATION).Update(
                not self.get_control(ct.K_SHOW_EVALUATION).get())
            self.update_after_move()
            return True

        if event == ct.EVENT_SHORTCUT_VISUALIZE_MCTS:
            # toggle visualize checkbox and redraw board
            self.get_control(ct.K_VISUALIZE_MCTS).Update(
                not self.get_control(ct.K_VISUALIZE_MCTS).get())
            self.update_after_move()
            return True

        if event == ct.EVENT_SHORTCUT_AUTOMOVE_1:
            check = self.get_control(ct.K_AUTO_MOVE, 1).get()
            self.get_control(ct.K_AUTO_MOVE, 1).Update(not check)
            self.stgs.set(ct.K_AUTO_MOVE[1], not check)
            return True

        if event == ct.EVENT_SHORTCUT_AUTOMOVE_2:
            check = self.get_control(ct.K_AUTO_MOVE, 2).get()
            self.get_control(ct.K_AUTO_MOVE, 2).Update(not check)
            self.stgs.set(ct.K_AUTO_MOVE[2], not check)
            return True

        def update_slider(player, func, limit, factor):
            trials_new = func(self.get_control(
                ct.K_TRIALS, player).Widget.get() + factor * ct.TRIALS_RESOLUTION, limit)
            self.stgs.set(ct.K_TRIALS[player], trials_new)
            self.get_control(ct.K_TRIALS, player).Update(trials_new)
            self.update_bots()
            return True

        if event == ct.EVENT_SHORTCUT_TRIALS_1_PLUS:
            return update_slider(1, min, ct.TRIALS_MAX, 1)
        if event == ct.EVENT_SHORTCUT_TRIALS_1_MINUS:
            return update_slider(1, max, 0, -1)
        if event == ct.EVENT_SHORTCUT_TRIALS_2_PLUS:
            return update_slider(2, min, ct.TRIALS_MAX, 1)
        if event == ct.EVENT_SHORTCUT_TRIALS_2_MINUS:
            return update_slider(2, max, 0, -1)

        return False

    def handle_event(self, event, values):

        # menue events
        if self.handle_menue_event(event, values):
            return


        # click on auto move or trials (no shortcuts)
        if event in [ct.K_AUTO_MOVE[1], ct.K_AUTO_MOVE[2], ct.K_TRIALS[1], ct.K_TRIALS[2]]:
            # handle trials sliders, auto-move check and heatmap boxes
            self.stgs.update(event, values)
            self.update_bots()
            return


        # click on evaluation checkbox (no shortcuts)
        if event == ct.K_SHOW_EVALUATION[1]:
            self.update_evals()
            return


        # thread events
        if event == ct.K_THREAD[1]:
            # handle event sent from bot
            self.handle_thread_event(values[ct.K_THREAD[1]])
            return

        # button events while bot is processing (Accept, Cancel)
        if self.thread_is_alive():
            self.handle_accept_and_cancel(event)
            return

        # keyboard shortcurt event (buttons and control bar)
        if self.handle_shortcut_event(event, values):
            return


        # button events while bot is not processing
        if self.handle_button_event(event, values):
            return

        # selection of mcts visualization
        if event == ct.K_VISUALIZE_MCTS[1]:
            self.update_bots()
            return

        # click on heatmap (no shortcuts)
        if event == ct.K_HEATMAP[1]:
            self.update_after_move()
            return
 
         # click on board event
        if event == ct.K_BOARD[1]:
            self.handle_board_click(values)
            return

        # other events go here...


def main():
    # initialize settings from config.json
    stgs = st.Settings()

    # Init logging
    logging.basicConfig(format=ct.LOG_FORMAT,
                       level=stgs.get(ct.K_LOG_LEVEL[1]))
    # logger = logging.getLogger(ct.LOGGER)

    # initialize game, pass "allow self crossing links" setting
    game = twixt.Game(stgs.get(ct.K_ALLOW_SCL[1]))

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


if __name__ == "__main__":
    main()
