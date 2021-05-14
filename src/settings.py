import PySimpleGUI as sg
from json import (load as jsonload, dump as jsondump)
import pathlib
import logging

import constants as ct


class Settings():
    def __init__(self):
        self.logger = logging.getLogger(ct.LOGGER)
        self.load()

    def get(self, key):
        return self.settings[key]

    def set(self, key, value):
        self.settings[key] = value

    def get_current(self, key, game):
        return self.settings[key[game.turn_to_player()]]

    def set_current(self, key, value, game):
        self.settings[key[game.turn_to_player()]] = value

    def update(self, event, values):
        if event in [ct.K_TRIALS[1], ct.K_TRIALS[2]]:
            self.settings[event] = int(values[event])
        else:
            self.settings[event] = values[event]

    def same_models(self):
        return pathlib.Path(self.get(ct.K_MODEL_FOLDER[1])).absolute() == \
            pathlib.Path(self.get(ct.K_MODEL_FOLDER[2])).absolute()

    def load(self):
        try:
            with open(ct.SETTINGS_FILE, 'r') as f:
                self.settings = jsonload(f)
                # backward compatibility: set board_size to new min 500
                self.settings[ct.K_BOARD_SIZE[1]] = max(
                    min(ct.BOARD_SIZE_LIST), self.settings[ct.K_BOARD_SIZE[1]])
        except Exception:
            sg.popup(ct.MSG_NO_CONFIG_FILE, keep_on_top=True)
            self.settings = {}

        # set defaults for settings that haven't been found in config file
        changes = False
        for key in ct.SETTING_KEYS:
            # general and player 1 defaults
            if key[1] not in self.settings:
                self.settings[key[1]] = key[3]
                changes = True
            if len(key) == 5 and key[2] not in self.settings:
                # player 2 defaults
                self.settings[key[2]] = key[4]
                changes = True

        if changes:
            self.save()

    def save(self, values=None):
        old_board_size = None
        if values is not None:
            for key in ct.SETTING_KEYS:
                # update self.settings with values read from settings file
                try:
                    if key[1] == ct.K_BOARD_SIZE[1]:
                        # keep current board_size until restart
                        old_board_size = self.settings[key[1]]
                    self.settings[key[1]] = values[key[1]]
                    if len(key) == 5:
                        # player 2
                        self.settings[key[2]] = values[key[2]]
                except Exception as e:
                    self.logger.error(ct.MSG_ERROR_UPDATING_KEY, str(key), str(e))

        with open(ct.SETTINGS_FILE, 'w') as f:
            jsondump(self.settings, f, indent=4, sort_keys=True)

        self.settings[ct.K_BOARD_SIZE[1]
                      ] = old_board_size or self.settings[ct.K_BOARD_SIZE[1]]

    def reset_to_default(self, window):
        for key in ct.SETTING_KEYS:  # update all settings with defaults
            try:
                k = key[1]
                # self.settings[k] = key[3]
                window[k].update(value=key[3])
                if len(key) == 5:
                    # player 2
                    k = key[2]
                    # self.settings[k] = key[4]
                    window[k].update(value=key[4])
            except Exception as e:
                self.logger.error(ct.MSG_ERROR_UPDATING_KEY, str(k), str(e))

    def update_window(self, window):
        for key in ct.SETTING_KEYS:
            k = key[1]
            try:
                window[k].update(value=self.get(k))
                if len(key) > 4:
                    # player2
                    k = key[2]
                    window[k].update(value=self.get(k))
            except Exception as e:
                self.logger.error(ct.MSG_ERROR_UPDATING_KEY, str(k), str(e))

    def get_tooltip(self, player):
        # show settings on mouse over auto-move check box
        text = ct.K_ALLOW_SWAP[0] + ":\t" + \
            str(self.get(ct.K_ALLOW_SWAP[1])) + "   \n"
        text += "allow scl" + ":\t" + \
            str(self.get(ct.K_ALLOW_SCL[1])) + "   \n"
        text += ct.K_SMART_ACCEPT[0] + ":\t" + \
            str(self.get(ct.K_SMART_ACCEPT[1])) + "   \n"
        text += "----  evaluation  ------------------\n"
        text += ct.K_MODEL_FOLDER[0] + ":\t" + \
            str(self.get(ct.K_MODEL_FOLDER[player])) + "   \n"
        text += ct.K_RANDOM_ROTATION[0] + ":\t" + \
            str(self.get(ct.K_RANDOM_ROTATION[player])) + "   \n"
        text += "----  MCTS  ------------------------\n"
        text += ct.K_SMART_ROOT[0] + ":\t" + \
            str(self.get(ct.K_SMART_ROOT[player])) + "   \n"
        text += ct.K_TEMPERATURE[0] + ":\t" + \
            str(self.get(ct.K_TEMPERATURE[player])) + "   \n"
        text += ct.K_ADD_NOISE[0] + ":\t" + \
            str(self.get(ct.K_ADD_NOISE[player])) + "   \n"
        text += ct.K_CPUCT[0] + ":\t\t" + \
            str(self.get(ct.K_CPUCT[player])) + "   "
        return text
