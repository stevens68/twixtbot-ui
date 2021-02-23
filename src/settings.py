import PySimpleGUI as sg
from json import (load as jsonload, dump as jsondump)
import pathlib

import constants as ct


def key_like(key, key_list):
    for k in key_list:
        if k in key:
            return True
    return False


class Settings():

    def __init__(self):
        self.load()

    def get_setting(self, key):
        return self.settings[key]

    def get_current(self, key, game):
        return self.settings[key[game.turn_to_player()]]

    def set_current(self, key, value, game):
        self.settings[key[game.turn_to_player()]] = value

    def update(self, event, values):
        if key_like(event, ['TRIALS']):
            self.settings[event] = int(values[event])
        else:
            self.settings[event] = values[event]

    def same_models(self):
        return pathlib.Path(self.get_setting(ct.K_MODEL_FOLDER[1])).absolute() == pathlib.Path(self.get_setting(ct.K_MODEL_FOLDER[2])).absolute()

    def load(self):
        try:
            with open(ct.SETTINGS_FILE, 'r') as f:
                self.settings = jsonload(f)
        except Exception:
            sg.popup(ct.MSG_NO_CONFIG_FILE, keep_on_top=True)

            self.settings = {}
            for key in ct.SETTING_KEYS:
                # player 1 defaults
                self.settings[key[1]] = key[3]
                if len(key) == 5:
                    # player 2 defaults
                    self.settings[key[2]] = key[4]

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
                    print(ct.MSG_ERROR_UPDATING_KEY + str(key) + " " + str(e))

        with open(ct.SETTINGS_FILE, 'w') as f:
            jsondump(self.settings, f, indent=4, sort_keys=True)

        self.settings[ct.K_BOARD_SIZE[1]
                      ] = old_board_size or self.settings[ct.K_BOARD_SIZE[1]]

    def reset_to_default(self):
        for key in ct.SETTING_KEYS:  # update all settings with defaults
            try:
                k = key[1]
                self.settings[k] = key[3]
                if len(key) == 5:
                    # player 2
                    k = key[2]
                    self.settings[k] = key[4]
            except Exception as e:
                print(ct.MSG_ERROR_UPDATING_KEY + str(k) + " " + str(e))

    def update_window(self, window):
        for key in ct.SETTING_KEYS:
            k = key[1]
            try:
                window[k].update(value=self.get_setting(k))
                if len(key) > 4:
                    # player2
                    k = key[2]
                    window[k].update(value=self.get_setting(k))
            except Exception as e:
                print(ct.MSG_ERROR_UPDATING_KEY + k + ": " + str(e))

    def get_tooltip(self, player):
        # show settings on mouse over auto-move check box
        text = ct.K_ALLOW_SWAP[0] + ":\t" + \
            str(self.get_setting(ct.K_ALLOW_SWAP[1])) + "   \n"
        text += "allow scl" + ":\t" + \
            str(self.get_setting(ct.K_ALLOW_SCL[1])) + "   \n"
        text += ct.K_SMART_ACCEPT[0] + ":\t" + \
            str(self.get_setting(ct.K_SMART_ACCEPT[1])) + "   \n"
        text += "----  evaluation  ------------------\n"
        text += ct.K_MODEL_FOLDER[0] + ":\t" + \
            str(self.get_setting(ct.K_MODEL_FOLDER[player])) + "   \n"
        text += ct.K_RANDOM_ROTATION[0] + ":\t" + \
            str(self.get_setting(ct.K_RANDOM_ROTATION[player])) + "   \n"
        text += "----  MCTS  ------------------------\n"
        text += ct.K_TEMPERATURE[0] + ":\t" + \
            str(self.get_setting(ct.K_TEMPERATURE[player])) + "   \n"
        text += ct.K_ADD_NOISE[0] + ":\t" + \
            str(self.get_setting(ct.K_ADD_NOISE[player])) + "   \n"
        text += ct.K_CPUCT[0] + ":\t\t" + \
            str(self.get_setting(ct.K_CPUCT[player])) + "   "
        return text
