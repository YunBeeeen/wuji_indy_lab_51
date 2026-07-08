from pynput import keyboard
from pynput.keyboard import Key

class KeyboardListener:
    def __init__(self, key_targets):
        """
        params:
            frequency: keyboard state update rate
            key_targets: keyboard candidates to save states
        """
        self.key_targets = key_targets
        self.key_states = dict()
        for key_target in key_targets:
            self.key_states[key_target] = False
        self.updated = False

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        key_str = str(key)
        key_str = key_str.strip("''")

        # update key states
        self.updated = True
        for key_target in self.key_targets:
            if key == key_target or key_str == key_target:
                self.key_states[key_target] = True
            else:
                self.key_states[key_target] = False

    def get_key_states(self):
        data = dict()
        data["updated"] = self.updated
        data["value"] = self.key_states
        self.updated = False
        return data

##########################################
import os

def check_dir(folder, generate=True):
    if os.path.isdir(folder):
        return True
    else:
        if generate:
            os.makedirs(folder)
            return True
        else:
            return False


##########################################
import pandas as pd
import numpy as np

class DataLogger:
    def __init__(self, folder: str, file: str, data_types: list):
        assert len(data_types) > 0, "At least one data type required."

        self.folder = folder
        self.file = file
        self.columns = data_types
        self.data = {col: [] for col in ["step", *data_types]}
        self.step = 0

    def append(self, *values):
        """
        Only support "scalar" or "vector" (not "matrix")
        """
        for col, val in zip(self.columns, values):
            self.data[col].append(val)

        self.step += 1
        self.data["step"].append(self.step)

    def save_to_csv(self):
        check_dir(self.folder, generate=True)

        for col in self.columns:
            val = np.array(self.data[col])
            if len(val.shape) == 2:
                dim = val.shape[-1]
                for i in range(dim):
                    self.data[f"{col}_{i}"] = val[:, i]
                del self.data[col]

        df = pd.DataFrame(self.data)
        df.to_csv(f"{self.folder}/{self.file}.csv", index=False)


##########################################
if __name__ == "__main__":
    keyboard_listener = KeyboardListener(key_targets=[Key.left, Key.up, Key.right, Key.down])
    while True:
        keyboard_data = keyboard_listener.get_key_states()
        if keyboard_data["updated"]:
            print(keyboard_data["value"])