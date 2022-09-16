
import numpy as np

import sys
import os

np.set_printoptions(threshold=np.inf)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
test = np.load('E:/GWJ/MLMAN2/MLMAN-master/_processed_data/train_pos1.npy')
type = sys.getfilesystemencoding()
sys.stdout = Logger('E:/GWJ/MLMAN2/4.text')

print(path)
print(test)
print(os.path.dirname(__file__))
print('---------------------------------------')
