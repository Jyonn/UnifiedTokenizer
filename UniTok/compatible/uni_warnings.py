import warnings
from typing import Callable

warned_flags = set()


class UniWarning:
    def __init__(self, msg, type_: Callable = warnings.warn):
        self.msg = msg
        self.type = type_

    def __call__(self):
        if self not in warned_flags:
            warned_flags.add(self)
            self.type(self.msg)


VocabMapDeprecationWarning = UniWarning(
    'index2obj and obj2index are deprecated, use i2o and o2i instead', type_=DeprecationWarning)
