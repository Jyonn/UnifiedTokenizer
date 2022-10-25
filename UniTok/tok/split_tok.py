import pandas as pd

from .tok import BaseTok


class SplitTok(BaseTok):
    def __init__(self, sep, **kwargs):
        super(SplitTok, self).__init__(**kwargs)
        self.sep = sep

    def t(self, obj,):
        ids = []
        if pd.notnull(obj):
            ts = obj.split(self.sep)
            for t in ts:
                if self.pre_handler:
                    t = self.pre_handler(t)
                ids.append(self.vocab.append(t))
        return ids
