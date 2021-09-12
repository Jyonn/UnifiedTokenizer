import pandas as pd

from .tok import BaseTok
from UniTok.vocab.vocab import Vocab


class SplitTok(BaseTok):
    def __init__(self, name, sep, vocab: Vocab = None):
        super(SplitTok, self).__init__(name=name, vocab=vocab)
        self.sep = sep

    def t(self, obj,):
        ids = []
        if pd.notnull(obj):
            ts = obj.split(self.sep)
            for t in ts:
                ids.append(self.vocab.append(t))
        return ids
