import pandas as pd

from .tok import BaseTok


class SplitTok(BaseTok):
    """
    Split Tokenizer

    Args:
        sep: separator of the string
        ...: other arguments of the BaseTok

    Example:
        >>> from UniTok.tok.split_tok import SplitTok
        >>> tok = SplitTok(sep=' ')
        >>> tok('Hello world')  # [0, 1]
        >>> tok('Debug the world')  # [2, 3, 1]
    """
    return_list = True

    def __init__(self, sep, **kwargs):
        super(SplitTok, self).__init__(**kwargs)
        self.sep = sep

    def t(self, obj):
        ids = []
        if pd.notnull(obj):
            ts = obj.split(self.sep)
            for t in ts:
                ids.append(self.insert(t))
        return ids
