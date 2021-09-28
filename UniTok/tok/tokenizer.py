import abc

from .tok import BaseTok


class Tokenizer(abc.ABC):
    def __init__(self, tok: BaseTok):
        self.tok = tok

    def tokenize(self, obj):
        raise NotImplementedError


class SingTokenizer(Tokenizer):
    def tokenize(self, obj):
        return self.tok(obj)


class ListTokenizer(Tokenizer):
    def __init__(self,
                 tok: BaseTok,
                 max_length: int = 0,
                 padding: bool = False,
                 ):
        super(ListTokenizer, self).__init__(tok=tok)
        self.max_length = max_length
        self.padding = padding

    def tokenize(self, obj):
        ids = self.tok(obj)  # type: list
        if self.max_length > 0:
            ids = ids[:self.max_length]
            if self.padding:
                ids.extend([self.tok.PAD] * (self.max_length - len(ids)))
        return ids


T = Tokenizer
SingT = SingTokenizer
ListT = ListTokenizer
