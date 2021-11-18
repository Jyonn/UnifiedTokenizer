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
                 slice_post: bool = False,
                 pad_pre: bool = False,
                 ):
        super(ListTokenizer, self).__init__(tok=tok)
        self.max_length = max_length
        self.slice_post = slice_post
        self.padding = padding
        self.pad_pre = pad_pre

    def tokenize(self, obj):
        ids = self.tok(obj)  # type: list
        if self.max_length > 0:
            ids = ids[-self.max_length:] if self.slice_post else ids[:self.max_length]
            if self.padding:
                pads = [self.tok.PAD] * (self.max_length - len(ids))
                if self.pad_pre:
                    ids = pads + ids
                else:
                    ids.extend(pads)
        return ids


T = Tokenizer
SingT = SingTokenizer
ListT = ListTokenizer
