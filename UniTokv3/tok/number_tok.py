from typing import Iterable

from .tok import BaseTok


class NumberTok(BaseTok):
    """
    Number Tokenizer

    Args:
        vocab_size: the maximum number of the vocabulary, if None, the vocabulary will be extended automatically
        ...: other arguments of the BaseTok

    Example:
        >>> from UniTokv3.tok.number_tok import NumberTok
        >>> tok = NumberTok()
        >>> tok(123)  # 123
        >>> tok(456)  # 456
        >>> tok = NumberTok(vocab_size=100)
        >>> tok(60)  # 60
        >>> tok(200)  # ValueError: vocab_size is 100, but 200 is given
    """
    return_list = BaseTok.Alternative

    def __init__(self, vocab_size=None, **kwargs):
        super(NumberTok, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        if vocab_size is not None:
            self.vocab.extend([str(i) for i in range(vocab_size)])

    def t(self, obj):
        # check is iterable
        if isinstance(obj, Iterable) and not isinstance(obj, str):
            obj = [int(o) for o in obj]
        else:
            obj = int(obj)
        objs = [obj] if isinstance(obj, int) else obj
        for o in objs:
            if o >= len(self.vocab):
                if self.vocab_size is not None:
                    raise ValueError('vocab_size is {}, but {} is given'.format(self.vocab_size, o))
                self.vocab.extend([str(i) for i in range(len(self.vocab), o + 1)], count=False)
        return obj
