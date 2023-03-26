from UniTok.tok import BaseTok


class NumberTok(BaseTok):
    """
    Number Tokenizer

    Args:
        vocab_size: the maximum number of the vocabulary, if None, the vocabulary will be extended automatically
        ...: other arguments of the BaseTok

    Example:
        >>> from UniTok.tok.number_tok import NumberTok
        >>> tok = NumberTok()
        >>> tok(123)  # 123
        >>> tok(456)  # 456
        >>> tok = NumberTok(vocab_size=100)
        >>> tok(60)  # 60
        >>> tok(200)  # ValueError: vocab_size is 100, but 200 is given
    """
    def __init__(self, vocab_size=None, **kwargs):
        super(NumberTok, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        if vocab_size is not None:
            self.vocab.extend([str(i) for i in range(vocab_size)])

    def t(self, obj):
        obj = int(obj)
        if obj >= len(self.vocab):
            if self.vocab_size is not None:
                raise ValueError('vocab_size is {}, but {} is given'.format(self.vocab_size, obj))
            self.vocab.extend([str(i) for i in range(len(self.vocab), obj + 1)])
        return obj
