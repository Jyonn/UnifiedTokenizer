from unitok.tokenizer import BaseTokenizer


class DigitTokenizer(BaseTokenizer):
    return_list = False
    name = 'digit'
    param_list = ['vocab_size']

    def __init__(self, vocab_size: int = None, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        if self.vocab_size is not None:
            self.vocab.extend([str(i) for i in range(vocab_size)])
            self.vocab.deny_edit()

    def __call__(self, obj):
        obj = int(obj)
        if obj >= len(self.vocab):
            if self.vocab_size is not None:
                raise ValueError(f'Vocabulary size is limited to {self.vocab_size}, but {obj} is given')
            self.vocab.extend([str(i) for i in range(len(self.vocab), obj + 1)])
        return obj


class DigitsTokenizer(DigitTokenizer):
    return_list = True
    name = 'digits'

    def __call__(self, obj):
        obj = [int(o) for o in obj]
        for o in obj:
            super().__call__(o)
        return obj
