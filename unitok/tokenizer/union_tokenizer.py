from unitok.tokenizer import BaseTokenizer


class UnionTokenizer(BaseTokenizer):
    param_list = []
    return_list = False

    prefix = 'union_'

    def __init__(self, tokenizer: BaseTokenizer, **kwargs):
        super().__init__(vocab=tokenizer.vocab, **kwargs)
        self.tokenizer = tokenizer
        self.classname = self.tokenizer._detailed_classname

    @property
    def _detailed_classname(self):
        return f'{self.__class__.__name__}[{self.classname}]'
