from unitok.tokenizer import BaseTokenizer


class UnknownTokenizer(BaseTokenizer):
    param_list = []
    return_list = False

    prefix = 'unk_'

    def __init__(self, classname, **kwargs):
        super().__init__(**kwargs)

        kwargs.pop('vocab')
        kwargs.pop('tokenizer_id')

        self.kwargs = kwargs
        self.classname = classname

    def __call__(self, obj):
        raise NotImplementedError('UnknownTokenizer is used as a placeholder and should not be called.')

    def json(self):
        return {
            'tokenizer_id': self.get_tokenizer_id(),
            'vocab': self.vocab.name,
            'classname': self.classname,
            'params': self.kwargs,
        }

    def __str__(self):
        return f'{self._detailed_classname}({self.get_tokenizer_id()}, vocab={self.vocab.name})'

    @property
    def _detailed_classname(self):
        return f'{self.__class__.__name__}[{self.classname}]'
