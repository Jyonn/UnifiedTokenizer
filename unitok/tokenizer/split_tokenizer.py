from unitok.tokenizer import EntitiesTokenizer


class SplitTokenizer(EntitiesTokenizer):
    param_list = ['sep']

    def __init__(self, sep, **kwargs):
        super().__init__(**kwargs)

        self.sep = sep

    def __call__(self, obj):
        tokens = obj.split(self.sep)
        return super().__call__(tokens)
