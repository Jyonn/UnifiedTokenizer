from UniTok.vocab.vocab import Vocab


class BaseTok:
    def __init__(self, name: str = None, vocab: Vocab = None, pre_handler=None):
        self.PAD = 0
        assert name or vocab, ValueError('name and vocab can not both be null')

        if vocab:
            self.vocab = vocab
        else:
            self.vocab = Vocab(name)
        self.pre_handler = pre_handler

    def t(self, obj) -> [int, list]:
        raise NotImplementedError

    def _t(self, obj):
        ids = self.t(obj)
        if isinstance(ids, list):
            return list(filter(lambda index: index > -1, ids))
        if ids == -1:
            raise ValueError('Single Tokenizer should provide vocab, but -1 is given')
        return ids

    def __call__(self, obj):
        return self._t(obj)

    def as_sing(self):
        from .tokenizer import SingT
        return SingT(self)

    def as_list(self, max_length=0, padding=False, slice_post=False, pad_pre=False):
        from .tokenizer import ListT
        return ListT(self, max_length=max_length, padding=padding, slice_post=slice_post, pad_pre=pad_pre)

    def load_vocab(self, store_dir: str, as_path=False):
        self.vocab.load(store_dir=store_dir, as_path=as_path)
        return self
