from typing import Optional

from UniTok.vocab.vocab import Vocab


class BaseTok:
    def __init__(self, name: str, vocab: Optional[Vocab] = None):
        self.PAD = 0
        self.name = name
        self.vocab = vocab or Vocab(name)

    def t(self, obj) -> [int, list]:
        raise NotImplementedError

    def __call__(self, obj):
        return self.t(obj)

    def as_sing(self):
        from .tokenizer import SingT
        return SingT(self)

    def as_list(self, max_length=0, padding=False):
        from .tokenizer import ListT
        return ListT(self, max_length=max_length, padding=padding)

    def load_vocab(self, store_dir: str, as_path=False):
        self.vocab.load(store_dir=store_dir, as_path=as_path)
        return self
