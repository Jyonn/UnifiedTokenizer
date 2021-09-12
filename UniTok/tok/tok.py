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
