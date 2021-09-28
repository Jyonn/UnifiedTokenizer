from .tok import BaseTok
from UniTok.vocab.vocab import Vocab


class EntTok(BaseTok):
    def __init__(self, name, vocab: Vocab = None):
        super(EntTok, self).__init__(name=name, vocab=vocab)

    def t(self, obj):
        return self.vocab.append(str(obj))
