from typing import Dict

from smartify import E

from .vocab import Vocab


class VocabDepot:
    def __init__(self):
        self.depot = {}  # type: Dict[str, Vocab]
        self.col_map = {}

    def append(self, col_or_vocab):
        if isinstance(col_or_vocab, Vocab):
            vocab = col_or_vocab
        else:
            col = col_or_vocab
            vocab = col_or_vocab.tok.vocab
            if vocab.name in self.col_map:
                self.col_map[vocab.name].append(col.name)
            else:
                self.col_map[vocab.name] = [col.name]

        assert vocab.name is not None
        if vocab.name in self.depot and self.depot[vocab.name] != vocab:
            raise f'Conflict name: {vocab.name}'
        self.depot[vocab.name] = vocab

    def get_vocab(self, name) -> Vocab:
        if name in self.depot:
            return self.depot[name]
        raise f'Not found vocab {name}'

    def __call__(self, name) -> Vocab:
        return self.get_vocab(name)
