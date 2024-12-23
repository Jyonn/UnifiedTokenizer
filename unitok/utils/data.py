from typing import Dict, List

from unitok.vocabulary import Vocabulary


class Data:
    def __init__(self, data: Dict[List], key: str, vocab: Vocabulary):
        self.data = data
        self.key = key
        self.vocab = vocab

    def combine(self, other: 'Data'):
        assert self.key == other.key, f'key mismatch: {self.key} != {other.key}'


