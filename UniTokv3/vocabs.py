import warnings

from .meta import Col, Voc

from .vocab import Vocab


class Vocabs(dict):
    def __init__(self):
        super().__init__()
        self.cols = {}

    @property
    def col_map(self):
        warnings.warn('vocab_depot.col_map is deprecated, '
                      'use vocabs.cols instead (will be removed in 4.x version)', DeprecationWarning)
        return self.cols

    @property
    def depots(self):
        warnings.warn('vocab_depot.depots is deprecated, '
                      'use vocabs instead (will be removed in 4.x version)', DeprecationWarning)
        return self

    def append(self, col_or_vocab):
        if isinstance(col_or_vocab, Vocab):
            vocab = col_or_vocab
        else:
            col = col_or_vocab
            vocab = col_or_vocab.tok.vocab
            if vocab.name in self.cols:
                self.cols[vocab.name].append(col.name)
            else:
                self.cols[vocab.name] = [col.name]

        assert vocab.name is not None
        if vocab.name in self and self[vocab.name] != vocab:
            raise ValueError(f'vocab {vocab.name} already exists')
        self[vocab.name] = vocab

    def get_info(self) -> dict:
        """
        Get the information of all vocabs
        """
        return {vocab.name: dict(
            size=len(vocab),
            cols=self.cols[vocab.name],
        ) for vocab in self.values()}

    def __call__(self, name) -> Vocab:
        return self[name]

    def __getitem__(self, item) -> Vocab:
        if isinstance(item, Col):
            item = item.voc
        if isinstance(item, Voc):
            item = item.name
        return super().__getitem__(item)
