from typing import Iterable, Callable

from UniTok.vocab.vocab import Vocab


class BaseTok:
    """
    Meta Tokenizer
    """
    return_list = None

    def __init__(self, name: str = None, vocab: Vocab = None, pre_handler: Callable = None):
        """
        :param name: vocab name
        :param vocab: vocab object
        :param pre_handler: pre handler for token
        """
        assert name or vocab, ValueError('name and vocab can not both be null')

        self.PAD = 0  # padding index
        self.vocab = vocab or Vocab(name)  # build vocab
        self.pre_handler = pre_handler

        assert self.return_list is not None, ValueError('class attribute return_list should be set')

    def insert(self, token):
        """
        insert token into vocab
        :return: token index
        """
        if self.pre_handler:
            token = self.pre_handler(token)
        return self.vocab.append(token)

    def t(self, obj) -> [int, list]:
        """
        tokenize object
        :return: token index or token index list
        """
        raise NotImplemented

    def _t(self, obj):
        """
        wrapped tokenize method, filter out unknown token
        """
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
