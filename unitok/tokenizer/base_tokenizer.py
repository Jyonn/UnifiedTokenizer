import abc
from typing import Union

from unitok.utils import Instance, function
from unitok.utils.hub import Hub
from unitok.vocabulary import Vocab, VocabHub


class BaseTokenizer(abc.ABC):
    return_list: bool
    param_list: list

    prefix = 'auto_'

    def __init__(
            self,
            vocab: Union[str, Vocab],
            tokenizer_id: str = None,
            **kwargs
    ):
        if isinstance(vocab, str):
            if VocabHub.has(vocab):
                self.vocab = VocabHub.get(vocab)
            else:
                self.vocab = Vocab(name=vocab)
        else:
            self.vocab = vocab

        self._tokenizer_id = tokenizer_id

        TokenizerHub.add(self)
        VocabHub.add(self.vocab)

    def get_tokenizer_id(self):
        if self._tokenizer_id is None:
            self._tokenizer_id = self.prefix + function.get_random_string(length=6)
        return self._tokenizer_id

    @classmethod
    def get_classname(cls):
        # return cls.classname.lower().replace('tokenizer', '')
        classname = cls.__name__.lower()
        if not classname.endswith('tokenizer'):
            raise ValueError(f'({classname}) Unexpected classname, expecting classname to end with "Tokenizer"')
        return classname.replace('tokenizer', '')

    def _convert_tokens_to_ids(self, tokens):
        return_list = isinstance(tokens, list)
        if return_list != self.return_list:
            raise ValueError(f'(tokenizer.{self.get_classname()}) Unexpected input, requiring return_list={self.return_list}')

        if not return_list:
            tokens = [tokens]

        ids = [self.vocab.append(token) for token in tokens]

        if not return_list:
            ids = ids[0]
        return ids

    def __call__(self, objs):
        return self._convert_tokens_to_ids(objs)

    def __str__(self):
        return f'{self._detailed_classname}({self.get_tokenizer_id()}, vocab={self.vocab.name})'

    def __repr__(self):
        return str(self)

    def json(self):
        return {
            'tokenizer_id': self.get_tokenizer_id(),
            'vocab': self.vocab.name,
            'classname': self.get_classname(),
            'params': {param: getattr(self, param) for param in self.param_list},
        }

    @property
    def _detailed_classname(self):
        return self.__class__.__name__


class TokenizerHub(Hub[BaseTokenizer]):
    _instance = Instance()

    @classmethod
    def add(cls, key, obj: BaseTokenizer = None):
        key, obj = key.get_tokenizer_id(), key
        return super().add(key, obj)
