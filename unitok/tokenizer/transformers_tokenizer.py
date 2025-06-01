from typing import Union

from pigmento import pnt
from transformers import AutoTokenizer

from unitok.vocabulary import Vocab
from unitok.tokenizer import BaseTokenizer


class TransformersTokenizer(BaseTokenizer):
    return_list = True

    def __init__(self, vocab: Union[str, Vocab], tokenizer_id: str = None, key: str = None, **kwargs):
        super().__init__(vocab=vocab, tokenizer_id=tokenizer_id)
        self.key = key

        self.kwargs = kwargs
        self.param_list = ['key']
        self.param_list.extend(list(kwargs.keys()))

        self.tokenizer = AutoTokenizer.from_pretrained(self.key, **self.kwargs)
        self.vocab.extend(self._generate_token_list())

    def _generate_token_list(self):
        if not hasattr(self.tokenizer, 'vocab'):
            pnt(f'transformer({self.key}): does not provide vocabulary, generating placeholders instead')
            return list(range(self.tokenizer.vocab_size))

        tokens = self.tokenizer.vocab
        if isinstance(tokens, list):
            return tokens
        if not isinstance(tokens, dict):
            pnt(f'transformer({self.key}): unsupported type of vocabulary, generating placeholders instead')
            return list(range(self.tokenizer.vocab_size))

        num_tokens = len(tokens)
        token_ids = list(tokens.values())
        if max(token_ids) != num_tokens - 1 or min(token_ids) != 0:
            raise ValueError(f'transformer({self.key}): vocabulary is not continuous')

        token_list = [None] * num_tokens
        for token, idx in tokens.items():
            token_list[idx] = token
        for idx, token in enumerate(token_list):
            if token is None:
                raise ValueError(f'transformer({self.key}): missing token {idx}')
        return token_list

    def __getattr__(self, item):
        return self.kwargs[item]

    def __call__(self, obj):
        tokens = self.tokenizer.tokenize(obj)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        for token in tokens:
            self.vocab.counter(token)
        return tokens

    def __getstate__(self):
        state = self.__dict__.copy()
        state['tokenizer'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.key, **self.kwargs)


class BertTokenizer(TransformersTokenizer):
    param_list = []

    def __init__(self, **kwargs):
        kwargs.pop('key', None)
        super().__init__(key='bert-base-uncased', **kwargs)
