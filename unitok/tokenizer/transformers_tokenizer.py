from typing import Union

from transformers import AutoTokenizer

from UniTokv3.vocab import Vocab
from unitok.tokenizer import CachableTokenizer


class TransformersTokenizer(CachableTokenizer):
    return_list = True

    def __init__(self, vocab: Union[str, Vocab], tokenizer_id: str = None, use_cache=False, key: str = None, **kwargs):
        super().__init__(vocab=vocab, tokenizer_id=tokenizer_id, use_cache=use_cache)
        self.key = key

        self.kwargs = kwargs
        self.param_list = ['key', 'use_cache']
        self.param_list.extend(list(kwargs.keys()))

        self.tokenizer = AutoTokenizer.from_pretrained(self.key, **self.kwargs)
        self.vocab.extend(self._generate_token_list())

    def _generate_token_list(self):
        tokens = self.tokenizer.vocab
        if isinstance(tokens, list):
            return tokens
        if not isinstance(tokens, dict):
            raise ValueError(f'transformer({self.key}): unsupported type of vocabulary')

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
        return super().__call__(tokens)


class BertTokenizer(TransformersTokenizer):
    param_list = []

    def __init__(self, **kwargs):
        super().__init__(key='bert-base-uncased', **kwargs)
