import collections
import math
import os
from typing import Union, List, Optional

import numpy as np
from smartify import E


@E.register(id_processor=E.idp_cls_prefix())
class VocabError:
    NotEditable = E('Vocab {} is not editable, but new word [{}] appears')
    NotEmptyForReserve = E('Vocab {} is not empty and not allowed reserve operation')


class Vocab:
    __VOCAB_ID = 0

    @classmethod
    def get_vocab_id(cls):
        vocab_id = cls.__VOCAB_ID
        cls.__VOCAB_ID += 1
        return vocab_id

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise ValueError('Vocab name should be string')

        self.name = name
        self.obj2index, self.index2obj = {}, {}
        self.editable = True
        self.frequency_mode = False
        self.oov_default = None
        self.vocab_id = Vocab.get_vocab_id()
        self.frequency = {}
        self.max_frequency = 0

        self.frequent_vocab = []
        self.reserve_tokens = None

    def init_frequency(self):
        self.frequency = {}
        self.max_frequency = 0

    def frequency_count(self, *ids):
        for index in ids:
            if index not in self.frequency:
                self.frequency[index] = 0
            self.frequency[index] += 1
            if self.max_frequency < self.frequency[index]:
                self.max_frequency = self.frequency[index]

    def trim_vocab(self, min_frequency=1, oov_default=None):
        self.oov_default = self.oov_default or oov_default
        self.frequent_vocab = []
        for index in self.frequency:
            if self.frequency[index] >= min_frequency:
                self.frequent_vocab.append(self.index2obj[index])
        self.index2obj = dict()
        self.obj2index = dict()

        if self.reserve_tokens is not None:
            self.reserve(self.reserve_tokens)
        self.extend(self.frequent_vocab)

        self.frequency_mode = True

    def frequency_analyse(self):
        max_count = self.max_frequency
        digits_max = 10
        while digits_max < max_count:
            digits_max = digits_max * 10

        bounds = []
        while digits_max >= 10:
            digits_min = digits_max // 10
            left_bound = (np.arange(9)[::-1] + 1) * digits_min
            right_bound = left_bound + digits_min
            bounds.extend(zip(left_bound, right_bound))
            digits_max = digits_min
        bounds.append((0, 1))
        bounds.reverse()

        bound_dict = dict()
        for bound in bounds:
            bound_dict[bound] = 0

        for index in self.frequency:
            for bound in bounds:
                if bound[1] > self.frequency[index] >= bound[0]:
                    bound_dict[bound] += 1
                    break

        for bound in bounds:
            if not bound_dict[bound]:
                del bound_dict[bound]

        return bound_dict

    def extend(self, objs):
        for obj in objs:
            self.append(obj)
        return self

    def append(self, obj) -> int:
        if obj not in self.obj2index:
            if self.frequency_mode:
                if self.oov_default is not None:
                    return self.oov_default
                return -1
            if not self.editable:
                if self.oov_default is not None:
                    return self.oov_default
                raise VocabError.NotEditable(self.name, obj)
            index = len(self.index2obj)
            self.obj2index[obj] = index
            self.index2obj[index] = obj
        return self.obj2index[obj]

    def reserve(self, tokens: Union[int, List[any]]):
        if self.get_size():
            raise VocabError.NotEmptyForReserve(self.name)

        self.reserve_tokens = tokens
        if isinstance(tokens, int):
            digits = int(math.log10(tokens))
            token_template = '[UNUSED%%0%sd]' % digits
            for token in range(tokens):
                self.append(token_template % token)
        else:
            for token in tokens:
                self.append(token)
        return self

    def get_tokens(self):
        return [self.index2obj[i] for i in range(len(self.index2obj))]

    def allow_edit(self):
        self.editable = True
        return self

    def deny_edit(self):
        self.editable = False
        return self

    def get_store_path(self, store_dir):
        return os.path.join(store_dir, 'tok.{}.dat'.format(self.name))

    def load(self, store_dir: str, as_path=False):
        store_path = store_dir if as_path else self.get_store_path(store_dir)

        self.obj2index, self.index2obj = {}, {}
        with open(store_path, 'r') as f:
            objs = f.read().split('\n')[:-1]
        for index, obj in enumerate(objs):
            self.obj2index[obj] = index
            self.index2obj[index] = obj

        return self

    def save(self, store_dir):
        store_path = self.get_store_path(store_dir)
        with open(store_path, 'w') as f:
            for i in range(len(self.index2obj)):
                f.write('{}\n'.format(self.index2obj[i]))
        return self

    def get_size(self):
        return len(self.index2obj)
