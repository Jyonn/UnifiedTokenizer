import math
import os
from typing import Union, List

import numpy as np

from UniTok.compatible.uni_warnings import VocabMapDeprecationWarning


class VocabMap(dict):
    def __call__(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)


class Vocab:
    """
    Vocabulary class for mapping object to index and vice versa.
    """

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise ValueError('vocab name must be a string')

        self.name = name
        self.o2i, self.i2o = VocabMap(), VocabMap()

        self._editable = True
        self.frequency_mode = False
        self.oov_default = None
        self.frequency = {}
        self.max_frequency = 0

        self.frequent_vocab = []
        self.reserve_tokens = None

    @property
    def obj2index(self) -> VocabMap:
        VocabMapDeprecationWarning()
        return self.o2i

    @property
    def index2obj(self) -> VocabMap:
        VocabMapDeprecationWarning()
        return self.i2o

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
                self.frequent_vocab.append(self.i2o[index])
        self.i2o = dict()
        self.o2i = dict()

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
        if obj not in self.o2i:
            if self.frequency_mode:
                if self.oov_default is not None:
                    return self.oov_default
                return -1
            if not self._editable:
                if self.oov_default is not None:
                    return self.oov_default
                raise ValueError(f'Vocab {self.name} is not editable, but new word [{obj}] appears')
            index = len(self.i2o)
            self.o2i[obj] = index
            self.i2o[index] = obj
        return self.o2i[obj]

    def reserve(self, tokens: Union[int, List[any]]):
        if self.get_size():
            raise ValueError(f'Vocab {self.name} is not empty and not allowed reserve operation')

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
        return [self.i2o[i] for i in range(len(self.i2o))]

    def allow_edit(self):
        self._editable = True
        return self

    def deny_edit(self):
        self._editable = False
        return self

    def get_store_path(self, store_dir):
        return os.path.join(store_dir, 'tok.{}.dat'.format(self.name))

    def load(self, store_dir: str, as_path=False):
        store_path = store_dir if as_path else self.get_store_path(store_dir)

        self.o2i, self.i2o = {}, {}
        with open(store_path, 'r') as f:
            objs = f.read().split('\n')[:-1]
        for index, obj in enumerate(objs):
            self.o2i[obj] = index
            self.i2o[index] = obj

        return self

    def save(self, store_dir):
        store_path = self.get_store_path(store_dir)
        with open(store_path, 'w') as f:
            for i in range(len(self.i2o)):
                f.write('{}\n'.format(self.i2o[i]))
        return self

    def get_size(self):
        return len(self.i2o)
