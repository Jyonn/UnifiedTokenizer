import math
import os
import warnings
from typing import Union, List

import numpy as np


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

        self.reserved_tokens = None  # reserved tokens

        self._editable = True  # whether vocab is editable
        self._oov_token = None  # out of vocabulary token
        self._stable_mode = False

        self._count_mode = False  # whether count mode is on
        self._counter = {}  # counter for counting occurrence of each token

    """
    Basic Methods
    """

    @property
    def obj2index(self) -> VocabMap:
        """
        Deprecated, use o2i instead
        """
        warnings.warn('vocab.index2obj and vocab.obj2index are deprecated, '
                      'use vocab.i2o and vocab.o2i instead (will be removed in 4.x version)', DeprecationWarning)
        return self.o2i

    @property
    def index2obj(self) -> VocabMap:
        """
        Deprecated, use i2o instead
        """
        warnings.warn('vocab.index2obj and vocab.obj2index are deprecated, '
                      'use vocab.i2o and vocab.o2i instead (will be removed in 4.x version)', DeprecationWarning)
        return self.i2o

    def extend(self, objs):
        """
        extend vocab with iterable object
        :return: index list
        """
        return [self.append(obj) for obj in objs]

    def append(self, obj):
        index = self._append(obj)
        return index

    def counts(self, indexes):
        if self._count_mode:
            for index in indexes:
                if index > -1:
                    self._counter[index] = self._counter.get(index, 0) + 1

    # def count(self, index):
    #     if self._count_mode and index > -1:
    #         self._counter[index] = self._counter.get(index, 0) + 1

    def _append(self, obj):
        """
        append object to vocab
        :return: object index
        """
        if obj in self.o2i:
            return self.o2i[obj]

        if self._stable_mode:
            if self._oov_token is not None:
                return self._oov_token
            return -1

        if not self._editable:
            if self._oov_token is not None:
                return self._oov_token
            raise ValueError(f'new token {obj} is not allowed to add to uneditable vocab {self.name}')

        index = len(self.i2o)
        self.o2i[obj] = index
        self.i2o[index] = obj
        return self.o2i[obj]

    def reserve(self, tokens: Union[int, List[any]]):
        """
        set first n tokens as reserved tokens
        """
        if len(self):
            raise ValueError(f'vocab {self.name} is not empty, can not reserve tokens')

        self.reserved_tokens = tokens
        if isinstance(tokens, int):
            digits = int(math.log10(tokens))
            token_template = '[UNUSED%%0%sd]' % digits
            tokens = [token_template % token for token in range(tokens)]  # [UNUSED000, UNUSED001, ...]

        self.extend(tokens)
        return self

    def get_tokens(self):
        warnings.warn('vocab.get_tokens is deprecated, '
                      'use list(vocab) instead (will be removed in 4.x version)', DeprecationWarning)
        return list(self)

    def get_size(self):
        warnings.warn('vocab.get_size is deprecated, '
                      'use len(vocab) instead (will be removed in 4.x version)', DeprecationWarning)
        return len(self)

    def __len__(self):
        return len(self.i2o)

    def __bool__(self):
        return True

    def __iter__(self):
        """vocab obj list iterator"""
        for i in range(len(self)):
            yield self.i2o[i]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.i2o[item]
        return self.o2i[item]

    """
    Editable Methods
    """

    @property
    def oov_default(self):
        warnings.warn('vocab.oov_default is deprecated, '
                      'use vocab.oov_token instead (will be removed in 4.x version)', DeprecationWarning)
        return self._oov_token

    def allow_edit(self):
        self._editable = True
        return self

    def deny_edit(self, oov_default=None):
        self._editable = False
        self._oov_token = oov_default or self._oov_token
        return self

    """
    Save & Load Methods
    """

    def get_store_path(self, store_dir):
        return os.path.join(store_dir, self.filename)

    @property
    def filename(self):
        return f'tok.{self.name}.dat'

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
            for token in self:
                f.write('{}\n'.format(token))

        return self

    """
    Count Mode Methods
    """

    def set_count_mode(self, count_mode=True, oov_token=None):
        """
        count mode: count occurrence of each token
        """
        self._count_mode = count_mode
        self._counter = {}
        self._oov_token = oov_token or self._oov_token
        return self

    def trim(self, min_count=None, min_frequency=1):
        """
        trim vocab by min frequency
        :return: trimmed tokens
        """
        _trimmed = []

        if min_count is None:
            warnings.warn('vocab.min_frequency is deprecated, '
                          'use vocab.min_count instead (will be removed in 4.x version)', DeprecationWarning)
            min_count = min_frequency

        vocabs = []
        for index in self._counter:
            if self._counter[index] >= min_count:
                vocabs.append(self.i2o[index])
            else:
                _trimmed.append(self.i2o[index])

        self.i2o = dict()
        self.o2i = dict()

        self.set_count_mode(False)
        if self.reserved_tokens is not None:
            self.reserve(self.reserved_tokens)
        self.extend(vocabs)

        self._stable_mode = True
        return _trimmed

    def summarize(self, base=10):
        """
        summarize vocab by frequency
        :param base: display base, default 10
        :return: counts of clustered bounds, e.g., { (1, 2): 100, (2, 3): 200, ... }
        """
        max_count = max(self._counter.values())
        digits_max = base
        while digits_max < max_count:
            digits_max = digits_max * base

        bounds = []
        while digits_max >= base:
            digits_min = digits_max // base
            left_bound = (np.arange(base - 1)[::-1] + 1) * digits_min
            right_bound = left_bound + digits_min
            bounds.extend(zip(left_bound, right_bound))
            digits_max = digits_min
        bounds.reverse()  # [(1, 2), ..., (9, 10), (10, 20), ..., (90, 100), (100, 200), ..., ...]

        counts = dict()
        for bound in bounds:
            counts[bound] = 0

        for index in self._counter:
            count = self._counter[index]
            # binary search
            left, right = 0, len(bounds) - 1
            while left <= right:
                mid = (left + right) // 2
                if bounds[mid][0] <= count < bounds[mid][1]:
                    counts[bounds[mid]] += 1
                    break
                elif count < bounds[mid][0]:
                    right = mid - 1
                else:
                    left = mid + 1

        for bound in bounds:
            if not counts[bound]:
                del counts[bound]

        return counts

    """
    Deprecated Attributes
    """

    def __getattr__(self, item):
        if item in ['frequency_mode', 'frequency', 'max_frequency', 'frequent_vocab']:
            raise AttributeError(f'{item} is deprecated after UniTok 3.0, '
                                 f'degrade to 2.4.3.2 or lower to use it, '
                                 f'or check new features of Vocab class')

    @property
    def trim_vocab(self):
        warnings.warn('vocab.trim_vocab is deprecated, '
                      'use vocab.trim instead (will be removed in 4.x version)', DeprecationWarning)
        return self.trim
