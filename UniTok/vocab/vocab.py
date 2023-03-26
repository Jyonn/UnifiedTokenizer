import math
import os
from typing import Union, List

import numpy as np

from UniTok.compatible.uni_warnings import VocabMapDeprecationWarning, OOVDefaultDeprecationWarning, \
    MinFrequencyDeprecationWarning


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

        self._count_mode = False  # whether count mode is on
        self._counter = {}  # counter for counting occurrence of each token

        # self.frequency_mode = False
        # self.frequency = {}
        # self.max_frequency = 0

        # self.frequent_vocab = []

    """
    Basic Methods
    """

    @property
    def obj2index(self) -> VocabMap:
        """
        Deprecated, use o2i instead
        """
        VocabMapDeprecationWarning()
        return self.o2i

    @property
    def index2obj(self) -> VocabMap:
        """
        Deprecated, use i2o instead
        """
        VocabMapDeprecationWarning()
        return self.i2o

    def extend(self, objs):
        """
        extend vocab with iterable object
        :return: index list
        """
        return [self.append(obj) for obj in objs]

    def append(self, obj):
        index = self._append(obj)
        if self._count_mode and index > -1:
            self._counter[index] = self._counter.get(index, 0) + 1
        return index

    def _append(self, obj):
        """
        append object to vocab
        :return: object index
        """
        if obj in self.o2i:
            return self.o2i[obj]

        if self._count_mode:
            return self._oov_token or -1

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
        if self.get_size():
            raise ValueError(f'vocab {self.name} is not empty, can not reserve tokens')

        self.reserved_tokens = tokens
        if isinstance(tokens, int):
            digits = int(math.log10(tokens))
            token_template = '[UNUSED%%0%sd]' % digits
            tokens = [token_template % token for token in range(tokens)]  # [UNUSED000, UNUSED001, ...]

        self.extend(tokens)
        return self

    def get_tokens(self):
        return [self.i2o[i] for i in range(len(self))]

    def get_size(self):
        return len(self.i2o)

    def __len__(self):
        return self.get_size()

    """
    Editable Methods
    """

    @property
    def oov_default(self):
        OOVDefaultDeprecationWarning()
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
            for i in range(len(self.i2o)):
                f.write('{}\n'.format(self.i2o[i]))

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
        :return:
        """
        if min_count is None:
            MinFrequencyDeprecationWarning()
            min_count = min_frequency

        vocabs = []
        for index in self._counter:
            if self._counter[index] >= min_count:
                vocabs.append(self.i2o[index])

        self.i2o = dict()
        self.o2i = dict()

        self.set_count_mode(False)
        if self.reserved_tokens is not None:
            self.reserve(self.reserved_tokens)
        self.extend(vocabs)

        # self.frequency_mode = True
        return self

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
