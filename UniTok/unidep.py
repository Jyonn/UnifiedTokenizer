import json
import os
import random
import warnings
from typing import Dict, List, Callable

import numpy as np
import tqdm

from .meta import Meta, Col, Voc
from .vocab import Vocab
from .vocabs import Vocabs


class UniDep:
    VER = Meta.VER

    def __init__(self, store_dir, silent=False):
        self.store_dir = os.path.expanduser(store_dir)
        self.meta = Meta(self.store_dir)

        self.silent = silent

        self.cached = False
        self.cached_samples = []

        self.data_path = os.path.join(self.store_dir, 'data.npy')
        self.data = np.load(self.data_path, allow_pickle=True)
        try:
            # noinspection PyTypeChecker
            self.data: dict = self.data.item()
        except Exception as err:
            print(err)
            return

        self.cols = self.meta.cols  # type: Dict[str, Col]
        self.vocs = self.meta.vocs  # type: Dict[str, Voc]

        self.id_col = self.meta.id_col
        self.id_voc = self.cols[self.id_col].voc
        self.sample_size = self.id_voc.size
        self.print('loaded', self.sample_size, 'samples!')

        self._sample_size = len(self.data[self.id_col])
        if self.sample_size != self._sample_size:
            self.print('resize sample_size to', self._sample_size)
            self.sample_size = self._sample_size

        self.vocabs = Vocabs()  # type: Union[Dict[str, Vocab], Vocabs]
        for vocab_name in self.vocs:
            self.vocabs.append(Vocab(name=vocab_name).load(self.store_dir))
        self.id2index = self.vocabs[self.id_voc.name].o2i

        self._indexes = list(range(self.sample_size))
        self.unions = dict()  # type: Dict[str, List[UniDep]]

    def print(self, *args, **kwargs):
        """
        silent-aware printer
        """

        if self.silent:
            return
        print(*args, **kwargs)

    def pack_sample(self, index) -> dict:
        """
        pack sample into dict by raw index (data index)
        """
        if self.cached:
            return self.cached_samples[index]

        sample = dict()
        for col_name in self.cols:
            if col_name in self.data:
                sample[col_name] = self.data[col_name][index]
                if col_name in self.unions:
                    for depot in self.unions[col_name]:
                        sample.update(depot[sample[col_name]])
        return sample

    def get_sample_by_id(self, obj_id):
        return self.pack_sample(self.id2index[obj_id])

    def start_caching(self):
        """
        cache all samples into memory
        """

        if self.cached:
            return

        self.cached = False
        self.cached_samples = [None] * self._sample_size
        for sample in tqdm.tqdm(self, disable=self.silent):
            self.cached_samples[sample[self.id_col]] = sample
        self.cached = True

    def __getitem__(self, index):
        index = self._indexes[index]
        return self.pack_sample(index)

    def __iter__(self):
        """vocab obj list iterator"""
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.sample_size

    def __str__(self):
        """        UniDep (dir):

        Sample Size: 1000
        Id Column: id
        Columns:
            id, vocab index (size 1000)
            text, vocab eng (size 30522), max length 100
            label, vocab label (size 2)
        """
        introduction = f"""
        UniDep ({self.meta.parse_version(self.meta.version)}): {self.store_dir}
        
        Sample Size: {self.sample_size}
        Id Column: {self.id_col}
        Columns:\n"""

        for col_name, col in self.cols.items():  # type: str, Col
            introduction += f'        \t{col_name}, vocab {col.voc.name} (size {col.voc.size})'
            if col.max_length:
                introduction += f', max length {col.max_length}'
            introduction += '\n'
        return introduction

    def __repr__(self):
        return str(self)

    """
    Advanced methods, including union and filter
    """

    @staticmethod
    def _merge(d1: dict, d2: dict) -> dict:
        d = d1.copy()
        d.update(d2)
        return d

    @classmethod
    def _merge_cols(cls, c1: Dict[str, Col], c2: Dict[str, Col]) -> Dict[str, Col]:
        for name, col in c2.items():
            if name in c1 and c1[name] != col:
                raise ValueError(f'col {name} config conflict')
        return cls._merge(c1, c2)

    @classmethod
    def _merge_vocs(cls, v1: Dict[str, Voc], v2: Dict[str, Voc]) -> Dict[str, Voc]:
        merged = v1.copy()
        for name, vocab in v2.items():
            if name in v1:
                if v1[name] != vocab:
                    raise ValueError(f'vocab {name} config conflict')
                vocab = v1[name].merge(vocab)
            merged[name] = vocab
        return merged

    def union(self, *depots: 'UniDep'):
        """
        union depots, where id columns in other depots must exist in current main depot
        """
        for depot in depots:
            # check if id col exists in current depot
            if depot.id_col not in self.cols:
                raise ValueError('current depot has no column named {}'.format(depot.id_col))

            if depot.id_col not in self.unions:
                self.unions[depot.id_col] = []
            self.unions[depot.id_col].append(depot)

            self.cols = self._merge_cols(self.cols, depot.cols)
            self.vocs = self._merge_vocs(self.vocs, depot.vocs)
            self.meta.cols = self.cols
            self.meta.vocs = self.vocs
        return self

    def filter(self, filter_func: Callable, col=None):
        """
        filter samples by filter_func
        :param filter_func: function to filter samples
        :param col: column name to filter, if None, filter by sample itself
        """
        visible_indexes = []

        for sample in tqdm.tqdm(self, disable=self.silent):
            target = sample[col] if col else sample
            if filter_func(target):
                visible_indexes.append(sample[self.id_col])
        self._indexes = visible_indexes
        self.sample_size = len(self._indexes)
        return self

    def export(self, store_dir):
        """
        export unioned or filtered depot
        """

        os.makedirs(store_dir, exist_ok=True)
        data = dict()

        for voc in self.meta.vocs.values():
            voc.export(store_dir)

        for sample in tqdm.tqdm(self, disable=self.silent):
            for col_name in sample:
                if col_name not in data:
                    data[col_name] = []
                data[col_name].append(sample[col_name])

        for col_name in data:
            data[col_name] = np.array(data[col_name])
        np.save(os.path.join(store_dir, 'data.npy'), data, allow_pickle=True)

        meta_data = self.meta.get_info()
        json.dump(meta_data, open(os.path.join(store_dir, 'meta.data.json'), 'w'), indent=2)

    """
    Deprecated properties and methods
    """

    @property
    def meta_data(self):
        warnings.warn('meta_data is deprecated, '
                      'use meta instead (will be removed in 4.x version)', DeprecationWarning)
        return self.meta

    @property
    def vocab_info(self):
        warnings.warn('vocab_info is deprecated, '
                      'use vocs instead (will be removed in 4.x version)', DeprecationWarning)
        return self.vocs

    @property
    def col_info(self):
        warnings.warn('col_info is deprecated, '
                      'use cols instead (will be removed in 4.x version)', DeprecationWarning)
        return self.cols

    def get_vocab_size(self, col_name, as_vocab=False):
        warnings.warn('unidep.get_vocab_size is deprecated (will be removed in 4.x version)', DeprecationWarning)
        vocab_id = col_name if as_vocab else self.cols[col_name].voc.name
        return self.vocs[vocab_id].size

    def get_vocab(self, col_name):
        warnings.warn('unidep.get_vocab is deprecated (will be removed in 4.x version)', DeprecationWarning)
        return self.cols[col_name].voc.name

    def get_max_length(self, col_name):
        warnings.warn('unidep.get_max_length is deprecated (will be removed in 4.x version)', DeprecationWarning)
        return self.cols[col_name].max_length

    def is_list_col(self, col_name):
        warnings.warn('unidep.is_list_col is deprecated (will be removed in 4.x version)', DeprecationWarning)
        return self.cols[col_name].list

    def shuffle(self, shuffle=True):
        warnings.warn('unidep.shuffle is deprecated (will be removed in 4.x version)', DeprecationWarning)
        if shuffle:
            random.shuffle(self._indexes)
        else:
            self._indexes = list(range(self.sample_size))
