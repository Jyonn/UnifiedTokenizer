import json
import os
import random
import warnings
from typing import Dict, List

import numpy as np
from oba import Obj

from .vocab import VocabDepot, Vocab


class UniDep:
    VER = 'UniDep-1.0'

    def __init__(self, store_dir):
        self.store_dir = os.path.expanduser(store_dir)

        self.meta_path = os.path.join(self.store_dir, 'meta.data.json')
        self.meta_data = Obj(json.load(open(self.meta_path)))

        if self.meta_data.version.startswith('UniDep'):
            if self.meta_data.version != UniDep.VER:
                raise ValueError(
                    'UniDep version mismatch, '
                    'current version: {}, '
                    'depot version: {}. '
                    'It may cause unexpected error.'.format(
                        UniDep.VER, self.meta_data.version
                    ))

        self.data_path = os.path.join(self.store_dir, 'data.npy')
        self.data = np.load(self.data_path, allow_pickle=True)
        try:
            self.data = self.data.item()  # type: dict
        except Exception as err:
            print(err)

        self.col_info = self.meta_data.col_info
        self.vocab_info = self.meta_data.vocab_info

        self.id_col = self.meta_data.id_col
        self.id_vocab = self.get_vocab(self.id_col)
        self.sample_size = self.get_vocab_size(self.id_col)
        print('Loaded', self.sample_size, 'samples!')

        data_sample_size = len(self.data[self.id_col])
        if self.sample_size != data_sample_size:
            print('Resize sample size to', data_sample_size)
            self.sample_size = data_sample_size

        self.vocab_depot = VocabDepot()
        for vocab_name in self.vocab_info:
            self.vocab_depot.append(Vocab(name=vocab_name).load(self.store_dir))
        self.id2index = self.vocab_depot[self.id_vocab].obj2index

        self._visible_indexes = list(range(self.sample_size))
        self.union_depots = dict()  # type: Dict[str, List[UniDep]]

    @staticmethod
    def _merge_col(c1: Obj, c2: Obj):
        for col_name in c2:
            col_data = c2[col_name]
            if col_name in c1 and Obj.raw(c1[col_name]) != Obj.raw(col_data):
                raise ValueError('Column Config Conflict In Key {}'.format(col_name))
        d = Obj.raw(c1)
        d.update(Obj.raw(c2))
        return Obj(d)

    @staticmethod
    def _merge_vocab(c1: Obj, c2: Obj):
        for vocab_name in c2:
            vocab_data = c2[vocab_name]
            if vocab_name in c1 and c1[vocab_name].size != vocab_data.size:
                raise ValueError('vocab config conflict in key {}'.format(vocab_name))
        d = Obj.raw(c1)
        d.update(Obj.raw(c2))
        return Obj(d)

    def union(self, *depots: 'UniDep'):
        for depot in depots:
            if depot.id_col not in self.col_info:
                raise ValueError('current depot has no column named {}'.format(depot.id_col))

            if depot.id_col not in self.union_depots:
                self.union_depots[depot.id_col] = []
            self.union_depots[depot.id_col].append(depot)
            self.col_info = self._merge_col(self.col_info, depot.col_info)
            self.vocab_info = self._merge_vocab(self.vocab_info, depot.vocab_info)
            self.meta_data.col_info = self.col_info
            self.meta_data.vocab_info = self.vocab_info

    def filter(self, filter_func):
        self._visible_indexes = [i for i in range(self.sample_size) if filter_func(self[i])]
        self.sample_size = len(self._visible_indexes)
        return self

    def is_list_col(self, col_name):
        return 'max_length' in self.col_info[col_name]

    def get_vocab_size(self, col_name, as_vocab=False):
        vocab_id = col_name if as_vocab else self.get_vocab(col_name)
        return self.vocab_info[vocab_id].size

    def get_vocab(self, col_name):
        return self.col_info[col_name].vocab

    def get_max_length(self, col_name):
        if self.is_list_col(col_name):
            return self.col_info[col_name].max_length

    def get_sample_by_id(self, obj_id):
        return self.pack_sample(self.id2index[obj_id])

    def shuffle(self, shuffle=True):
        warnings.warn('shuffle is deprecated, data shuffle is more encouraged in the data loader', DeprecationWarning)
        if shuffle:
            random.shuffle(self._visible_indexes)
        else:
            self._visible_indexes = list(range(self.sample_size))

    def pack_sample(self, index):
        sample = dict()
        for col_name in self.col_info:
            if col_name in self.data:
                sample[col_name] = self.data[col_name][index]
                if col_name in self.union_depots:
                    for depot in self.union_depots[col_name]:
                        sample.update(depot[sample[col_name]])
        return sample

    def __getitem__(self, index):
        index = self._visible_indexes[index]
        return self.pack_sample(index)

    def __len__(self):
        return self.sample_size

    def __str__(self):
        return f'UniDep from {self.store_dir}'

    def __repr__(self):
        return str(self)
