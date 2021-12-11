import json
import os
import random
from typing import Dict, List

import numpy as np

from .unitok import UniTok
from .classify import Classify
from .vocab import VocabDepot, Vocab


class UniDep:
    def __init__(self, store_dir):
        self.store_dir = os.path.expanduser(store_dir)

        self.meta_path = os.path.join(self.store_dir, 'meta.data.json')
        self.meta_data = Classify(json.load(open(self.meta_path)))

        if self.meta_data.version != UniTok.VER:
            print('UniTok version not match, it may occur unexpected errors in loading phase!')

        self.data_path = os.path.join(self.store_dir, 'data.npy')
        self.data = np.load(self.data_path, allow_pickle=True)
        try:
            self.data = self.data.item()  # type: dict
        except Exception as err:
            print(err)

        self.id_col = self.meta_data.id_col
        self.id_vocab = self.meta_data.col_info.d[self.id_col].vocab
        self.sample_size = self.meta_data.vocab_info.d[self.id_vocab].size
        print('Loaded', self.sample_size, 'samples!')

        data_sample_size = len(self.data[self.id_col])
        if self.sample_size != data_sample_size:
            print('Resize sample size to', data_sample_size)
            self.sample_size = data_sample_size

        self.col_info = self.meta_data.col_info
        self.vocab_info = self.meta_data.vocab_info

        self.vocab_depot = VocabDepot()
        for vocab_name in self.vocab_info.d:
            self.vocab_depot.append(Vocab(name=vocab_name).load(self.store_dir))
        self.id2index = self.vocab_depot.depot[self.id_vocab].obj2index

        self.index_order = list(range(self.sample_size))
        self.union_depots = dict()  # type: Dict[str, List[UniDep]]

    @staticmethod
    def merge_col(c1: Classify, c2: Classify):
        for col_name in c2.d:
            if col_name in c1.d and c1.d[col_name].dict() != c2.d[col_name].dict():
                raise ValueError('Column Config Conflict In Key {}'.format(col_name))
        d = c1.dict()
        d.update(c2.dict())
        return Classify(d)

    @staticmethod
    def merge_vocab(c1: Classify, c2: Classify):
        for vocab_name in c2.d:
            if vocab_name in c1.d and c1.d[vocab_name].size != c2.d[vocab_name].size:
                raise ValueError('Vocab Config Conflict In Key {}'.format(vocab_name))
        d = c1.dict()
        d.update(c2.dict())
        return Classify(d)

    def union(self, *depots: 'UniDep'):
        for depot in depots:
            if depot.id_col not in self.col_info.d:
                raise ValueError('Current Depot Has No Column Named {}'.format(depot.id_col))

            if depot.id_col not in self.union_depots:
                self.union_depots[depot.id_col] = []
            self.union_depots[depot.id_col].append(depot)
            self.col_info = self.merge_col(self.col_info, depot.col_info)
            self.vocab_info = self.merge_vocab(self.vocab_info, depot.vocab_info)
            self.meta_data.col_info = self.col_info
            self.meta_data.vocab_info = self.vocab_info

    def is_list_col(self, col_name):
        return 'max_length' in self.col_info.d[col_name].d

    def get_vocab_size(self, col_name, as_vocab=False):
        vocab_id = col_name if as_vocab else self.get_vocab(col_name)
        return self.vocab_info.d[vocab_id].size

    def get_vocab(self, col_name):
        return self.col_info.d[col_name].vocab

    def get_max_length(self, col_name):
        if self.is_list_col(col_name):
            return self.col_info.d[col_name].max_length

    def get_sample_by_id(self, obj_id):
        return self.pack_sample(self.id2index[obj_id])

    def shuffle(self, shuffle=True):
        if shuffle:
            random.shuffle(self.index_order)
        else:
            self.index_order = list(range(self.sample_size))

    def pack_sample(self, index):
        sample = dict()
        for col_name in self.col_info.d:
            if col_name in self.data:
                sample[col_name] = self.data[col_name][index]
                if col_name in self.union_depots:
                    for depot in self.union_depots[col_name]:
                        sample.update(depot[sample[col_name]])
        return sample

    def __getitem__(self, index):
        index = self.index_order[index]
        return self.pack_sample(index)

    def __len__(self):
        return self.sample_size
