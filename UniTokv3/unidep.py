import json
import os
import random
import warnings
from collections import OrderedDict
from typing import Dict, List, Callable, Union, Optional, cast

import numpy as np
import tqdm

from .meta import Meta, Col, Voc
from .vocab import Vocab
from .vocabs import Vocabs


class UniDep:
    VER = Meta.VER

    def __init__(self, store_dir, verbose=True, silent=None):
        """
        Unified Data Depot Initialization
        :param store_dir: Store directory of the data processed by our UniTok or Fut
        :param verbose:
        """
        self.store_dir = os.path.expanduser(store_dir)
        self.meta = Meta(self.store_dir)

        if silent is not None:
            warnings.warn('unidep.silent is deprecated, '
                          'use verbose instead (will be removed in 4.x version)', DeprecationWarning)
            verbose = not silent
        self.verbose = verbose

        self.print(
            "UniTok-v4 is coming!\n"
            "Try the new version with:\n"
            "    `from unitok import UniTok`\n"
            "UniTok-v3 is still available, but will be deprecated in the future:\n"
            "    `from UniTokv3 import UniTok`\n"
            "Documentation: https://unitok.github.io"
        )

        self.cached = False
        self.cached_samples = []

        self.data_path = os.path.join(self.store_dir, 'data.npy')
        self.data = np.load(self.data_path, allow_pickle=True)
        try:
            self.data = self.data.item()
            self.data = cast(dict, self.data)
        except Exception as err:
            print(err)
            return

        self.cols = self.meta.cols  # type: Dict[str, Col]
        self.vocs = self.meta.vocs  # type: Dict[str, Voc]

        self.id_col = self.meta.id_col
        self.id_voc = self.cols[self.id_col].voc

        self._indexes = []
        self.sample_size = -1
        self.set_sample_size(self.id_voc.size)

        self._sample_size = len(self.data[self.id_col])
        if self.sample_size != self._sample_size:
            self.set_sample_size(self._sample_size)

        self.vocabs = Vocabs()
        for vocab_name in self.vocs:
            self.vocabs.append(Vocab(name=vocab_name).load(self.store_dir))
        for voc in self.vocs:
            self.vocs[voc].vocab = self.vocabs[voc]
        self.id2index = self.id_voc.vocab.o2i

        self.unions = OrderedDict()  # type: Dict[str, List[UniDep]]
        self._deep_union = False

    def set_sample_size(self, size):
        modify_flag = self.sample_size > -1

        self.sample_size = size
        self._indexes = list(range(self.sample_size))

        if modify_flag:
            self.print('modify sample_size to', self.sample_size)
        else:
            self.print(f'loaded {self.sample_size} samples from {self.store_dir}')

    def print(self, *args, **kwargs):
        """
        silent-aware printer
        """

        if not self.verbose:
            return
        print(*args, **kwargs)

    def pack_sample(self, index) -> dict:
        """
        pack sample into dict by raw index (data index)
        """
        if self.cached:
            return self.cached_samples[index]

        sample = dict()

        for col_name in self.data:
            sample[col_name] = self.data[col_name][index]
        if self._deep_union:
            return sample

        for col_name in self.unions:
            col_value = sample[col_name]
            for depot in self.unions[col_name]:
                sample.update(depot[col_value])
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
        for sample in tqdm.tqdm(self, disable=not self.verbose):
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
    Advanced methods, including union, filter 
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

    def deep_union(self, value):
        if self._deep_union != value and self.unions:
            raise ValueError('deep_union can not be changed after union-ed')
        self._deep_union = value

    def union(self, *depots: 'UniDep', union_col: str = None):
        """
        union depots, where id columns in other depots must exist in current main depot
        """
        if union_col and union_col not in self.cols:
            raise ValueError(f'current depot has no column named {union_col}')

        for depot in depots:
            # check if id col exists in current depot
            if not union_col:
                assert depot.id_col in self.cols, (
                    ValueError(f'current depot has no column named {union_col}'))
            else:
                assert self.cols[union_col].voc == depot.cols[depot.id_col].voc, (
                    ValueError(f'the vocabs of union col {union_col} and target id col {depot.id_col} are not matched'))
            current_union_col = union_col or depot.id_col

            if current_union_col not in self.unions:
                self.unions[current_union_col] = []
            self.unions[current_union_col].append(depot)

            self.cols = self._merge_cols(self.cols, depot.cols)
            self.vocs = self._merge_vocs(self.vocs, depot.vocs)
            self.meta.cols = self.cols
            self.meta.vocs = self.vocs

            if not self._deep_union:
                continue

            columns = {col_name: [] for col_name in depot.cols}

            for index in self.data[current_union_col]:
                for col_name in columns:
                    columns[col_name].append(depot.data[col_name][index])

            for col_name in columns:
                values = np.array(columns[col_name], dtype=object)
                self.data[col_name] = values
        return self

    def inject(self, depot: 'UniDep', col_names: Union[list, dict]):
        if isinstance(col_names, list):
            col_names = {col_name: col_name for col_name in col_names}
        if depot.id_col not in self.cols:
            raise ValueError(f'current depot has no column named {depot.id_col}')

        columns = {col_names[col_name]: [] for col_name in col_names}
        for index in self.data[depot.id_col]:
            for col_name in columns:
                columns[col_name].append(depot.data[col_name][index])

        for col_name in columns:
            self.set_col(
                name=col_name,
                values=columns[col_name],
                vocab=depot.cols[col_name].voc.vocab,
            )

    def rename_col(self, old_name: str, new_name: str):
        """
        rename a column
        """
        if old_name not in self.cols:
            raise ValueError(f'column {old_name} not found')
        if new_name in self.cols:
            raise ValueError(f'column {new_name} already exists')
        if old_name is self.id_col:
            self.meta.id_col = self.id_col = new_name
        self.cols[new_name] = self.cols[old_name]
        self.cols[new_name].name = new_name
        del self.cols[old_name]
        self.data[new_name] = self.data[old_name]
        del self.data[old_name]

    def rename_vocab(self, old_name: str, new_name: str):
        """
        rename a vocab
        """
        if old_name not in self.vocs:
            raise ValueError(f'vocab {old_name} not found')
        if new_name in self.vocs:
            if self.vocs[old_name].size != self.vocs[new_name].size:
                raise ValueError(f'vocab {new_name} already exists with different size')
            self.vocs[new_name].cols.extend(self.vocs[old_name].cols)
        else:
            self.vocs[new_name] = self.vocs[old_name]
            self.vocabs[new_name] = self.vocabs[old_name]
            self.vocs[new_name].name = new_name
            self.vocabs[new_name].name = new_name
        del self.vocs[old_name]
        del self.vocabs[old_name]

    def remove_col(self, col_name: str):
        """
        remove a column
        """
        if col_name not in self.cols:
            raise ValueError(f'column {col_name} not found')
        if col_name is self.id_col:
            raise ValueError('id column can not be removed')
        voc = self.cols[col_name].voc
        voc.cols = [col for col in voc.cols if col.name != col_name]
        if not voc.cols:
            del self.vocs[voc.name]
            del self.vocabs[voc.name]
        del self.cols[col_name]
        del self.data[col_name]

    def filter(self, filter_func: Callable, col=None):
        """
        filter samples by filter_func
        :param filter_func: function to filter samples
        :param col: column name to filter, if None, filter by sample itself
        """
        visible_indexes = []

        for sample in tqdm.tqdm(self, disable=not self.verbose):
            target = sample if col is None else sample[col]
            if filter_func(target):
                visible_indexes.append(sample[self.id_col])

        self._indexes = visible_indexes
        self.sample_size = len(self._indexes)
        return self

    def reset_index(self):
        """
        reset index to 0, 1, 2, ...
        """
        data = {col: [] for col in self.cols}
        for i, sample in enumerate(self):
            for col in sample:
                data[col].append(i if col == self.id_col else sample[col])
        self.reset(data)
        tokens = data[self.id_col]
        i2o = {i: o for i, o in enumerate(tokens)}
        o2i = {o: i for i, o in enumerate(tokens)}
        voc = self.cols[self.id_col].voc
        vocab = voc.vocab  # type: Vocab
        vocab.i2o, vocab.o2i = i2o, o2i
        voc.size = len(vocab)
        return self

    def export(self, store_dir):
        """
        export modified, union-ed or filtered depot
        """

        os.makedirs(store_dir, exist_ok=True)
        data = dict()

        for voc in self.vocabs:
            self.vocabs[voc].save(store_dir)

        for sample in tqdm.tqdm(self, disable=not self.verbose):
            for col_name in sample:
                if col_name not in data:
                    data[col_name] = []
                data[col_name].append(sample[col_name])

        for col_name in data:
            data[col_name] = np.array(data[col_name], dtype=object)
        np.save(os.path.join(store_dir, 'data.npy'), data, allow_pickle=True)

        meta_data = self.meta.get_info()
        json.dump(meta_data, open(os.path.join(store_dir, 'meta.data.json'), 'w'), indent=2)

    """
    Editing methods
    """

    def select_cols(self, col_names: List[str]):
        """
        select columns to keep
        """
        delete_cols = []
        for col_name in self.data:
            if col_name not in col_names:
                delete_cols.append(col_name)

        for col_name in delete_cols:
            del self.data[col_name]
            del self.cols[col_name]

    def reset(self, data):
        """
        reset data with new data
        """
        self.data = data
        self.set_sample_size(len(data[self.id_col]))
        self.cached = False

    @staticmethod
    def _get_max_length(values):
        if isinstance(values[0], list) or isinstance(values[0], np.ndarray):
            return max([len(value) for value in values])
        return None

    def set_vocab(self, vocab: Vocab):
        """
        reset or add a vocab, if vocab with same name exists, it will be reset
        """
        voc = Voc(
            name=vocab.name,
            size=len(vocab),
            cols=[],
            store_dir=self.store_dir,
            vocab=vocab,
        )
        if vocab.name in self.vocs:
            voc.cols = self.vocs[vocab.name].cols

    def set_col(self, name: str, values: Union[list, np.ndarray], vocab: Optional[Union[str, Voc, Vocab]] = None):
        """
        reset or add a column, vocab with same name will not be reset (you can use set_vocab to reset it)
        """
        assert len(values) == self.sample_size, 'values length must be equal to sample size'

        if isinstance(values, list):
            values = np.array(values, dtype=object)

        if vocab is None:
            assert name in self.cols, 'vocab must be specified when adding a new column'
            vocab = self.cols[name].voc
        if isinstance(vocab, str):
            vocab = self.vocs[vocab]
        if isinstance(vocab, Voc):
            vocab = vocab.vocab

        if vocab.name in self.vocs:
            voc = self.vocs[vocab.name]
        else:
            voc = Voc(
                name=vocab.name,
                size=len(vocab),
                cols=[],
                store_dir=self.store_dir,
                vocab=vocab,
            )
            self.vocs[vocab.name] = voc
            self.vocabs[vocab.name] = vocab

        self.data[name] = values
        self.cols[name] = Col(
            name=name,
            voc=voc,
            vocab=vocab,
            max_length=self._get_max_length(values),
        )
        voc.cols.append(self.cols[name])

    def add_samples(self, samples: Dict[str, list]):
        """
        add samples to depot
        """
        sample_size = 0
        for name, values in samples.items():
            if isinstance(values, np.ndarray):
                values = values.tolist()
            assert self.sample_size == 0 or len(values) == self.sample_size, "sample size not match"
            sample_size = len(values)
            new_list = self.data[name].tolist()
            new_list.extend(values)
            self.data[name] = np.array(new_list, dtype=object)
        self.set_sample_size(self.sample_size + sample_size)

    """
    Deprecated properties and methods
    """

    def reset_data(self, data):
        warnings.warn('reset_data is deprecated, '
                      'use reset instead (will be removed in 4.x version)', DeprecationWarning)
        self.reset(data)

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
