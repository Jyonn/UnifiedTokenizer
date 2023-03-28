import json
import os
import warnings
from typing import List


class Col:
    def __init__(self, name, voc=None, max_length=None, padding=None, vocab=None):
        self.name: str = name
        self.voc: Voc = voc or vocab
        self.max_length = max_length
        self.padding = padding
        self.list = max_length is not None

    def __eq__(self, other):
        return self.name == other.name and self.voc.name == other.voc.name and self.max_length == other.max_length

    def get_info(self):
        info = {
            'voc': self.voc.name,
        }
        if self.list:
            info['max_length'] = self.max_length
            info['padding'] = self.padding
        return info


class Voc:
    def __init__(self, name, size, cols, store_dir):
        self.name: str = name
        self.size: int = size
        self.cols: List[Col] = cols
        self.store_dir = store_dir

    def __eq__(self, other):
        return self.name == other.name and self.size == other.size

    def get_info(self):
        return {
            'size': self.size,
            'cols': [col.name for col in self.cols]
        }

    def export(self, store_dir):
        from .vocab import Vocab
        vocab = Vocab(name=self.name).load(self.store_dir)
        vocab.save(store_dir)

    def merge(self, other):
        cols = self.cols.copy()
        for col in other.cols:
            for _col in cols:
                if col.name == _col.name:
                    break
            else:
                cols.append(col)
        return Voc(
            name=self.name,
            size=self.size,
            cols=cols,
            store_dir=self.store_dir
        )


class Meta:
    VER = 'UniDep-2.0'

    def __init__(self, store_dir):
        self.store_dir = store_dir
        self.path = os.path.join(self.store_dir, 'meta.data.json')

        data = self.load()
        self.version = data['version']
        self.cols = data.get('cols') or data['col_info']
        self.vocs = data.get('vocs') or data['vocab_info']
        self.id_col = data['id_col']

        # build col-voc graph
        self.cols = {col: Col(**self.cols[col], name=col) for col in self.cols}
        self.vocs = {voc: Voc(**self.vocs[voc], name=voc, store_dir=self.store_dir) for voc in self.vocs}

        # connect class objects
        for col in self.cols.values():
            col.voc = self.vocs[col.voc]
        for voc in self.vocs.values():
            voc.cols = [self.cols[col] for col in voc.cols]

        self.version_check()

    @staticmethod
    def parse_version(version):
        if version.startswith('UniDep-'):
            return version[7:]
        return f'0.{version}'

    def get_info(self):
        return {
            'version': Meta.VER,
            'id_col': self.id_col,
            'cols': {col.name: col.get_info() for col in self.cols.values()},
            'vocs': {voc.name: voc.get_info() for voc in self.vocs.values()}
        }

    def load(self) -> dict:
        return json.load(open(self.path))

    def save(self):
        json.dump(self.get_info(), open(os.path.join(self.store_dir, 'meta.data.json'), 'w'), indent=2)

    def version_check(self):
        current_version = self.parse_version(Meta.VER)
        depot_version = self.parse_version(self.version)

        if current_version != depot_version:
            warnings.warn(
                f'meta version of depot ({self.store_dir}) mismatch, '
                f'current version: {current_version}, '
                f'depot version: {depot_version}. '
                f'It may cause unexpected error.')

        if current_version <= depot_version:
            return

        command = input('Press Y to upgrade meta data for future use (Y/n): ')
        if command.lower() == 'y':
            os.rename(self.path, self.path + '.bak')
            print('Old meta data backed up to {}.'.format(self.path + '.bak'))
            self.save()
            print('Meta data upgraded.')

    @property
    def col_info(self):
        warnings.warn('col_info is deprecated, use cols instead.'
                      '(meta.col_info -> meta.cols)', DeprecationWarning)
        return self.cols

    @property
    def vocab_info(self):
        warnings.warn('vocab_info is deprecated, use vocs instead.'
                      '(meta.vocab_info -> meta.vocs)', DeprecationWarning)
        return self.vocs
