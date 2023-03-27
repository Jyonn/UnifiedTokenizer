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
            'name': self.name,
            'voc': self.voc.name,
        }
        if self.list:
            info['max_length'] = self.max_length
            info['padding'] = self.padding
        return info


class Voc:
    def __init__(self, name, size, cols):
        self.name: str = name
        self.size: int = size
        self.cols: List[Col] = cols

    def __eq__(self, other):
        return self.name == other.name and self.size == other.size

    def get_info(self):
        return {
            'name': self.name,
            'size': self.size,
            'cols': [col.name for col in self.cols]
        }


class Meta:
    VER = 'UniDep-2.0'

    def __init__(self, version, id_col, col_info=None, vocab_info=None, cols=None, vocs=None):
        self.version = version

        self.cols = cols or col_info
        self.vocs = vocs or vocab_info
        self.id_col = id_col

        # build col-voc graph
        self.cols = {col: Col(**self.cols[col], name=col) for col in self.cols}
        self.vocs = {voc: Voc(**self.vocs[voc], name=voc) for voc in self.vocs}

        # connect class objects
        for col in self.cols.values():
            col.voc = self.vocs[col.voc]
        for voc in self.vocs.values():
            voc.cols = [self.cols[col] for col in voc.cols]

        self.upgrade = self.version_check()

    @staticmethod
    def parse_version(version):
        if version.startswith('UniDep-'):
            return version[7:]
        return version

    def get_info(self):
        return {
            'version': Meta.VER,
            'id_col': self.id_col,
            'cols': {col.name: col.get_info() for col in self.cols.values()},
            'vocs': {voc.name: voc.get_info() for voc in self.vocs.values()}
        }

    def save(self, store_dir):
        json.dump(self.get_info(), open(os.path.join(store_dir, 'meta.json'), 'w'), indent=2)

    def version_check(self):
        current_version = self.parse_version(Meta.VER)
        depot_version = self.parse_version(self.version)

        if current_version != depot_version:
            warnings.warn(
                'Meta version mismatch, '
                'current version: {}, '
                'depot version: {}. '
                'It may cause unexpected error.'.format(
                    current_version, depot_version
                ))

        if current_version <= depot_version:
            return

        command = input('Press Y to upgrade meta data for future use (Y/n): ')
        if command.lower() != 'y':
            return True
