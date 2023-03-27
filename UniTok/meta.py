import warnings
from typing import List


class Col:
    def __init__(self, name, voc, max_length=None, padding=None):
        self.name: str = name
        self.voc: Voc = voc
        self.max_length = max_length
        self.padding = padding
        self.list = max_length is not None

    def __eq__(self, other):
        return self.name == other.name and self.voc.name == other.voc.name and self.max_length == other.max_length


class Voc:
    def __init__(self, name, size, cols):
        self.name: str = name
        self.size: int = size
        self.cols: List[Col] = cols

    def __eq__(self, other):
        return self.name == other.name and self.size == other.size


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

        self.version_check()

    def version_check(self):
        if self.version != Meta.VER:
            warnings.warn(
                'Meta version mismatch, '
                'current version: {}, '
                'depot version: {}. '
                'It may cause unexpected error.'.format(
                    Meta.VER, self.version
                ))
