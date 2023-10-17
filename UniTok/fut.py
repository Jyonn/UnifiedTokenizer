from typing import Callable, Iterable, Union

import pandas as pd
from .tok import IdTok, SeqTok, EntTok

from .unitok import UniTok
from .unidep import UniDep
from .cols import Cols


class Fut:
    """
    Fast Unified Tokenizer, ignore vocabs
    """

    def __init__(
            self,
            data: pd.DataFrame,
            *refers: UniDep,
            id_col: str = None,
            refer_cols: Union[Callable, Iterable] = None,
    ):
        self.data = data
        self.refers = refers

        self.id_col = self.get_id_col(id_col)
        self.refer_cols = self.pack_refer_cols(refer_cols)

        self.col_names = self.data.columns.tolist()
        self.vocabs = self.get_vocabs()
        self.col_to_vocab = self.get_col_to_vocab()

    @staticmethod
    def pack_refer_cols(refer_cols: Union[Callable, Iterable] = None):
        def wrapper(col):
            return col in refer_cols

        if refer_cols is None:
            return None
        if isinstance(refer_cols, Iterable):
            return wrapper
        return refer_cols

    def get_id_col(self, id_col: str):
        col_names = self.data.columns.tolist()
        if id_col:
            return id_col
        for refer in self.refers:
            if refer.id_col in col_names:
                return refer.id_col
        print('Warning: id_col not found, use the DataFrame index instead')
        return None

    def get_vocabs(self):
        vocabs = dict()
        if not self.refer_cols:
            for refer in self.refers:
                vocabs.update(refer.vocabs)
        else:
            for refer in self.refers:
                for col in refer.cols:
                    if self.refer_cols(col) and col in self.col_names:
                        col_obj = refer.cols[col]
                        vocabs[col_obj.voc.name] = col_obj.voc.vocab
        return vocabs

    def get_col_to_vocab(self):
        col_to_vocab = dict()
        if not self.refer_cols:
            for refer in self.refers:
                for col in refer.cols:
                    col_obj = refer.cols[col]
                    col_to_vocab[col_obj.name] = self.vocabs[col_obj.voc.name]
        else:
            for refer in self.refers:
                for col in refer.cols:
                    if self.refer_cols(col) and col in self.col_names:
                        col_obj = refer.cols[col]
                        col_to_vocab[col_obj.name] = self.vocabs[col_obj.voc.name]
        return col_to_vocab

    def construct(self):
        unitok = UniTok().read(self.data)

        if not self.id_col:
            unitok.add_index_col()
            unitok.id_col.data = self.data.index.tolist()
        else:
            vocab = self.col_to_vocab.get(self.id_col)
            tok = IdTok(vocab=vocab) if vocab else IdTok
            unitok.add_col(
                col=self.id_col,
                tok=tok
            )
            unitok.id_col.data = self.data[self.id_col].tolist()
        if not self.col_to_vocab.get(self.id_col):
            unitok.id_col.tok.vocab.reserve(unitok.id_col.data)

        for col in self.col_names:
            if col == self.id_col:
                continue

            is_list = isinstance(self.data[col][0], list)
            vocab = self.col_to_vocab.get(col)
            tok = SeqTok if is_list else EntTok
            tok = tok(vocab=vocab) if vocab else tok
            unitok.add_col(
                col=col,
                tok=tok,
            )
            unitok.cols[col].data = self.data[col].tolist()

            if not vocab:
                max_id = 0
                if is_list:
                    for ids in unitok.cols[col].data:
                        if ids:
                            max_id = max(max_id, max(ids))
                else:
                    max_id = max(unitok.cols[col].data)
                unitok.cols[col].tok.vocab.reserve(list(range(max_id + 1)))

        return unitok

    def store(self, store_dir):
        unitok = self.construct()
        unitok.store(store_dir)
        return unitok
