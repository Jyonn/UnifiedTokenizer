import json
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .cols import Cols
from .column import Column, IndexColumn
from .tok.bert_tok import BertTok
from .tok.entity_tok import EntTok
from .tok.id_tok import IdTok
from .vocab import Vocab
from .vocabs import Vocabs


class UniTok:
    """
    Unified Tokenizer, which can be used to tokenize different types of data in a DataFrame.
    """
    VER = 'v3.0'

    def __init__(self):
        self.cols = Cols()
        self.vocabs = Vocabs()
        self.id_col = None  # type: Optional[Column]
        self.data = None

    @property
    def vocab_depots(self):
        warnings.warn('vocab_depot is deprecated, '
                      'use vocabs instead (will be removed in 4.x version)', DeprecationWarning)
        return self.vocabs

    def add_col(self, col: Column):
        """
        Declare a column in the DataFrame to be tokenized.
        """
        if isinstance(col.tok, IdTok):
            if self.id_col:
                raise ValueError(f'already exists id column {self.id_col.name} before adding {col.name}')
            self.id_col = col

        self.cols[col.name] = col
        self.vocabs.append(col)

        return self

    def add_index_col(self, name='index'):
        """
        Declare a column in the DataFrame to be tokenized as index column.
        """
        if self.id_col:
            raise ValueError(f'already exists id column {self.id_col.name} before adding IndexColumn')

        col = IndexColumn(name=name)
        self.cols[col.name] = col
        self.vocabs.append(col)
        self.id_col = col
        return self

    def read_file(self, df, sep=None):
        """
        Read data from a file
        """
        if isinstance(df, str):
            use_cols = list(self.cols.keys())
            df = pd.read_csv(df, sep=sep, usecols=use_cols)
        self.data = df
        return self

    def __getitem__(self, col):
        """
        Get the data of a column
        """
        if isinstance(col, IndexColumn):
            return self.data.index
        if isinstance(col, Column):
            col = col.name
        return self.data[col]

    def analyse(self):
        """
        Analyse the data, including:
            1. length distribution of list-element columns
            2. frequency distribution of vocabularies
        """
        for vocab in self.vocabs.values():
            vocab.set_count_mode(True)

        print('[ COLUMNS ]')
        for col_name in self.cols:
            col = self.cols[col_name]  # type: Column
            print('[ COL:', col.name, ']')
            col.analyse(self[col])
            print()

        print('[ VOCABS ]')
        for vocab in self.vocabs.values():
            print('[ VOC:', vocab.name, 'with ', len(vocab), 'tokens ]')
            print('[ COL:', ', '.join(self.vocabs.cols[vocab.name]), ']')
            print('[ FRQ:', vocab.summarize(), ']')
            print()
        return self

    def tokenize(self):
        """
        Tokenize the data
        """
        if not self.id_col:
            raise ValueError('id column is not set')

        for vocab in self.vocabs.values():
            vocab.set_count_mode(False)

        for col_name in self.cols:
            print('[ COL:', col_name, ']')
            col = self.cols[col_name]  # type: Column
            col.data = []
            col.tokenize(self[col])
        return self

    def get_tok_path(self, col_name, store_dir):
        """
        Get the store path of the tokenizer of a column
        """
        warnings.warn('unitok.get_tok_path is deprecated (will be removed in 4.x version)', DeprecationWarning)
        return self.cols[col_name].tok.vocab.get_store_path(store_dir)

    def store_data(self, store_dir):
        """
        Store the tokenized data
        """
        os.makedirs(store_dir, exist_ok=True)

        for vocab in self.vocabs.values():  # type: Vocab
            vocab.save(store_dir)

        data = dict()
        for col in self.cols.values():  # type: Column
            data[col.name] = np.array(col.data, dtype=object)
        np.save(os.path.join(store_dir, 'data.npy'), data, allow_pickle=True)

        from UniTok import UniDep
        meta_data = dict(
            version=UniDep.VER,
            vocs=self.vocabs.get_info(),
            cols=self.cols.get_info(),
            id_col=self.id_col.name,
        )
        json.dump(meta_data, open(os.path.join(store_dir, 'meta.data.json'), 'w'))
        return self


if __name__ == '__main__':
    df = pd.read_csv(
        filepath_or_buffer='news-sample.tsv',
        sep='\t',
        names=['nid', 'cat', 'subCat', 'title', 'abs', 'url', 'titEnt', 'absEnt'],
        usecols=['nid', 'cat', 'subCat', 'title', 'abs'],
    )

    ut = UniTok()
    id_tok = IdTok(name='news')
    cat_tok = EntTok(name='cat')
    txt_tok = BertTok(name='english', vocab_dir='bert-base-uncased')
    cat_tok.vocab.reserve(100)

    ut.add_col(Column(
        name='nid',
        tok=id_tok,
    )).add_col(Column(
        name='cat',
        tok=cat_tok,
    )).add_col(Column(
        name='subCat',
        tok=cat_tok,
    )).add_col(Column(
        name='title',
        tok=txt_tok,
    )).add_col(Column(
        name='abs',
        tok=txt_tok,
    )).read_file(df).tokenize()
    ut.store_data('news-sample')
