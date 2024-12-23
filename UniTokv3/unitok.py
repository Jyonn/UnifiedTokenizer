import json
import os
import tempfile
import warnings
from typing import Optional, Type, Dict, Union

import numpy as np
import pandas as pd

from .cols import Cols
from .column import Column, IndexColumn
from .tok import BaseTok, BertTok, EntTok, IdTok
from .vocab import Vocab
from .vocabs import Vocabs


class UniTok:
    """
    Unified Tokenizer, which can be used to tokenize different types of data in a DataFrame.

    Example:
        >>> import pandas as pd
        >>> from UniTokv3 import UniTok, Column, Vocab
        >>>
        >>> # load data
        >>> df = pd.read_csv(
        ... filepath_or_buffer='news-sample.tsv',
        ... sep='\t',
        ... names=['nid', 'cat', 'subCat', 'title', 'abs', 'url', 'titEnt', 'absEnt'],
        ... usecols=['nid', 'cat', 'subCat', 'title', 'abs'],
        ... )
        >>>
        >>> # define tokenizers
        >>> id_tok = IdTok(name='nid')
        >>> cat_tok = EntTok(name='cat')
        >>> text_tok = BertTok(name='eng', vocab_dir='bert-base-uncased')
        >>>
        >>> # define UniTok
        >>> tok = UniTok().add_index_col(name='nid').add_col(Column(
        ...     name='cat',
        ...     tok=cat_tok,
        ... )).add_col(Column(
        ...     name='subCat',
        ...     tok=cat_tok,
        ... ))add_col(Column(
        ...     name='title',
        ...     tok=text_tok,
        ...     max_length=20,
        ... )).add_col(Column(
        ...     name='abs',
        ...     tok=text_tok,
        ...     max_length=30,
        ... ))
        >>>
        >>> # tokenize
        >>> tok.read_file(df).tokenize().store_data('news-sample')
    """
    VER = 'v3.0'

    def __init__(self):
        self.cols = Cols()  # type: Union[Dict[str, Column], Cols]
        self.vocabs = Vocabs()  # type: Union[Dict[str, Vocab], Vocabs]
        self.id_col = None  # type: Optional[Column]
        self.data = None  # type: Optional[pd.DataFrame]

        print(
            "UniTok-v4 is coming!\n"
            "Try the new version with:\n"
            "    `from unitok import UniTok`\n"
            "UniTok-v3 is still available, but will be deprecated in the future:\n"
            "    `from UniTokv3 import UniTok`\n"
            "Documentation: https://unitok.github.io"
        )

    @property
    def vocab_depots(self):
        warnings.warn('vocab_depot is deprecated, '
                      'use vocabs instead (will be removed in 4.x version)', DeprecationWarning)
        return self.vocabs

    def add_col(self, col: Union[Column, str], tok: Union[BaseTok, Type[BaseTok]] = None):
        """
        Declare a column in the DataFrame to be tokenized.
        """

        if isinstance(col, str):
            assert tok is not None, 'tok must be specified when col is a string'
            col = Column(tok, name=col)

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

    def read(self, df: pd.DataFrame):
        """
        Read data from a file
        """
        self.data = df
        return self

    def read_file(self, df, sep=None):
        warnings.warn('read_file is deprecated, use read instead '
                      '(will be removed in 4.x version)', DeprecationWarning)
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
        warnings.warn('unitok.store_data is deprecated, use store instead '
                      '(will be removed in 4.x version)', DeprecationWarning)
        self.store(store_dir)

    def store(self, store_dir):
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

        from UniTokv3 import UniDep
        meta_data = dict(
            version=UniDep.VER,
            vocs=self.vocabs.get_info(),
            cols=self.cols.get_info(),
            id_col=self.id_col.name,
        )
        json.dump(meta_data, open(os.path.join(store_dir, 'meta.data.json'), 'w'), indent=2)
        return self

    def to_unidep(self):
        from UniTokv3 import UniDep
        store_dir = tempfile.gettempdir()
        self.store(store_dir)
        return UniDep(store_dir)
