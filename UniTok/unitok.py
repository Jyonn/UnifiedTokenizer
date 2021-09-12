import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from .column import Column
from .tok.bert_tok import BertTok
from .tok.entity_tok import EntTok
from .tok.id_tok import IdTok
from .tok.tokenizer import ListTokenizer, SingTokenizer
from UniTok.vocab.depot import VocabDepot
from UniTok.vocab.vocab import Vocab


class UniTok:
    def __init__(self):
        self.cols = dict()  # type: Dict[str, Column]
        self.vocab_depot = VocabDepot()
        self.data = None
        self.version = 'v2.0'

    def add_col(self, col: Column):
        self.cols[col.name] = col
        self.vocab_depot.append(col)
        return self

    def read_file(self, df, sep=None):
        if isinstance(df, str):
            use_cols = list(self.cols.keys())
            df = pd.read_csv(df, sep=sep, usecols=use_cols)
        self.data = df
        return self

    def analyse(self):
        print('[ COLUMNS ]')
        for col_name in self.cols:
            col = self.cols[col_name]  # type: Column
            print('[ COL:', col.name, ']')
            col.analyse(self.data[col_name])
            print()

        print('[ VOCABS ]')
        for vocab_name in self.vocab_depot.col_map:  # type: Vocab
            vocab = self.vocab_depot.depot[vocab_name]
            print('[ VOC:', vocab.name, 'with ', vocab.get_size(), 'tokens ]')
            print('[ COL:', ', '.join(self.vocab_depot.col_map[vocab_name]), ']')
            print()

    def tokenize(self):
        for col_name in self.cols:
            print('[ COL:', col_name, ']')
            col = self.cols[col_name]  # type: Column
            col.data = []
            col.tokenize(self.data[col_name])

    def store_data(self, store_dir):
        os.makedirs(store_dir, exist_ok=True)

        vocab_info = dict()
        for vocab_name in self.vocab_depot.depot:
            vocab = self.vocab_depot.depot[vocab_name]
            vocab.save(store_dir)
            vocab_info[vocab_name] = dict(
                size=vocab.get_size(),
                cols=self.vocab_depot.col_map[vocab_name],
            )

        data = dict()
        for col_name in self.cols:
            col = self.cols[col_name]
            data[col_name] = np.array(col.data, dtype=object)
        np.save(os.path.join(store_dir, 'data.npy'), data, allow_pickle=True)

        col_info = dict()
        for col_name in self.cols:
            col = self.cols[col_name]
            col_info[col_name] = dict(
                vocab=col.tok.vocab.name,
            )
            if isinstance(col.tokenizer, ListTokenizer):
                max_length = col.tokenizer.max_length
                if max_length < 1:
                    for ids in col.data:
                        if len(ids) > max_length:
                            max_length = len(ids)
                col_info[col_name].update(dict(
                    max_length=max_length,
                    padding=col.tokenizer.padding,
                ))

        meta_data = dict(
            version=self.version,
            vocab_info=vocab_info,
            col_info=col_info,
        )
        json.dump(meta_data, open(os.path.join(store_dir, 'meta.data.json'), 'w'))


if __name__ == '__main__':
    df = pd.read_csv(
        filepath_or_buffer='~/Data/MIND/MINDlarge/train/news.tsv',
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
        tokenizer=SingTokenizer(id_tok),
    )).add_col(Column(
        name='cat',
        tokenizer=SingTokenizer(cat_tok)
    )).add_col(Column(
        name='subCat',
        tokenizer=SingTokenizer(cat_tok),
    )).add_col(Column(
        name='title',
        tokenizer=ListTokenizer(txt_tok),
    )).add_col(Column(
        name='abs',
        tokenizer=ListTokenizer(txt_tok),
    )).read_file(df).tokenize()
    ut.store_data('MINDlarge_train')
