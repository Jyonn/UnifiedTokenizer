import pandas as pd
from transformers import BertTokenizer

from .tok import BaseTok


class BertTok(BaseTok):
    def __init__(self, name, vocab_dir):
        super(BertTok, self).__init__(name=name)
        self.tokenizer = BertTokenizer.from_pretrained(vocab_dir)
        self.vocab.extend(self.tokenizer.vocab)

    def t(self, obj) -> [int, list]:
        if pd.notnull(obj):
            ts = self.tokenizer.tokenize(obj)
            ids = self.tokenizer.convert_tokens_to_ids(ts)
        else:
            ids = []
        return ids

    def get_vocabs(self):
        return self.tokenizer.vocab
