import pandas as pd
from transformers import BertTokenizer

from .tok import BaseTok


class BertTok(BaseTok):
    """
    Bert Tokenizer

    Args:
        name: name of the tokenizer
        vocab_dir: directory of the vocabulary

    Example:
        >>> from UniTok.tok.bert_tok import BertTok
        >>> tok = BertTok(name='text')
        >>> tok('Hello World!')  # [101, 7592, 2088, 999, 102]
    """
    return_list = True

    def __init__(self, name, vocab_dir='bert-base-uncased'):
        super(BertTok, self).__init__(name=name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=vocab_dir)
        self.vocab.extend(self.tokenizer.vocab)

    def t(self, obj) -> [int, list]:
        if pd.notnull(obj):
            ts = self.tokenizer.tokenize(obj)
            ids = self.tokenizer.convert_tokens_to_ids(ts)
        else:
            ids = []
        return ids
