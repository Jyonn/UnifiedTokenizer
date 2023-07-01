from typing import Type, Union

from tqdm import tqdm

from .global_setting import Global
from .analysis.lengths import Lengths
from .tok import IdTok, BaseTok
from .vocab import Vocab


class SeqOperator:
    def __init__(
            self,
            max_length: int = 0,
            padding: bool = False,
            slice_post: bool = False,
            pad_pre: bool = False,
            pad: int = 0,
    ):
        self.max_length = max_length
        self.slice_post = slice_post
        self.padding = padding
        self.pad_pre = pad_pre
        self.pad = pad

    def __call__(self, ids):
        if self.max_length > 0:
            ids = ids[-self.max_length:] if self.slice_post else ids[:self.max_length]
            if self.padding:
                pads = [self.pad] * (self.max_length - len(ids))
                if self.pad_pre:
                    ids = pads + ids
                else:
                    ids.extend(pads)
        return ids


class Column:
    """
    A column of data in a DataFrame.

    Args:
        name (str): The name of the column.
        tok (BaseTok): The tokenizer of the column.
        operator (SeqOperator): The operator of the column.
    """
    def __init__(self, tok: Union[BaseTok, Type[BaseTok]], name=None, operator: SeqOperator = None, **kwargs):
        self.tok = tok
        self.name = name or tok.vocab.name
        self.operator = operator

        if isinstance(tok, type):
            assert issubclass(tok, BaseTok)
            assert name is not None, 'name must be set when tok is a class'
            self.tok = tok(vocab=Vocab(name=name))

        if kwargs:
            if operator:
                raise ValueError('operator and kwargs cannot be set at the same time')
            self.operator = SeqOperator(
                max_length=kwargs.get('max_length', 0),
                padding=kwargs.get('padding', False),
                slice_post=kwargs.get('slice_post', False),
                pad_pre=kwargs.get('pad_pre', False),
                pad=kwargs.get('pad', 0),
            )

        self.list = bool(self.operator) or tok.return_list is True  # whether the column is a list-element column

        self.data = []
        self.lengths = Lengths()

    def tokenize(self, objs):
        """
        Tokenize the column.
        """
        self.data = []
        for obj in tqdm(objs, disable=Global.is_silence()):
            ids = self.tok(obj)
            if self.operator:
                ids = self.operator(ids)
            self.data.append(ids)

    def analyse(self, objs):
        """
        Analyse the column.
        """
        if self.list:
            self.lengths.clean()
            for obj in tqdm(objs, disable=Global.is_silence()):
                self.lengths.push(len(self.tok(obj)))
            self.lengths.summarize()
        else:
            # print('[ NOT list-element column ]')
            for obj in tqdm(objs, disable=Global.is_silence()):
                self.tok(obj)

    def get_info(self):
        """
        Get the information of the column.
        """
        if not self.list:
            return dict(
                vocab=self.tok.vocab.name,
            )

        if not self.operator or self.operator.max_length < 1:
            max_length = 0
            for ids in self.data:
                if len(ids) > max_length:
                    max_length = len(ids)
        else:
            max_length = self.operator.max_length

        padding = self.operator.padding if self.operator else False
        return dict(
            vocab=self.tok.vocab.name,
            max_length=max_length,
            padding=padding,
        )


class IndexColumn(Column):
    def __init__(self, name='index'):
        super().__init__(tok=IdTok(name=name))
