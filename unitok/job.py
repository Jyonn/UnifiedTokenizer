from unitok.tokenizer.union_tokenizer import UnionTokenizer

from unitok.tokenizer import BaseTokenizer
from unitok.utils import Symbols, Instance
from unitok.utils.hub import Hub


class Job:
    def __init__(
            self,
            tokenizer: BaseTokenizer,
            column: str,
            name: str = None,
            truncate: int = None,
            order: int = -1,
            key: bool = False,
            max_len: int = 0,
    ):
        self.tokenizer: BaseTokenizer = tokenizer
        self.column: str = column
        self.name: str = name
        self.truncate: int = truncate
        self.order: int = order
        self.slice: slice = self.get_slice(truncate)
        self.key: bool = key
        self.max_len = max_len
        self.from_union = isinstance(self.tokenizer, UnionTokenizer)

        JobHub.add(self.name, self)

    @property
    def return_list(self):
        return self.truncate is not None

    def clone(self, **kwargs):
        attributes = {'tokenizer', 'column', 'name', 'truncate', 'order', 'key', 'max_len'}
        params = dict()
        for attr in attributes:
            params[attr] = kwargs[attr] if attr in kwargs else getattr(self, attr)

        return Job(**params)

    def __str__(self):
        if self.key:
            return f'Job({self.column} => {self.name}) [PK]'
        return f'Job({self.column} => {self.name})'

    def __repr__(self):
        return str(self)

    @property
    def is_processed(self):
        return self.order >= 0

    def json(self):
        column = str(Symbols.idx) if self.column is Symbols.idx else self.column
        return {
            'name': self.name,
            'column': column,
            'tokenizer': self.tokenizer.get_tokenizer_id(),
            'truncate': self.truncate,
            'order': self.order,
            'key': self.key,
            'max_len': self.max_len,
        }

    @staticmethod
    def get_slice(truncate):
        if truncate is None:
            truncate = 0
        if truncate > 0:
            return slice(0, truncate)
        if truncate < 0:
            return slice(truncate, None)
        return slice(None)


class JobHub(Hub[Job]):
    _instance = Instance(compulsory_space=True)
