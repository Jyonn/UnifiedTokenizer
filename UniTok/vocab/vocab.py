import math
import os
from typing import Union, List

from smartify import E


@E.register(id_processor=E.idp_cls_prefix())
class VocabError:
    NotEditable = E('Vocab {} is not editable')
    NotEmptyForReserve = E('Vocab {} is not empty and not allowed reserve operation')


class Vocab:
    __VOCAB_ID = 0

    @classmethod
    def get_vocab_id(cls):
        vocab_id = cls.__VOCAB_ID
        cls.__VOCAB_ID += 1
        return vocab_id

    def __init__(self, name):
        self.name = name
        self.obj2index, self.index2obj = {}, {}
        self.editable = True
        self.vocab_id = Vocab.get_vocab_id()

    def extend(self, objs):
        for obj in objs:
            self.append(obj)
        return self

    def append(self, obj) -> int:
        if obj not in self.obj2index:
            if not self.editable:
                raise VocabError.NotEditable(self.name)
            index = len(self.index2obj)
            self.obj2index[obj] = index
            self.index2obj[index] = obj
        return self.obj2index[obj]

    def reserve(self, tokens: Union[int, List[any]]):
        if self.get_size():
            raise VocabError.NotEmptyForReserve(self.name)

        if isinstance(tokens, int):
            digits = int(math.log10(tokens))
            token_template = '[UNUSED%%0%sd]' % digits
            for token in range(tokens):
                self.append(token_template % token)
        else:
            for token in tokens:
                self.append(token)
        return self

    def get_tokens(self):
        return [self.index2obj[i] for i in range(len(self.index2obj))]

    def allow_edit(self):
        self.editable = True
        return self

    def deny_edit(self):
        self.editable = False
        return self

    def get_store_path(self, store_dir):
        return os.path.join(store_dir, 'tok.{}.dat'.format(self.name))

    def load(self, store_dir: str, as_path=False):
        store_path = store_dir if as_path else self.get_store_path(store_dir)

        self.obj2index, self.index2obj = {}, {}
        with open(store_path, 'r') as f:
            objs = f.read().split('\n')[:-1]
        for index, obj in enumerate(objs):
            self.obj2index[obj] = index
            self.index2obj[index] = obj

        return self

    def save(self, store_dir):
        store_path = self.get_store_path(store_dir)
        with open(store_path, 'w') as f:
            for i in range(len(self.index2obj)):
                f.write('{}\n'.format(self.index2obj[i]))
        return self

    def get_size(self):
        return len(self.index2obj)
