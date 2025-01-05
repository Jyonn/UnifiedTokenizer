import os

from unitok import PickleHandler
from unitok.utils import Map, Instance
from unitok.utils.hub import Hub
from unitok.vocabulary.counter import Counter


class Vocabulary:
    """
    Vocabulary class for mapping object to index and vice versa.
    """

    def __init__(self, name: str):
        self._name = str(name)
        self.o2i, self.i2o = Map(), Map()

        self._editable = True  # whether vocab is editable
        self.counter = Counter()

        VocabularyHub.add(self.name, self)

    def equals(self, other: 'Vocabulary'):
        return self.name == other.name and len(self) == len(other)

    @property
    def name(self):
        return self._name

    # add name setter
    @name.setter
    def name(self, value):
        self._name = value

    """
    Basic Methods
    """

    def extend(self, objs):
        """
        extend vocab with iterable object
        :return: index list
        """
        return [self.append(obj) for obj in objs]

    def append(self, obj, oov_token=None):
        obj = str(obj)
        if obj not in self.o2i:
            if '\n' in obj:
                raise ValueError(f'token ({obj}) contains line break')

            if not self._editable:
                if oov_token is None:
                    raise ValueError(f'the fixed vocab {self.name} is not allowed to add new token ({obj})')
                return oov_token

            index = len(self)
            self.o2i[obj] = index
            self.i2o[index] = obj

        index = self.o2i[obj]
        self.counter(index)
        return index

    @property
    def size(self):
        return len(self)

    def __len__(self):
        return len(self.i2o)

    def __bool__(self):
        return True

    def __iter__(self):
        for i in range(len(self)):
            yield self.i2o[i]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.i2o[item]
        return self.o2i[item]

    def __str__(self):
        return f'Vocabulary({self.name}, vocab_size={len(self)})'

    """
    Editable Methods
    """

    def allow_edit(self):
        self._editable = True
        return self

    def deny_edit(self):
        self._editable = False
        return self

    def trim(self, min_count):
        valid_indices = self.counter.trim(min_count=min_count)
        valid_objs = [self.i2o[index] for index in valid_indices]

        self.o2i, self.i2o = Map(), Map()
        self.counter.deactivate()
        editable = self._editable
        self.allow_edit().extend(valid_objs)
        self._editable = editable

    def summarize(self, base=10):
        return self.counter.summarize(base=base)

    """
    Save & Load Methods
    """

    def filepath(self, store_dir):
        return os.path.join(store_dir, self.filename)

    @property
    def filename(self):
        return f'{self.name}.vocab'

    def load(self, save_dir: str):
        if not save_dir.endswith('.vocab'):
            save_dir = self.filepath(save_dir)

        self.o2i, self.i2o = {}, {}
        objs = PickleHandler.load(save_dir)
        for index, obj in enumerate(objs):
            self.o2i[obj] = index
            self.i2o[index] = obj

        return self

    def save(self, save_dir):
        store_path = self.filepath(save_dir)
        PickleHandler.save(list(self), store_path)

        return self

    def json(self):
        return {
            'name': self.name,
            'vocab_size': len(self),
        }


class VocabularyHub(Hub[Vocabulary]):
    _instance = Instance()
