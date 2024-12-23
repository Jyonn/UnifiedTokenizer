import abc
from typing import Iterable, TypeVar, Generic

T = TypeVar('T')


class IndexSet(abc.ABC, set, Generic[T]):
    def __init__(self, iterable: Iterable[T] = None):
        self._index = {}

        if iterable is None:
            super().__init__()
            return

        super().__init__(iterable)
        for obj in iterable:
            self._index[self._get_key(obj)] = obj

    @staticmethod
    def _get_key(obj: T):
        raise NotImplementedError

    # override set methods with indexing ability

    def add(self, obj: T):
        super().add(obj)
        self._index[self._get_key(obj)] = obj

    def remove(self, obj: T):
        super().remove(obj)
        del self._index[self._get_key(obj)]

    def discard(self, obj: T):
        super().discard(obj)
        self._index.pop(self._get_key(obj), None)

    def pop(self) -> T:
        obj = super().pop()
        del self._index[self._get_key(obj)]
        return obj

    def clear(self):
        super().clear()
        self._index.clear()

    def update(self, other: Iterable[T]):
        """ update is used to add multiple objects that ensure no key conflict """
        for obj in other:
            if obj not in self:
                if self.has(self._get_key(obj)):
                    raise ValueError(f'key conflict: {self._get_key(obj)}')
                self.add(obj)

    def merge(self, other: 'IndexSet[T]', **kwargs):
        """ merge is used to add multiple objects that filter out key conflict """
        for obj in other:
            if not self.has(self._get_key(obj)):
                self.add(obj)

    def has(self, key) -> bool:
        return key in self._index

    def get(self, key, **kwargs) -> T:
        if self.has(key):
            return self._index[key]
        if 'default' in kwargs:
            return kwargs['default']
        raise KeyError(key)

    def __getitem__(self, key) -> T:
        return self.get(key)
