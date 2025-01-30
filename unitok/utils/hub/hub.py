import abc
from typing import TypeVar, Generic

from unitok.utils import Instance


T = TypeVar('T')


class Hub(abc.ABC, Generic[T]):
    _instance: Instance

    @classmethod
    def add(cls, key, obj: T = None):
        instance = cls._instance.current()
        if key in instance and cls.notequal(instance[key], obj):
            raise ValueError(f'Conflict object declaration: {obj} and {instance[key]}')
        instance[key] = obj

    @classmethod
    def notequal(cls, a: T, b: T) -> bool:
        return a is not b

    @classmethod
    def get(cls, name: str, **kwargs) -> T:
        """
        Get an instance by name
        """
        instance = cls._instance.current()
        if 'default' in kwargs:
            return instance.get(name, kwargs['default'])
        return instance[name]

    @classmethod
    def has(cls, name: str) -> bool:
        """
        Check if a instance exists
        """
        instance = cls._instance.current()
        return name in instance

    @classmethod
    def list(cls):
        """
        List all instances
        """
        instance = cls._instance.current()
        return instance.keys()
