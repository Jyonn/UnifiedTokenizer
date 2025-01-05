import pickle

from typing import Protocol, cast


class SupportsWrite(Protocol):
    def write(self, __s: bytes) -> object:
        ...


class PickleHandler:
    @staticmethod
    def load(path: str):
        return pickle.load(open(path, "rb"))

    @staticmethod
    def save(data: any, path: str):
        with open(path, "wb") as f:
            pickle.dump(data, cast(SupportsWrite, f))
