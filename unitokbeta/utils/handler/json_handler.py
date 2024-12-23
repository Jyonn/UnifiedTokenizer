import json

from typing import Protocol, cast


class SupportsWrite(Protocol):
    def write(self, __s: str) -> object:
        ...


class JsonHandler:
    @staticmethod
    def load(filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def loads(s: str):
        return json.loads(s)

    @staticmethod
    def dumps(obj) -> str:
        return json.dumps(obj, indent=2, ensure_ascii=False)

    @staticmethod
    def save(obj, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(obj, cast(SupportsWrite, f), indent=2, ensure_ascii=False)
