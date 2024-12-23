import glob
import importlib
import os
import sys

from unitok.utils.verbose import warning
from unitok.tokenizer import BaseTokenizer
from unitok.utils import Symbols
from unitok.utils.hub import ParamHub


class ClassPool:
    _tokenizers = None

    @staticmethod
    def get_official_tokenizer_path():
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tokenizer')

    @classmethod
    def tokenizers(cls, tokenizer_lib: str = None):
        if cls._tokenizers is not None:
            return cls._tokenizers

        official_tokenizer_path = cls.get_official_tokenizer_path()
        official_pool = ClassPool(
            base_class=BaseTokenizer,
            module_dir=official_tokenizer_path,
            module_type='Tokenizer',
        )

        if tokenizer_lib is None:
            tokenizer_lib = ParamHub.get(Symbols.tokenizer, default=None)
        if tokenizer_lib is not None:
            custom_pool = cls.custom_tokenizers(tokenizer_lib)
            official_pool.union(custom_pool)

        cls._tokenizers = official_pool
        return cls._tokenizers

    @staticmethod
    def custom_tokenizers(tokenizer_lib: str):
        tokenizer_lib = os.path.abspath(tokenizer_lib)
        return ClassPool(
            base_class=BaseTokenizer,
            module_dir=tokenizer_lib,
            module_type='Tokenizer',
        )

    def __init__(self, base_class, module_dir: str, module_type: str):
        """
        @param base_class: e.g., BaseTokenizer, BaseTokenizerConfig, BaseColumn
        @param module_dir: e.g., tokenizer, column
        @param module_type: e.g., Tokenizer, TokenizerConfig, Column
        """

        self._base_class = base_class
        self._module_dir = module_dir
        self._module_type = module_type.lower()

        self.module_base = os.path.dirname(module_dir)
        if self.module_base not in sys.path:
            sys.path.append(self.module_base)

        self._class_list = self.get_class_list()
        self._class_dict = dict()
        for class_ in self._class_list:
            name = class_.__name__.lower()  # type: str
            name = name.replace(self._module_type, '')
            self._class_dict[name] = class_

        self._reverse_class_dict = {v: k for k, v in self._class_dict.items()}

    def get_class_list(self):
        file_paths = glob.glob(f'{self._module_dir}/*_{self._module_type}.py')
        class_list = []

        for file_path in file_paths:
            file_name = os.path.basename(file_path).split('.')[0]
            relative_module_path = os.path.relpath(self._module_dir, self.module_base).replace(os.sep, ".")
            module_name = f"{relative_module_path}.{file_name}"

            module = importlib.import_module(module_name)

            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, self._base_class) and obj is not self._base_class:
                    class_list.append(obj)
        return class_list

    def __getitem__(self, name):
        return self._class_dict[name]

    def __contains__(self, name):
        return name in self._class_dict

    def union(self, other):
        self_classnames = set(self._class_dict.keys())
        other_classnames = set(other._class_dict.keys())

        for classname in other_classnames:
            if classname not in self_classnames:
                self._class_dict[classname] = other._class_dict[classname]
                self._reverse_class_dict[other._class_dict[classname]] = classname
            else:
                warning(f'(unitok.ClassPool) Class {classname} already exists in the class pool')

    def get_name(self, class_):
        return self._reverse_class_dict[class_]
