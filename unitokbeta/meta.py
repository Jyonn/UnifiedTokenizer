import json
import os
from datetime import datetime

from unitokbeta import warning
from unitokbeta.job import Job
from unitokbeta.tokenizer import TokenizerHub
from unitokbeta.tokenizer.union_tokenizer import UnionTokenizer
from unitokbeta.tokenizer.unknown_tokenizer import UnknownTokenizer
from unitokbeta.utils import Symbols
from unitokbeta.utils.handler import JsonHandler
from unitokbeta.utils.class_pool import ClassPool
from unitokbeta.utils.index_set import VocabSet, TokenizerSet, JobSet
from unitokbeta.vocabulary import Vocab, VocabHub


class Meta:
    version = 'unidep-v4beta'

    def __init__(self):
        self.note = ('Not compatible with unitok-v3 or lower version, '
                     'please upgrade by `pip install unitok>4.0.0` to load the data.')
        self.website = 'https://unitok.github.io'
        self.modified_at = self.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.vocabularies = VocabSet()
        self.tokenizers = TokenizerSet()
        self.jobs = JobSet()

    @staticmethod
    def parse_vocabulary(name: str, **kwargs):
        return Vocab(name)

    @staticmethod
    def parse_tokenizer(tokenizer_id: str, classname: str, vocab: str, params: dict):
        tokenizer_classes = ClassPool.tokenizers()

        if not VocabHub.has(vocab):
            raise ValueError(f"(unitok.meta) Vocabulary {vocab} not found in the vocabulary hub.")
        vocab = VocabHub.get(vocab)

        if (classname not in tokenizer_classes or
                classname in [UnknownTokenizer.get_classname(), UnionTokenizer.get_classname()]):
            warning(f"(unitok.meta) Tokenizer class {classname} not found in the class hub.")
            return UnknownTokenizer(tokenizer_id=tokenizer_id, classname=classname, vocab=vocab, **params)
        return tokenizer_classes[classname](tokenizer_id=tokenizer_id, vocab=vocab, **params)

    @staticmethod
    def parse_job(name: str, column: str, tokenizer: str, truncate: int, order: int, key: bool, max_len: int):
        if not TokenizerHub.has(tokenizer):
            raise ValueError(f"(unitok.meta) Tokenizer {tokenizer} not found in the tokenizer hub.")
        tokenizer = TokenizerHub.get(tokenizer)

        if column == str(Symbols.idx):
            column = Symbols.idx

        return Job(
            name=name,
            column=column,
            tokenizer=tokenizer,
            truncate=truncate,
            order=order,
            key=key,
            max_len=max_len,
        )

    @staticmethod
    def parse_version(version):
        if version.startswith('unidep-v'):
            return version[8:]

        if version.startswith('UniDep-'):
            raise ValueError(f'UniDep version ({version}) is not supported. '
                             f'Please downgrade the unitok version by `pip install unitok==3.5.3`, '
                             f'or use `unidep-upgrade-v4` to upgrade the version.')

        raise ValueError(f'UniDep version ({version}) is not supported. '
                         f'Please downgrade the unitok version by `pip install unitok==3.5.3` for compatible upgrade, '
                         f'and then install the latest unitok version, '
                         f'following the use of `unidep-upgrade-v4` to upgrade the version.')

    @classmethod
    def filename(cls, save_dir):
        return os.path.join(save_dir, 'meta.json')

    @classmethod
    def _deprecated_filename(cls, save_dir):
        return os.path.join(save_dir, 'meta.data.json')

    @classmethod
    def _compatible_readfile(cls, save_dir):
        filename = cls.filename(save_dir)
        if not os.path.exists(filename):
            filename = cls._deprecated_filename(save_dir)
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Meta file not found in {save_dir}")

        meta_data = json.load(open(filename))

        if 'version' not in meta_data:
            raise ValueError(f"Version not found in the meta file {filename}")

        current_version = cls.parse_version(cls.version)
        depot_version = cls.parse_version(meta_data.get('version'))

        if current_version != depot_version:
            warning('Version mismatch, unexpected error may occur.')

        return meta_data

    @classmethod
    def load(cls, save_dir):
        kwargs = cls._compatible_readfile(save_dir)

        meta = cls()
        meta.created_at = kwargs.get('created_at')
        meta.vocabularies = {cls.parse_vocabulary(**v).load(save_dir) for v in kwargs.get('vocabularies')}
        meta.tokenizers = {cls.parse_tokenizer(**t) for t in kwargs.get('tokenizers')}
        meta.jobs = {cls.parse_job(**j) for j in kwargs.get('jobs')}

        return meta

    def json(self):
        return {
            "version": self.version,
            "note": self.note,
            "website": self.website,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "vocabularies": [v.json() for v in self.vocabularies],
            "tokenizers": [t.json() for t in self.tokenizers],
            "jobs": [j.json() for j in self.jobs],
        }

    def save(self, save_dir):
        filename = self.filename(save_dir)
        JsonHandler.save(self.json(), filename)
