import os
from typing import Union, Optional, cast, Callable

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from unitokbeta import info, warning
from unitokbeta.meta import Meta
from unitokbeta.status import Status
from unitokbeta.tokenizer import BaseTokenizer, TokenizerHub, DigitTokenizer
from unitokbeta.tokenizer.unknown_tokenizer import UnknownTokenizer
from unitokbeta.utils import Symbols, Symbol, PickleHandler
from unitokbeta.job import Job
from unitokbeta.utils.hub import ParamHub


class UniTok(Status):
    idx = Symbols.idx

    def __init__(self):
        super().__init__()

        self.data = dict()
        self.meta = Meta()
        self.key_job: Optional[Job] = None

        # sample size is the number of rows in the table, while len(self) is the number of legal indices
        self._legal_indices = []
        self._legal_flags = []
        self._indices_is_init = False
        self._sample_size = None

        self._union_type = None
        self._soft_unions = dict()

    @property
    def is_soft_union(self):
        return self._union_type == Symbols.soft

    @property
    def is_hard_union(self):
        return self._union_type == Symbols.hard

    def set_union_type(self, soft_union: bool):
        union_type = Symbols.soft if soft_union else Symbols.hard
        if self._union_type is None:
            self._union_type = union_type
        elif self._union_type != union_type:
            raise ValueError(f'union type is already set: {self._union_type}')

    @Status.require_not_initialized
    def init_indices(self):
        self._indices_is_init = True
        self._sample_size = len(self.data[self.key_job.name])
        self._legal_indices = list(range(self._sample_size))
        self._legal_flags = [True] * self._sample_size

    @classmethod
    def load(cls, save_dir: str, tokenizer_lib: str = None):
        with cls() as ut:
            ParamHub.add(Symbols.tokenizer, tokenizer_lib)
            ut.meta = Meta.load(save_dir)
            ut.data = PickleHandler.load(cls.filepath(save_dir))

        for job in ut.meta.jobs:
            if job.key:
                if ut.key_job is not None:
                    raise ValueError(f'multiple key jobs found: {cast(Job, ut.key_job).name} and {job.name}')
                ut.key_job = job

        if ut.key_job is None:
            raise ValueError('key job not found')

        ut.status = Symbols.tokenized
        ut.init_indices()

        return ut

    @Status.require_not_initialized
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        self.meta.save(save_dir)

        for vocab in self.meta.vocabularies:
            vocab.save(save_dir)
        PickleHandler.save(self.data, self.filepath(save_dir))

    @staticmethod
    def filepath(save_dir: str):
        return os.path.join(save_dir, 'data.pkl')

    def __enter__(self):
        from unitokbeta.utils import Space
        Space.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from unitokbeta.utils import Space
        Space.unset()

    @Status.require_initialized
    def add_index_job(self, name: str = 'index', tokenizer: DigitTokenizer = None):
        tokenizer = tokenizer or DigitTokenizer(vocab=name)
        if not isinstance(tokenizer, DigitTokenizer):
            raise ValueError('tokenizer for index job must be DigitTokenizer')
        return self.add_job(
            tokenizer=tokenizer,
            column=self.idx,
            name=name,
            key=True,
        )

    @Status.require_not_organized
    def add_job(
            self,
            tokenizer: Union[str, BaseTokenizer],
            column: Union[str, Symbol] = None,
            name: str = None,
            truncate: int = None,
            key: bool = False,
    ):
        """
        Add tokenization job
        :param tokenizer: Tokenizer name
        :param column: Column name, it can be UniTok.idx to indicate the index column
        :param name: Export column name, default is column name
        :param truncate: Truncate sequence length, if truncate < 0, truncate from the end, if truncate = None, it is an atomic value
        :param key: Whether the export column is primary key
        """

        if column is self.idx:
            if name is None:
                raise ValueError('name must be set when column is UniTok.idx')

        # parameter normalization
        if isinstance(tokenizer, str):
            tokenizer = TokenizerHub.get(tokenizer)

        if column is None:
            column = tokenizer.vocab.name

        if name is None:
            name = column

        if tokenizer.return_list and truncate is None:
            truncate = 0

        if not tokenizer.return_list:
            if truncate is not None:
                warning(f'truncate ({truncate}) will be ignored for atomic value, as tokenizer does not return list')
            truncate = None

        job = Job(
            tokenizer=tokenizer,
            column=column,
            name=name,
            truncate=truncate,
            key=key,
        )

        self.meta.jobs.add(job)
        self.meta.tokenizers.add(tokenizer)
        self.meta.vocabularies.add(tokenizer.vocab)

        if key:
            if tokenizer.return_list:
                raise AttributeError('Column content of the key job should be tokenized into atomic value')
            if self.key_job:
                raise ValueError(f'key key already exists: {self.key_job.name}')
            self.key_job = job

    @Status.require_not_organized
    def tokenize(self, df: pd.DataFrame):
        # TODO: in different times, the order of the primary key may be different, a sort operation is needed
        # validate whether each column exists in the dataframe
        for job in self.meta.jobs:
            if job.is_processed or job.column == self.idx:
                continue
            if job.column not in df.columns:
                raise ValueError(f'Column {job.column} not found in dataframe')

        # add index to the index vocabulary
        if self.key_job is None:
            raise ValueError(f'key key should be set')
        if self.key_job.column is self.idx:
            self.key_job.tokenizer(len(df) - 1)

        if self._indices_is_init and self._sample_size != len(df):
            raise ValueError(f'sample size mismatch: {self._sample_size} != {len(df)}')

        order_index = self.meta.jobs.next_order()
        for job in self.meta.jobs:
            info(f'Tokenizing job: {job.tokenizer} ({job.column} -> {job.name})')

            if job.is_processed:  # already tokenized
                continue

            token_lines = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                if job.column == self.idx:
                    value = i
                else:
                    value = row[job.column]
                line = job.tokenizer(value)
                if job.tokenizer.return_list:
                    line = line[job.slice]
                    job.max_len = max(job.max_len, len(line))
                token_lines.append(line)

            job.order = order_index
            self.data[job.name] = token_lines

        self.status = Symbols.tokenized
        if not self._indices_is_init:
            self.init_indices()

        return self

    """
    UniTok table methods
    """

    @Status.require_not_initialized
    @Status.to_organized
    def union(self, other: 'UniTok', soft_union=True, union_key=None):
        """
        Union other UniTok table by its primary key
        :param other: UniTok table to union
        :param soft_union: Two tables are stored separately and the union is performed on the fly
        :param union_key: Key column to link two tables
        """

        self.set_union_type(soft_union)

        if union_key is None:
            union_key = other.key_job.name

        if not self.meta.jobs.has(union_key):
            raise KeyError(f'union key {union_key} not found in the table')

        current_job = self.meta.jobs[union_key]
        other_job = other.key_job

        if not current_job.is_processed or not other_job.is_processed:
            raise ValueError('job of union key should be processed')

        if current_job.tokenizer.vocab != other_job.tokenizer.vocab:
            raise ValueError(f'union key vocab mismatch: {current_job.tokenizer.vocab} != {other_job.tokenizer.vocab}')

        self.meta.vocabularies.merge(other.meta.vocabularies)
        self.meta.tokenizers.merge(other.meta.tokenizers)
        self.meta.jobs.merge(other.meta.jobs, key_job=other.key_job)

        if soft_union:
            """ Soft union, store the union relationship and union on the fly """
            if current_job not in self._soft_unions:
                self._soft_unions[current_job] = set()
            self._soft_unions[current_job].add(other)
            return

        """ Hard union, union the tables directly """
        union_data = {job.name: [] for job in other.meta.jobs}

        for index in self.data[current_job.name]:
            for job in other.meta.jobs:
                union_data[job.name].append(other.data[job.name][index])

        for job in other.meta.jobs:
            if job is not other.key_job:
                self.data[job.name] = union_data[job.name]

    @Status.require_not_initialized
    def summarize(self):
        console = Console()

        # Prepare introduction header
        introduction_header = Text.assemble(
            (f"UniTok ({self.meta.parse_version(self.meta.version)})\n", "bold cyan"),
            (f"Sample Size: {self._sample_size}\n", "green"),
            (f"ID Column: {self.key_job.name}\n", "magenta"),
            style="dim"
        )

        # Create a table
        table = Table(title="Jobs", expand=True, title_style="bold yellow", show_lines=True)

        # Add columns to the table
        table.add_column("Tokenizer", justify="left", style="cyan", no_wrap=True)
        table.add_column("Tokenizer ID", justify="center", style="green")
        table.add_column("Column Mapping", justify="left", style="blue")
        table.add_column("Vocab", justify="left", style="magenta")
        table.add_column("Max Length", justify="center", style="yellow")

        # Add rows to the table
        for job in self.meta.jobs:
            tokenizer: BaseTokenizer = job.tokenizer

            # Gather job information
            tokenizer_name = tokenizer.__class__.__name__
            if isinstance(tokenizer, UnknownTokenizer):
                tokenizer_name = f"{tokenizer.__class__.__name__} [{tokenizer.classname}]"
            tokenizer_id = tokenizer.get_tokenizer_id()
            column_mapping = f"{job.column} -> {job.name}"
            vocab_info = f"{tokenizer.vocab.name} (size={len(tokenizer.vocab)})"
            max_len = f"{abs(job.max_len)}" if job.return_list else "N/A"

            # Add row to the table
            table.add_row(tokenizer_name, str(tokenizer_id), column_mapping, vocab_info, max_len)

        # Combine introduction and table
        console.print(introduction_header)
        console.print(table)

    def pack(self, index):
        sample = dict()
        for job in self.meta.jobs:
            sample[job.name] = self.data[job.name][index]

        for job in self._soft_unions:
            index = sample[job.name]
            for ut in self._soft_unions[job]:
                sample.update(ut[index])
        return sample

    def _parse_index(self, index):
        selector = None
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError('index should be tuple with 2 elements: (index, selector)')
            index, selector = index

        if isinstance(index, str):
            # key_id is used
            index = self.key_job.tokenizer.vocab[index]
            if not self._legal_flags[index]:
                raise ValueError(f'current sample has been filtered out: {index}')
        else:
            index = self._legal_indices[index]
        return index, selector

    @Status.require_not_initialized
    def __getitem__(self, index):
        """
        ut[123]: get sample by index
        ut[123, 'title']: get value by index and job_name
        ut[123, ('title', 'abstract')]: get sample by index and selected job_names
        ut[123, title_job]: get value by index and job
        ut[123, BertTokenizer]: get sample by index and tokenizer class
        ut[123, bert_tokenizer]: get sample by index and tokenizer instance

        ut['abc']: sample index can be replaced by key_id
        """
        index, selector = self._parse_index(index)
        sample = self.pack(index)

        if selector is None:
            return sample

        if isinstance(selector, Job):
            selector = selector.name
        if isinstance(selector, str):
            return sample[selector]

    def get_sample_by_id(self, key_id):
        index = self.key_job.tokenizer.vocab[key_id]
        return self[index]

    def __len__(self):
        return len(self._legal_indices)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        warning('__str__ is deprecated, '
                'please use summarize() method to display UniTok table')
        return f'UniTok(size={self._sample_size})'

    def __repr__(self):
        return str(self)

    @Status.require_not_initialized
    @Status.to_organized
    def filter(self, filter_func: Callable, job: Optional[Union[Job, str]] = None):
        if isinstance(job, Job):
            job = job.name

        _legal_indices = []
        _legal_flags = [False] * self._sample_size

        if job is not None:
            for index in self._legal_indices:
                if filter_func(self.data[job][index]):
                    _legal_indices.append(index)
                    _legal_flags[index] = True
        else:
            for sample in self:
                if filter_func(sample):
                    index = sample[self.key_job.name]
                    _legal_indices.append(index)
                    _legal_flags[index] = True

        self._legal_indices = _legal_indices
        self._legal_flags = _legal_flags
        return self
