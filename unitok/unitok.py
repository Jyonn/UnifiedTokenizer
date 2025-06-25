import os
import warnings
from typing import Union, Optional, cast, Callable

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from unitok.feature import Feature
from unitok.selector import Selector
from unitok.utils.verbose import info, warning
from unitok.meta import Meta
from unitok.status import Status
from unitok.tokenizer import BaseTokenizer, TokenizerHub, DigitTokenizer
from unitok.tokenizer.unknown_tokenizer import UnknownTokenizer
from unitok.utils import Symbols, Symbol, PickleHandler
from unitok.utils.hub import ParamHub


class UniTok(Status):
    idx = Symbols.idx

    def __init__(self):
        super().__init__()

        self.data = dict()
        self.meta = Meta()
        self.key_feature: Optional[Feature] = None
        self.save_dir = None

        # sample size is the number of rows in the table, while len(self) is the number of legal indices
        self._legal_indices = []
        self._legal_flags = []
        self._indices_is_init = False
        self._sample_size = None

        self._union_type = None
        self._soft_unions = dict()

    @property
    def key_job(self):
        warnings.warn('key_job is deprecated, use key_feat instead', DeprecationWarning, stacklevel=2)
        return self.key_feature

    @key_job.setter
    def key_job(self, value):
        warnings.warn('key_job is deprecated, use key_feat instead', DeprecationWarning, stacklevel=2)
        self.key_feature = value

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
            raise ValueError(f'Union type is already set: {self._union_type}')

    @Status.require_not_initialized
    def init_indices(self):
        self._indices_is_init = True
        self._sample_size = len(self.data[self.key_feature.name])
        self._legal_indices = list(range(self._sample_size))
        self._legal_flags = [True] * self._sample_size

    @classmethod
    def load(cls, save_dir: str, tokenizer_lib: str = None):
        with cls() as ut:
            ParamHub.add(Symbols.tokenizer, tokenizer_lib)
            ut.save_dir = save_dir
            ut.meta = Meta.load(save_dir)
            ut.data = PickleHandler.load(ut.filepath)

        for feature in ut.meta.features:
            if feature.key:
                if ut.key_feature is not None:
                    raise ValueError(f'multiple key features found: '
                                     f'{cast(Feature, ut.key_feature).name} and {feature.name}')
                ut.key_feature = feature

        if ut.key_feature is None:
            raise ValueError('key feature not found')

        ut.status = Symbols.tokenized
        ut.init_indices()

        return ut

    @Status.require_not_initialized
    def save(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.meta.save(self.save_dir)
        for vocab in self.meta.vocabularies:
            vocab.save(save_dir)
        PickleHandler.save(self.data, self.filepath)

    @property
    def filepath(self):
        return os.path.join(self.save_dir, 'data.pkl')

    def __enter__(self):
        from unitok.utils import Space
        Space.push(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from unitok.utils import Space
        Space.pop(self)

    @Status.require_initialized
    def add_index_feature(self, name: str = 'index', tokenizer: DigitTokenizer = None):
        tokenizer = tokenizer or DigitTokenizer(vocab=name)
        if not isinstance(tokenizer, DigitTokenizer):
            raise ValueError('tokenizer for index feature must be DigitTokenizer')
        return self.add_feature(
            tokenizer=tokenizer,
            column=self.idx,
            name=name,
            key=True,
        )

    def add_index_job(self, name: str = 'index', tokenizer: DigitTokenizer = None):
        warnings.warn('`add_index_job` is deprecated, use `add_job` instead', DeprecationWarning, stacklevel=2)
        return self.add_index_feature(name=name, tokenizer=tokenizer)

    def add_job(
            self,
            tokenizer: Union[str, BaseTokenizer],
            column: Union[str, Symbol] = None,
            name: str = None,
            truncate: int = None,
            key: bool = False,
    ):
        warnings.warn('`add_job` is deprecated, use `add_feature` instead', DeprecationWarning, stacklevel=2)
        return self.add_feature(
            tokenizer=tokenizer,
            column=column,
            name=name,
            truncate=truncate,
            key=key,
        )

    @Status.require_not_organized
    def add_feature(
            self,
            tokenizer: Union[str, BaseTokenizer],
            column: Union[str, Symbol] = None,
            name: str = None,
            truncate: int = None,
            key: bool = False,
    ):
        """
        Add tokenization feature
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

        feature = Feature(
            tokenizer=tokenizer,
            column=column,
            name=name,
            truncate=truncate,
            key=key,
        )

        self.meta.features.add(feature)
        self.meta.tokenizers.add(tokenizer)
        self.meta.vocabularies.add(tokenizer.vocab)

        if key:
            if tokenizer.return_list:
                raise AttributeError('Column content of the key feature should be tokenized into atomic value')
            if self.key_feature:
                raise ValueError(f'Key column already exists: {self.key_feature.name}')
            self.key_feature = feature

    @Status.require_not_organized
    def tokenize(self, df: pd.DataFrame):
        # TODO: in different times, the order of the primary key may be different, a sort operation is needed
        # validate whether each column exists in the dataframe
        for feature in self.meta.features:
            if feature.is_processed or feature.column == self.idx:
                continue
            if feature.column not in df.columns:
                raise ValueError(f'Column {feature.column} not found in dataframe')

        # add index to the index vocabulary
        if self.key_feature is None:
            raise ValueError(f'key key should be set')
        if self.key_feature.column is self.idx:
            self.key_feature.tokenizer(len(df) - 1)

        if self._indices_is_init and self._sample_size != len(df):
            raise ValueError(f'sample size mismatch: {self._sample_size} != {len(df)}')

        order_index = self.meta.features.next_order()
        for feature in self.meta.features:
            info(f'Tokenizing feature: {feature.tokenizer} ({feature.column} -> {feature.name})')

            if feature.is_processed:  # already tokenized
                continue

            token_lines = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                if feature.column == self.idx:
                    value = i
                else:
                    value = row[feature.column]
                line = feature.tokenizer(value)
                if feature.tokenizer.return_list:
                    line = line[feature.slice]
                    feature.max_len = max(feature.max_len, len(line))
                token_lines.append(line)

            feature.order = order_index
            self.data[feature.name] = token_lines

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
            union_key = other.key_feature.name

        if not self.meta.features.has(union_key):
            raise KeyError(f'union key {union_key} not found in the table')

        current_feature = self.meta.features[union_key]
        other_feature = other.key_feature

        if not current_feature.is_processed or not other_feature.is_processed:
            raise ValueError('feature of union key should be processed')

        if not current_feature.tokenizer.vocab.equals(other_feature.tokenizer.vocab):
            raise ValueError(f'union key vocab mismatch: {current_feature.tokenizer.vocab} != {other_feature.tokenizer.vocab}')

        self.meta.vocabularies.merge(other.meta.vocabularies)
        self.meta.tokenizers.merge(other.meta.tokenizers)
        self.meta.features.merge(other.meta.features, key_feature=other.key_feature)

        if soft_union:
            """ Soft union, store the union relationship and union on the fly """
            if current_feature not in self._soft_unions:
                self._soft_unions[current_feature] = set()
            self._soft_unions[current_feature].add(other)
            return

        """ Hard union, union the tables directly """
        union_data = {feature.name: [] for feature in other.meta.features}

        for index in self.data[current_feature.name]:
            for feature in other.meta.features:
                union_data[feature.name].append(other.data[feature.name][index])

        for feature in other.meta.features:
            if feature is not other.key_feature:
                self.data[feature.name] = union_data[feature.name]

    @Status.require_not_initialized
    @Status.to_organized
    def replicate(self, feature: Union[Feature, str], new_name: str, lazy=False):
        if isinstance(feature, str):
            feature = self.meta.features[feature]

        if not feature.is_processed:
            raise ValueError(f'feature {feature.name} is not processed')

        if feature.from_union and self.is_soft_union:
            raise ValueError(f'feature {feature.name} is from a soft union, please use hard union or save-and-load the unitok.')

        new_feature = feature.clone(name=new_name)
        self.meta.features.add(new_feature)

        if lazy or not feature.return_list:
            self.data[new_feature.name] = self.data[feature.name]
        else:
            # deep copy the data
            self.data[new_feature.name] = [line.copy() for line in self.data[feature.name]]

    @Status.require_not_initialized
    def summarize(self):
        console = Console()

        # Prepare introduction header
        introduction_header = Text.assemble(
            (
                f"UniTok (v{self.meta.parse_version(Meta.version)}), "
                f"Data (v{self.meta.parse_version(self.meta.version)})\n",
                "bold cyan"),
            (f"Sample Size: {self._sample_size}\n", "green"),
            (f"ID Column: {self.key_feature.name}\n", "magenta"),
            style="dim"
        )

        # Create a table
        table = Table(title="Features", expand=True, title_style="bold yellow", show_lines=True)

        # Add columns to the table
        table.add_column("Tokenizer", justify="left", style="cyan", no_wrap=True)
        table.add_column("Tokenizer ID", justify="center", style="green")
        table.add_column("Column Mapping", justify="left", style="blue")
        table.add_column("Vocab", justify="left", style="magenta")
        table.add_column("Max Length", justify="center", style="yellow")

        # Add rows to the table
        for feature in self.meta.features:
            tokenizer: BaseTokenizer = feature.tokenizer

            # Gather feature information
            tokenizer_name = tokenizer.__class__.__name__
            if isinstance(tokenizer, UnknownTokenizer):
                tokenizer_name = f"{tokenizer.__class__.__name__} [{tokenizer.classname}]"
            tokenizer_id = tokenizer.get_tokenizer_id()
            column_mapping = f"{feature.column} -> {feature.name}"
            vocab_info = f"{tokenizer.vocab.name} (size={len(tokenizer.vocab)})"
            max_len = f"{abs(feature.max_len)}" if feature.return_list else "N/A"

            # Add row to the table
            table.add_row(tokenizer_name, str(tokenizer_id), column_mapping, vocab_info, max_len)

        # Combine introduction and table
        console.print(introduction_header)
        console.print(table)

    def _pack_soft_union(self, index):
        sample = dict()
        for feature in self.meta.features:
            if not feature.from_union:
                sample[feature.name] = self.data[feature.name][index]

        for feature in self._soft_unions:
            index = sample[feature.name]
            for ut in self._soft_unions[feature]:
                sample.update(ut[index])

        return sample

    def _pack_hard_union(self, index):
        sample = dict()
        for feature in self.meta.features:
            sample[feature.name] = self.data[feature.name][index]
        return sample

    @Status.require_not_initialized
    def pack(self, index):
        if self.is_soft_union:
            return self._pack_soft_union(index)
        return self._pack_hard_union(index)

    def _parse_index(self, index) -> [int, Optional[Selector]]:
        selector = None
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError('index should be tuple with 2 elements: (index, selector)')
            index, selector = index
            if not isinstance(selector, Selector):
                if not isinstance(selector, tuple):
                    selector = (selector,)
                selector = Selector(self.meta, *selector)

        if isinstance(index, str):
            # key_id is used
            index = self.key_feature.tokenizer.vocab[index]
            if not self._legal_flags[index]:
                raise ValueError(f'current sample has been filtered out: {index}')
        else:
            index = self._legal_indices[index]
        return index, selector

    @Status.require_not_initialized
    def __getitem__(self, index):
        """
        ut[123]: get sample by index
        ut[123, 'title']: get value by index and feature_name
        ut[123, ('title', 'abstract')]: get sample by index and selected feature_names
        ut[123, title_feature]: get value by index and feature
        ut[123, BertTokenizer]: get sample by index and tokenizer class
        ut[123, bert_tokenizer]: get sample by index and tokenizer instance

        ut['abc']: sample index can be replaced by key_id
        """
        index, selector = self._parse_index(index)
        sample = self.pack(index)

        if selector is None:
            return sample

        return selector(sample)

    @Status.require_not_initialized
    def select(self, sample, selector: Union[Selector, str, Feature, tuple]):
        if not isinstance(selector, Selector):
            if not isinstance(selector, tuple):
                selector = (selector,)
            selector = Selector(self.meta, *selector)
        return selector(sample)

    def __len__(self):
        return len(self._legal_indices)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        ut_name = self.save_dir or 'Runtime Instance'
        return f'UniTok({ut_name}, size={self._sample_size})'

    def __repr__(self):
        return str(self)

    @Status.require_not_initialized
    @Status.to_organized
    def filter(self, filter_func: Callable, feature: Optional[Union[Feature, str]] = None):
        if isinstance(feature, Feature):
            feature = feature.name

        _legal_indices = []
        _legal_flags = [False] * self._sample_size

        if feature is not None:
            for index in self._legal_indices:
                if filter_func(self.data[feature][index]):
                    _legal_indices.append(index)
                    _legal_flags[index] = True
        else:
            for sample in self:
                if filter_func(sample):
                    index = sample[self.key_feature.name]
                    _legal_indices.append(index)
                    _legal_flags[index] = True

        self._legal_indices = _legal_indices
        self._legal_flags = _legal_flags
        return self

    @Status.require_not_initialized
    def retruncate(self, feature: Union[Feature, str], truncate: int):
        if isinstance(feature, str):
            feature = self.meta.features[feature]

        if truncate == 0:
            warning(f'retruncate method with truncate=0 will do nothing')

        if abs(truncate) >= feature.max_len:
            warning(f'Feature {feature.name} has the max length of {feature.max_len}, which is shorter than truncate value')

        if not feature.return_list:
            raise ValueError(f'Feature {feature.name} does not return list, not applicable to the retruncate method.')

        if self.is_soft_union and feature.from_union:
            raise ValueError(f'Feature {feature.name} is a soft union feature, please use hard union or save-and-load the unitok.')

        slicer = feature.get_slice(truncate)
        max_len = 0

        series = []
        for i in range(len(self)):
            value = self.data[feature.name][i][slicer]
            if len(value) > max_len:
                max_len = len(value)
            series.append(value)

        feature.max_len = max_len
        self.data[feature.name] = series

    def remove_feature(self, feature: Union[Feature, str]):
        if isinstance(feature, str):
            feature = self.meta.features[feature]

        if feature.key:
            raise ValueError('key feature cannot be removed')

        self.meta.features.remove(feature)

        tokenizer = feature.tokenizer
        for j in self.meta.features:
            if j.tokenizer == tokenizer:
                break
        else:
            self.meta.tokenizers.remove(tokenizer)
            vocab = tokenizer.vocab
            for t in self.meta.tokenizers:
                if t.vocab == vocab:
                    break
            else:
                self.meta.vocabularies.remove(vocab)

        if feature.is_processed:
            self.data.pop(feature.name)

    def remove_job(self, feature: Union[Feature, str]):
        warnings.warn(f'`remove_job` is deprecated, use `remove_feature` instead.', DeprecationWarning, stacklevel=2)
        self.remove_feature(feature)
