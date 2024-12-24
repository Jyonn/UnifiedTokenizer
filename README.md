# UniTok V4

If you want to use the old version, please refer to [v3](README_v3.md) in Chinese.

## Overview

[![PyPI version](https://badge.fury.io/py/unitok.svg)](https://badge.fury.io/py/unitok)

Welcome to the UniTok documentation! 
This library provides a unified preprocessing solution for machine learning datasets, handling diverse data types like text, categorical features, and numerical values. 
It introduces **SQL-like** data table combinations and a modular workflow that transitions datasets through three states: `initialized`, `tokenized`, and `organized`.

UniTok is designed to simplify preprocessing by offering reusable components such as tokenizers and vocabularies, making it flexible for various datasets and scenarios.

## Road from V3 to V4

### Changes and Comparisons

| Feature           | UniTok v3                                                   | UniTok v4                                           | Comments                                                                      |
|-------------------|-------------------------------------------------------------|-----------------------------------------------------|-------------------------------------------------------------------------------|
| `UniTok` class    | Solely for tokenization                                     | Manages the entire preprocessing lifecycle          |                                                                               |
| `UniDep` class    | Data loading and combining                                  | Removed                                             | V4 combines the functionalities of `UniTok` and `UniDep` into a single class. |
| `Column` class    | Column name is for both the original and tokenized datasets | N/A                                                 | V4 introduces a `Job` class.                                                  |
| `Job` class       | N/A                                                         | Defines how a specific column should be tokenized   |                                                                               |
| `Tokenizer` class | Ambiguous return type definition                            | `return_list` parameter must be of type `bool`      |                                                                               |
| `Tokenizer` class | Only supports `BertTokenizer` for text processing           | Supports all Tokenizers in the transformers library | New `TransformersTokenizer` class                                             |
| `analyse` method  | Supported                                                   | Not supported Currently                             |                                                                               |
| `Meta` class      | Only for human-friendly displaying                          | Manager for `Job`, `Tokenizer`, and `Vocab`         |                                                                               |
| `unitok` command  | Visualization in the terminal                               | More colorful and detailed output                   |                                                                               | 

### How to Migrate the Processed Data

```bash
unidep-upgrade-v4 <path>
```

## Installation

**Requirements**

- Python 3.7 or later
- Dependencies:
  - pandas
  - transformers
  - tqdm
  - rich

**Install UniTok via pip**

```bash
pip install unitok
```

## Core Concepts

**States**

- `initialized`: The initial state after creating a UniTok instance.
- `tokenized`: Achieved after applying tokenization to the dataset.
- `organized`: Reached after combining multiple datasets via operations like union.

**Components**

- UniTok: Manages the dataset preprocessing lifecycle.
- Job: Defines how a specific column should be tokenized.
- Tokenizer: Encodes data using various methods (e.g., BERT, splitting by delimiters).
- Vocabulary: Stores and manages unique tokens across datasets.

**Primary Key (key_job)**

The `key_job` acts as the primary key for operations like `getitem` and `union`, ensuring consistency across datasets.

## Usage Guide

### Loading Data

Load datasets using pandas:

```python
import pandas as pd

item = pd.read_csv(
    filepath_or_buffer='news-sample.tsv',
    sep='\t',
    names=['nid', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
    usecols=['nid', 'category', 'subcategory', 'title', 'abstract'],
)
item['abstract'] = item['abstract'].fillna('')  # Handle missing values

user = pd.read_csv(
    filepath_or_buffer='user-sample.tsv',
    sep='\t',
    names=['uid', 'history'],
)

interaction = pd.read_csv(
    filepath_or_buffer='interaction-sample.tsv',
    sep='\t',
    names=['uid', 'nid', 'click'],
)
```

### Defining and Adding Jobs

Define tokenization jobs for different columns:

```python
from unitok import UniTok, Vocab
from unitok.tokenizer import BertTokenizer, TransformersTokenizer, EntityTokenizer, SplitTokenizer, DigitTokenizer

item_vocab = Vocab(name='nid')  # will be used across datasets
user_vocab = Vocab(name='uid')  # will be used across datasets

with UniTok() as item_ut:
    bert_tokenizer = BertTokenizer(vocab='bert')
    llama_tokenizer = TransformersTokenizer(vocab='llama', key='huggyllama/llama-7b')

    item_ut.add_job(tokenizer=EntityTokenizer(vocab=item_vocab), column='nid', key=True)
    item_ut.add_job(tokenizer=bert_tokenizer, column='title', name='title@bert', truncate=20)
    item_ut.add_job(tokenizer=llama_tokenizer, column='title', name='title@llama', truncate=20)
    item_ut.add_job(tokenizer=bert_tokenizer, column='abstract', name='abstract@bert', truncate=50)
    item_ut.add_job(tokenizer=llama_tokenizer, column='abstract', name='abstract@llama', truncate=50)
    item_ut.add_job(tokenizer=EntityTokenizer(vocab='category'), column='category')
    item_ut.add_job(tokenizer=EntityTokenizer(vocab='subcategory'), column='subcategory')

with UniTok() as user_ut:
    user_ut.add_job(tokenizer=EntityTokenizer(vocab=user_vocab), column='uid', key=True)
    user_ut.add_job(tokenizer=SplitTokenizer(vocab=item_vocab, sep=','), column='history', truncate=30)

with UniTok() as inter_ut:
    inter_ut.add_index_job(name='index')
    inter_ut.add_job(tokenizer=EntityTokenizer(vocab=user_vocab), column='uid')
    inter_ut.add_job(tokenizer=EntityTokenizer(vocab=item_vocab), column='nid')
    inter_ut.add_job(tokenizer=DigitTokenizer(vocab='click', vocab_size=2), column='click')
```

### Tokenizing Data

Tokenize and save the processed data:

```python
item_ut.tokenize(item).save('sample-ut/item')
item_vocab.deny_edit()  # will raise an error if new items are detected in the user or interaction datasets
user_ut.tokenize(user).save('sample-ut/user')
inter_ut.tokenize(interaction).save('sample-ut/interaction')
```

### Combining Datasets

Combine datasets using union:

```python
# => {'category': 0, 'nid': 0, 'title@bert': [1996, 9639, 3035, 3870, ...], 'title@llama': [450, 1771, 4167, 10470, ...], 'abstract@bert': [4497, 1996, 14960, 2015, ...], 'abstract@llama': [1383, 459, 278, 451, ...], 'subcategory': 0}
print(item_ut[0])

# => {'uid': 0, 'history': [0, 1, 2]}
print(user_ut[0])

# => {'uid': 0, 'nid': 7, 'index': 0, 'click': 1}
print(inter_ut[0])

with inter_ut:
    inter_ut.union(user_ut)

    # => {'index': 0, 'click': 1, 'uid': 0, 'nid': 7, 'history': [0, 1, 2]}
    print(inter_ut[0])
```

### Glance at the Terminal

```bash
unitok sample-ut/item
```

```text
UniTok (4beta)
Sample Size: 10
ID Column: nid

                                                                                 Jobs                                                                                  
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Tokenizer                            ┃     Tokenizer ID      ┃ Column Mapping                               ┃ Vocab                             ┃    Max Length     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ TransformersTokenizer                │      auto_2VN5Ko      │ abstract -> abstract@llama                   │ llama (size=32024)                │        50         │
├──────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────┼───────────────────┤
│ EntityTokenizer                      │      auto_C0b9Du      │ subcategory -> subcategory                   │ subcategory (size=8)              │        N/A        │
├──────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────┼───────────────────┤
│ TransformersTokenizer                │      auto_2VN5Ko      │ title -> title@llama                         │ llama (size=32024)                │        20         │
├──────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────┼───────────────────┤
│ EntityTokenizer                      │      auto_4WQYxo      │ category -> category                         │ category (size=4)                 │        N/A        │
├──────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────┼───────────────────┤
│ BertTokenizer                        │      auto_Y9tADT      │ abstract -> abstract@bert                    │ bert (size=30522)                 │        46         │
├──────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────┼───────────────────┤
│ BertTokenizer                        │      auto_Y9tADT      │ title -> title@bert                          │ bert (size=30522)                 │        16         │
├──────────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────┼───────────────────┤
│ EntityTokenizer                      │      auto_qwQALc      │ nid -> nid                                   │ nid (size=10)                     │        N/A        │
└──────────────────────────────────────┴───────────────────────┴──────────────────────────────────────────────┴───────────────────────────────────┴───────────────────┘
```

## Contributing

We welcome contributions to UniTok! We appreciate your feedback, bug reports, and pull requests.

Our TODO list includes:

- [ ] More detailed documentation
- [ ] More examples and tutorials
- [ ] More SQL-like operations
- [ ] Analysis and visualization tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
