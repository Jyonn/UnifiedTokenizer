# Unified Tokenizer

## Introduction

Unified Tokenizer, shortly **UniTok**, 
offers pre-defined various tokenizers in dealing with textual data. 
It is a central data processing tool 
that allows algorithm engineers to focus more on the algorithm itself 
instead of tedious data preprocessing.

It incorporates the BERT tokenizer from the [transformers]((https://github.com/huggingface/transformers)) library, 
while it supports custom via the general word segmentation module (i.e., the `BaseTok` class).

## Installation

`pip install UnifiedTokenizer`

## Usage

We use the head of the training set of the [MINDlarge](https://msnews.github.io/) dataset as an example (see `news-sample.tsv` file).

### Data Declaration (more info see [MIND GitHub](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md))

Each line in the file is a piece of news, including 7 features, which are divided by the tab (`\t`) symbol:

- News ID
- Category
- SubCategory
- Title
- Abstract
- URL
- Title Entities (entities contained in the title of this news)
- Abstract Entities (entities contained in the abstract of this news) 

We only use its first 5 columns for demonstration.

### Pre-defined Tokenizers

| Tokenizer | Description                                                           | Parameters  |
|-----------|-----------------------------------------------------------------------|-------------|
| BertTok   | Provided by the ``transformers` library, using the WordPiece strategy | `vocab_dir` |
| EntTok    | The column data is regarded as an entire token                        | /           |
| IdTok     | A specific version of EntTok, required to be identical                | /           |
| SplitTok  | Tokens are joined by separators like tab, space                       | `sep`       |

### Imports

```python
import pandas as pd


from UniTok import UniTok, Column
from UniTok.tok import IdTok, EntTok, BertTok
```

### Read data

```python
df = pd.read_csv(
    filepath_or_buffer='path/news-sample.tsv',
    sep='\t',
    names=['nid', 'cat', 'subCat', 'title', 'abs', 'url', 'titEnt', 'absEnt'],
    usecols=['nid', 'cat', 'subCat', 'title', 'abs'],
)
```

### Construct UniTok

```python
from UniTok import UniTok, Column
from UniTok.tok import EntTok, BertTok

cat_tok = EntTok(name='cat')  # one tokenizer for both cat and subCat
text_tok = BertTok(name='english', vocab_dir='bert-base-uncased')  # specify the bert vocab

unitok = UniTok().add_index_col(
    name='nid'
).add_col(Column(
    name='cat',
    tokenizer=cat_tok.as_sing()
)).add_col(Column(
    name='subCat',
    tokenizer=cat_tok.as_sing(),
)).add_col(Column(
    name='title',
    tokenizer=text_tok.as_list(),
)).add_col(Column(
    name='abs',
    tokenizer=text_tok.as_list(),
)).read_file(df)
```

### Analyse Data

```python
unitok.analyse()
```

It shows the distribution of the length of each column (if using _ListTokenizer_). It will help us determine the _max_length_ of the tokens for each column.

```
[ COLUMNS ]
[ COL: nid ]
[NOT ListTokenizer]

[ COL: cat ]
[NOT ListTokenizer]

[ COL: subCat ]
[NOT ListTokenizer]

[ COL: title ]
[ MIN: 6 ]
[ MAX: 16 ]
[ AVG: 12 ]
[ X-INT: 1 ]
[ Y-INT: 0 ]
       |   
       |   
       |   
       |   
       || |
       || |
       || |
| |  | || |
| |  | || |
| |  | || |
-----------

[ COL: abs ]
100%|██████████| 10/10 [00:00<00:00, 119156.36it/s]
100%|██████████| 10/10 [00:00<00:00, 166440.63it/s]
100%|██████████| 10/10 [00:00<00:00, 164482.51it/s]
100%|██████████| 10/10 [00:00<00:00, 2172.09it/s]
100%|██████████| 10/10 [00:00<00:00, 1552.30it/s]
[ MIN: 0 ]
[ MAX: 46 ]
[ AVG: 21 ]
[ X-INT: 1 ]
[ Y-INT: 0 ]
|                                              
|                                              
|                                              
|                                              
|                                              
|               | | ||    ||               |  |
|               | | ||    ||               |  |
|               | | ||    ||               |  |
|               | | ||    ||               |  |
|               | | ||    ||               |  |
-----------------------------------------------

[ VOCABS ]
[ VOC: news with  10 tokens ]
[ COL: nid ]

[ VOC: cat with  112 tokens ]
[ COL: cat, subCat ]

[ VOC: english with  30522 tokens ]
[ COL: title, abs ]
```

### ReConstruct Unified Tokenizer

```python
unitok = UniTok().add_index_col(
    name='nid'
).add_col(Column(
    name='cat',
    tokenizer=cat_tok.as_sing()
)).add_col(Column(
    name='subCat',
    tokenizer=cat_tok.as_sing(),
)).add_col(Column(
    name='title',
    tokenizer=text_tok.as_list(max_length=10),
)).add_col(Column(
    name='abs',
    tokenizer=text_tok.as_list(max_length=30),
)).read_file(df)
```

In this step, we set _max_length_ of each column. If _max_length_ is not set, we will keep the **whole** sequence and not truncate it.

### Tokenize and Store

```python
unitok.tokenize()
unitok.store_data('TokenizedData')
```
