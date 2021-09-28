# Unified Tokenizer

## Introduction

When dealing with textual information used in some models (e.g. Bert), the first step is the tokenization. [**Transformers**](https://github.com/huggingface/transformers) provides _BertTokenizer_ to split words (multi-lingual) with _WordPiece_ algorithm. However, in some cases, some texts may be entities which is not required to split, although this entity contains many sub-words; some are the arrays of entities, joined by some characters (e.g. `|`, `,`). 

**Different cases require different tokenizer.** So it comes **Unified Tokenizer** (or **UniTok**). You can either customize your tokenizer, or use our pre-defined tokenizers.

## Installation

`pip install UniTok`

## Usage

Here we use the first 10 lines of the training set of [MINDlarge](https://msnews.github.io/) as an example (see `news-sample.tsv` file). Assume the data path is `/home/ubuntu/news-sample.tsv`.

### Data Declaration (more info see [MIND GitHub](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md))

Each line in the file is the information about one piece of news.

It has 7 columns, which are divided by the tab symbol:

- News ID
- Category
- SubCategory
- Title
- Abstract
- URL
- Title Entities (entities contained in the title of this news)
- Abstract Entities (entites contained in the abstract of this news) 

We only use its first 5 columns for demonstration.

### Imports

```python
import pandas as pd


from UniTok import UniTok, Column
from UniTok.tok import IdTok, EntTok, BertTok, SingT, ListT
```

### Read data

```python
df = pd.read_csv(
    filepath_or_buffer='/home/ubuntu/news-sample.tsv',
    sep='\t',
    names=['nid', 'cat', 'subCat', 'title', 'abs', 'url', 'titEnt', 'absEnt'],
    usecols=['nid', 'cat', 'subCat', 'title', 'abs'],
)
```

### Initialize Tokenizers

```python
id_tok = IdTok(name='news')  # for news id
cat_tok = EntTok(name='cat')  # for category & subcategory
txt_tok = BertTok(name='english', vocab_dir='bert-base-uncased')  # for title & abstract
cat_tok.vocab.reserve(100)  # first 100 tokens are reserved for some special usage in the downstream model, if any, and please be reminded that the first token is always PAD
```

### Construct Unified Tokenizer

_SingleTokenizer_ means it only omits one token id, while _ListTokenizer_ generates a sequence of ids.

```python
ut = UniTok()
ut.add_col(Column(
    name='nid',
    tokenizer=id_tok.as_sing(),
)).add_col(Column(
    name='cat',
    tokenizer=cat_tok.as_sing()
)).add_col(Column(
    name='subCat',
    tokenizer=cat_tok.as_sing(),
)).add_col(Column(
    name='title',
    tokenizer=txt_tok.as_list(),
)).add_col(Column(
    name='abs',
    tokenizer=txt_tok.as_list(),
)).read_file(df)
```

Here we leave the _max_length_ of the output of the _ListTokenizer_ behind.

### Analyse Data

```python
ut.analyse()
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
ut = UniTok()
ut.add_col(Column(
    name='nid',
    tokenizer=id_tok.as_sing(),
)).add_col(Column(
    name='cat',
    tokenizer=cat_tok.as_sing()
)).add_col(Column(
    name='subCat',
    tokenizer=cat_tok.as_sing(),
)).add_col(Column(
    name='title',
    tokenizer=txt_tok.as_list(max_length=10),
)).add_col(Column(
    name='abs',
    tokenizer=txt_tok.as_list(max_length=30),
)).read_file(df)
```

In this step, we set _max_length_ of each column. If _max_length_ is not set, we will keep the **whole** sequence and not truncate it.

### Tokenize and Store

```python
ut.tokenize()
ut.store_data('TokenizedData')
```
