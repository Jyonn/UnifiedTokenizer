# UniTok V3: 类SQL数据预处理工具包

Updated on Christmas Eve, 2024.

Even when you have installed UniTok V4, you can still use UniTok V3 by importing it as follows:
```python
from UniTokv3 import UniTok
```

Or you can install UniTok V3 by running:
```bash
pip install unitok==3.5.3
```

## 1. 简介

UniTok 是史上第一个类SQL的数据预处理工具包，提供了一整套的数据封装和编辑工具。

UniTok 主要包括两大组件：负责统一数据处理的`UniTok` 和 负责数据读取和二次编辑的`UniDep`：
- `UniTok` 通过分词器（Tokenizers）和数据列（Columns）等组件将生数据（Raw Data）进行分词与ID化操作，并最终以numpy数组格式存储为一张数据表。
- `UniDep` 读取由`UniTok`生成的数据表以及元数据（如词表信息），可以直接与Pytorch的Dataset结合使用，也可以完成二次编辑、和其他数据表合并、导出等操作。
- 在3.1.9版本后，我们推出`Fut` 组件，它是`UniTok`的替代品，可以更快速地完成数据预处理。

## 2. 安装

使用pip安装：

```bash
pip install unitok==3.5.3
```

## 3. 主要功能

### 3.1 UniTok

UniTok提供了一整套的数据预处理工具，包括不同类型的分词器、数据列的管理等。具体来说，UniTok 提供了多种类型的分词器，可以满足不同类型数据的分词需求。每个分词器都继承自 `BaseTok` 类。

此外，UniTok 提供了 `Column` 类来管理数据列。每个 `Column` 对象包含一个分词器（Tokenizer）和一个序列操作器（SeqOperator）。

我们以新闻推荐系统场景为例，数据集可能包含以下部分：

- 新闻内容数据`(news.tsv)`：每一行是一条新闻，包含新闻ID、新闻标题、摘要、类别、子类别等多个特征，用`\t`分隔。
- 用户历史数据`(user.tsv)`：每一行是一位用户，包含用户ID和用户历史点击新闻的ID列表，新闻ID用` `分隔。
- 交互数据：包含训练`(train.tsv)`、验证`(dev.tsv)`和测试数据`(test.tsv)`。每一行是一条交互记录，包含用户ID、新闻ID、是否点击，用`\t`分隔。

我们首先分析以上每个属性的数据类型：

| 文件        | 属性       | 类型  | 样例                                                                   | 备注                      |
|-----------|----------|-----|----------------------------------------------------------------------|-------------------------|
| news.tsv  | nid      | str | N1234                                                                | 新闻ID，唯一标识               |
| news.tsv  | title    | str | After 10 years, the iPhone is still the best smartphone in the world | 新闻标题，通常用BertTokenizer分词 |
| news.tsv  | abstract | str | The iPhone 11 Pro is the best smartphone you can buy right now.      | 新闻摘要，通常用BertTokenizer分词 |
| news.tsv  | category | str | Technology                                                           | 新闻类别，不可分割               |
| news.tsv  | subcat   | str | Mobile                                                               | 新闻子类别，不可分割              |
| user.tsv  | uid      | str | U1234                                                                | 用户ID，唯一标识               |
| user.tsv  | history  | str | N1234 N1235 N1236                                                    | 用户历史，被` `分割             |
| train.tsv | uid      | str | U1234                                                                | 用户ID，与`user.tsv`一致      |
| train.tsv | nid      | str | N1234                                                                | 新闻ID，与`news.tsv`一致      |
| train.tsv | label    | int | 1                                                                    | 是否点击，0表示未点击，1表示点击       |

我们可以对以上属性进行分类：

| 属性               | 类型  | 预设分词器     | 备注                                  |
|------------------|-----|-----------|-------------------------------------|
| nid, uid, index  | str | IdTok     | 唯一标识                                |
| title, abstract  | str | BertTok   | 指定参数`vocab_dir="bert-base-uncased"` |
| category, subcat | str | EntityTok | 不可分割                                |
| history          | str | SplitTok  | 指定参数`sep=' '`                       |
| label            | int | NumberTok | 指定参数`vocab_size=2`，只有0和1两种情况        |

通过以下代码，我们可以针对每个文件构建一个UniTok对象：

```python
from UniTokv3 import UniTok, Column, Vocab
from UniTokv3.tok import IdTok, BertTok, EntTok, SplitTok, NumberTok

# Create a news id vocab, commonly used in news data, history data, and interaction data.
nid_vocab = Vocab('nid')

# Create a bert tokenizer, commonly used in tokenizing title and abstract.
eng_tok = BertTok(vocab_dir='bert-base-uncased', name='eng')

# Create a news UniTok object.
news_ut = UniTok()

# Add columns to the news UniTok object.
news_ut.add_col(Column(
    # Specify the vocab. The column name will be set to 'nid' automatically if not specified.
    tok=IdTok(vocab=nid_vocab),
)).add_col(Column(
    # The column name will be set to 'title', rather than the name of eng_tok 'eng'.
    name='title',
    tok=eng_tok,
    max_length=20,  # Specify the max length. The exceeding part will be truncated.
)).add_col(Column(
    name='abstract',
    tok=eng_tok,  # Abstract and title use the same tokenizer.
    max_length=30,
)).add_col(Column(
    name='category',
    tok=EntTok,  # Vocab will be created automatically, and the vocab name will be set to 'category'.
)).add_col(Column(
    name='subcat',
    tok=EntTok,  # Vocab will be created automatically, and the vocab name will be set to 'subcat'.
))

# Read the data file.
news_ut.read('news.tsv', sep='\t')

# Tokenize the data.
news_ut.tokenize()

# Store the tokenized data.
news_ut.store('data/news')

# Create a user id vocab, commonly used in user data and interaction data.
uid_vocab = Vocab('uid')  # 在用户数据和交互数据中都会用到

# Create a user UniTok object.
user_ut = UniTok()

# Add columns to the user UniTok object.
user_ut.add_col(Column(
    tok=IdTok(vocab=uid_vocab),
)).add_col(Column(
    name='history',
    tok=SplitTok(sep=' '),  # The news id in the history data is separated by space.
))

# Read the data file.
user_ut.read('user.tsv', sep='\t')

# Tokenize the data.
user_ut.tokenize()

# Store the tokenized data.
user_ut.store('data/user')


def inter_tokenize(mode):
    # Create an interaction UniTok object.
    inter_ut = UniTok()

    # Add columns to the interaction UniTok object.
    inter_ut.add_index_col(
        # The index column in the interaction data is automatically generated, and the tokenizer does not need to be specified.
    ).add_col(Column(
        # Align with the uid column in user_ut.
        tok=EntTok(vocab=uid_vocab),
    )).add_col(Column(
        # Align with the nid column in news_ut.
        tok=EntTok(vocab=nid_vocab),
    )).add_col(Column(
        name='label',
        # The label column in the interaction data only has two values, 0 and 1.
        tok=NumberTok(vocab_size=2),  # NumberTok is supported by UniTok >= 3.0.11.
    ))

    # Read the data file.
    inter_ut.read(f'{mode}.tsv', sep='\t')

    # Tokenize the data.
    inter_ut.tokenize()

    # Store the tokenized data.
    inter_ut.store(mode)


inter_tokenize('data/train')
inter_tokenize('data/dev')
inter_tokenize('data/test')
```

### 3.2 UniDep

UniDep 是一个数据依赖处理类，可以用于加载和访问 UniTok 预处理后的数据。UniDep 包括词汇表（Vocabs），元数据（Meta）等。

`Vocabs` 类是用来集中管理所有的词汇表的。每个 `Vocab` 对象包含了对象到索引的映射，索引到对象的映射，以及一些其它的属性和方法。

`Meta` 类用来管理元数据，包括加载、保存和升级元数据。

以下是一个简单的使用示例：

```python
from UniTokv3 import UniDep

# Load the data.
dep = UniDep('data/news')

# Get sample size.
print(len(dep))

# Get the first sample.
print(dep[0])
```
