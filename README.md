# UniTok V3.0

## 介绍

UniTok是一个面向机器学习的统一文本数据预处理工具。它提供了一系列预定义的分词器，以便于处理不同类型的文本数据。UniTok简单易上手，让算法工程师能更专注优化算法本身，大大降低了数据预处理的难度。
UniDep是UniTok预处理后数据的解析工具，能和PyTorch的Dataset类配合使用。

## 安装

`pip install unitok>=3.0.11`

## 使用

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
from UniTok import UniTok, Column, Vocab
from UniTok.tok import IdTok, BertTok, EntTok, SplitTok, NumberTok

nid_vocab = Vocab('nid')  # 在新闻数据、历史数据和交互数据中都会用到
eng_tok = BertTok(vocab_dir='bert-base-uncased', name='eng')  # 用于处理英文文本

news_ut = UniTok().add_col(Column(
    name='nid',  # 数据栏名称，如果和tok的name一致，可以省略
    tok=IdTok(vocab=nid_vocab),  # 指定分词器，每个UniTok对象必须有且只能有一个IdTok
)).add_col(Column(
    name='title',
    tok=eng_tok,
    max_length=20,  # 指定最大长度，超过的部分会被截断
)).add_col(Column(
    name='abstract',
    tok=eng_tok,  # 摘要和标题使用同一个分词器
    max_length=30,  # 指定最大长度，超过的部分会被截断
)).add_col(Column(
    name='category',
    tok=EntTok(name='cat'),  # 不显式指定Vocab，会根据name自动创建
)).add_col(Column(
    name='subcat',
    tok=EntTok(name='subcat'),
))

news_ut.read('news.tsv', sep='\t')  # 读取数据文件
news_ut.tokenize()  # 运行分词
news_ut.store('data/news')  # 保存分词结果

uid_vocab = Vocab('uid')  # 在用户数据和交互数据中都会用到

user_ut = UniTok().add_col(Column(
    name='uid',
    tok=IdTok(vocab=uid_vocab),
)).add_col(Column(
    name='history',
    tok=SplitTok(sep=' '),  # 历史数据中的新闻ID用空格分割
))

user_ut.read('user.tsv', sep='\t')  # 读取数据文件
user_ut.tokenize()  # 运行分词
user_ut.store('data/user')  # 保存分词结果


def inter_tokenize(mode):
    # 由于train/dev/test的index不同，每次预处理前都需要重新构建UniTok对象
    # 如果不重新构建，index词表可能不准确，导致元数据和真实数据不一致
    # 但通过UniDep解析数据后，能修正index的误差
    
    inter_ut = UniTok().add_index_col(
        # 交互数据中的index列是自动生成的，不需要指定分词器
    ).add_col(Column(
        name='uid',
        tok=IdTok(vocab=uid_vocab),  # 指定和user_ut中的uid列一致
    )).add_col(Column(
        name='nid',
        tok=IdTok(vocab=nid_vocab),  # 指定和news_ut中的nid列一致
    )).add_col(Column(
        name='label',
        tok=NumberTok(vocab_size=2),  # 0和1两种情况，>=3.0.11版本支持
    ))

    inter_ut.read(f'{mode}.tsv', sep='\t')  # 读取数据文件
    inter_ut.tokenize()  # 运行分词
    inter_ut.store(mode)  # 保存分词结果

    
inter_tokenize('data/train')
inter_tokenize('data/dev')
inter_tokenize('data/test')
```

我们可以用UniDep解析数据：

```python
from UniTok import UniDep

news_dep = UniDep('data/news')  # 读取分词结果
print(len(news_dep))
print(news_dep[0])
```
