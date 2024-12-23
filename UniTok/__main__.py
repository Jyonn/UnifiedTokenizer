import argparse
import pickle
import random
from typing import cast, Protocol

import numpy as np
from rich.console import Console
from rich.table import Table

from UniTok import UniDep, Meta, Vocab
from unitokbeta.vocabulary import Vocab as VocabBeta
from unitokbeta.job import Job as JobBeta
from unitokbeta.tokenizer.unknown_tokenizer import UnknownTokenizer
from unitokbeta.unitok import UniTok as UniTokBeta
from unitokbeta.meta import Meta as MetaBeta


class SupportsWrite(Protocol):
    def write(self, __s: bytes) -> object:
        ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str, default='.', help='Path to the UniDep dataset')
    parser.add_argument('--head', type=int, default=0, help='Number of rows to display')
    parser.add_argument('--tail', type=int, default=0, help='Number of rows to display')
    parser.add_argument('--rand', type=int, default=0, help='Number of rows to display')
    parser.add_argument('--cols', type=str, nargs='+', help='Columns to display')

    args = parser.parse_args()
    path = args.path

    depot = UniDep(path, silent=True)

    if not args.head and not args.tail and not args.rand:
        print(depot)
        exit(0)

    # indices = []
    # sample_size = depot.sample_size
    # if args.head:
    #     indices += list(range(args.head))
    # if args.rand:
    #     for _ in range(args.rand):
    #         indices.append(random.randint(0, sample_size - 1))
    # if args.tail:
    #     indices += list(range(sample_size - args.tail, sample_size))
    #
    # cols = args.cols if args.cols else list(depot.cols.keys())
    #
    # table = PrettyTable(cols)
    # for i in indices:
    #     row = [depot[i][col] for col in cols]
    #     table.add_row(row)
    # print(table)
    console = Console()

    # Determine indices
    indices = []
    sample_size = depot.sample_size
    if args.head:
        indices += list(range(args.head))
    if args.rand:
        indices += [random.randint(0, sample_size - 1) for _ in range(args.rand)]
    if args.tail:
        indices += list(range(sample_size - args.tail, sample_size))

    # Determine columns
    cols = args.cols if args.cols else list(depot.cols.keys())

    # Create rich table
    table = Table(title="Sample Data", title_style="bold yellow")

    # Add columns
    for col in cols:
        table.add_column(col, justify="left", style="cyan", no_wrap=True)

    # Add rows
    for i in indices:
        row = [str(depot[i][col]) for col in cols]
        table.add_row(*row)

    # Print the table
    console.print(table)


def upgrade():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str, default='.', help='Path to the UniDep dataset')
    args = parser.parse_args()
    path = args.path

    print(f'Upgrade {path} to {MetaBeta.version}')

    data = np.load(f'{path}/data.npy', allow_pickle=True).item()
    data = cast(dict, data)
    pkl_data = dict()
    for col in data:
        pkl_data[col] = []

        for value in data[col]:
            if isinstance(value, np.ndarray):
                pkl_data[col].append(value.tolist())
            if isinstance(value, (np.int64, np.int32)):
                pkl_data[col].append(int(value))
            else:
                assert isinstance(value, (int, list))
                pkl_data[col].append(value)

    data = pkl_data

    with open(f'{path}/data.pkl', 'wb') as f:
        pickle.dump(data, cast(SupportsWrite, f))

    meta = Meta(store_dir=path)
    meta.load()

    with UniTokBeta() as ut:

        for voc in meta.vocs.values():
            print(f'Upgrade vocabulary {voc.name}')
            vocab = Vocab(name=voc.name).load(path)
            vocab_beta = VocabBeta(name=voc.name)
            vocab_beta.extend(vocab.get_tokens())
            ut.meta.vocabularies.add(vocab_beta)

            for col in voc.cols:
                print(f'\tUpgrade job {col.name}')
                col_data = data[col.name]
                if not len(col_data):
                    print(f'\t\tWarning: empty column {col.name}, defaulting to an atom column')
                    max_len = 0
                    truncate = None
                else:
                    sample = col_data[0]
                    if isinstance(sample, list):
                        print(f'\t\tlist column detected')
                        max_len = col.max_length
                        truncate = 0
                    else:
                        print(f'\t\tatom column detected')
                        max_len = 0
                        truncate = None

                tokenizer = UnknownTokenizer(
                    classname='UnknownTokenizer',
                    tokenizer_id='upgrade_' + col.name,
                    vocab=vocab_beta,
                )
                job = JobBeta(
                    name=col.name,
                    column=col.name,
                    tokenizer=tokenizer,
                    truncate=truncate,
                    order=0,
                    key=meta.id_col == col.name,
                    max_len=max_len,
                )
                ut.meta.tokenizers.add(tokenizer)
                ut.meta.jobs.add(job)

    ut.meta.save(path)

    for vocab in ut.meta.vocabularies:
        vocab.save(path)

    print('Successfully upgraded.')
