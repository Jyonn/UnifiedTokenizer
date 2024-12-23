import argparse

import pandas as pd

from unitokbeta.tokenizer import BaseTokenizer
from unitokbeta.unitok import UniTok
from unitokbeta.utils.class_pool import ClassPool


def integrate():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='.', help='path to a unitok data directory')
    parser.add_argument('--file', '-f', type=str, help='csv, tsv, parquet format data')
    parser.add_argument('--lib', type=str, default=None, help='custom tokenizer library')
    parser.add_argument('--column', '-c', type=str, help='column name to tokenize')
    parser.add_argument('--name', '-n', type=str, help='job name and export column name')
    parser.add_argument('--vocab', '-v', type=str, default=None, help='vocabulary name')
    parser.add_argument('--tokenizer', '-t', type=str, default=None, help='tokenizer classname')
    parser.add_argument('--tokenizer_id', type=str, default=None, help='tokenizer id')
    parser.add_argument('--truncate', type=int, help='truncate length', default=None)
    args, unknown_args = parser.parse_known_args()

    tokenizer_params = dict()
    current_param = None
    for arg in unknown_args:
        if current_param:
            tokenizer_params[current_param] = arg
            current_param = None
        if arg.startswith('--t.'):
            current_param = arg[4:]
        elif arg.startswith('--tokenizer.'):
            current_param = arg[11:]

    if args.file.endswith('.csv') or args.file.endswith('.tsv'):
        df = pd.read_csv(args.file, sep='\t')
    elif args.file.endswith('.parquet'):
        df = pd.read_parquet(args.file)
    else:
        raise ValueError(f'Unsupported file format: {args.file}')

    with UniTok.load(args.path, tokenizer_lib=args.lib) as ut:
        if args.tokenizer_id:
            for t in ut.meta.tokenizers:  # type: BaseTokenizer
                if t.get_tokenizer_id() == args.tokenizer_id:
                    tokenizer = t
                    break
            else:
                raise ValueError(f'Unknown tokenizer id: {args.tokenizer_id}')
        else:
            assert args.tokenizer is not None and args.vocab is not None, 'Tokenizer classname and vocabulary must be specified'
            tokenizers = ClassPool.tokenizers(args.lib)
            assert args.tokenizer in tokenizers, f'Unknown tokenizer: {args.tokenizer}. Available tokenizers: {tokenizers.keys()}'
            tokenizer = tokenizers[args.tokenizer](vocab=args.vocab, **tokenizer_params)

        ut.add_job(tokenizer=tokenizer, column=args.column, name=args.name, truncate=args.truncate)
        ut.tokenize(df).save(args.path)


def summarize():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='.', help='path to a unitok data directory')
    args, _ = parser.parse_known_args()

    with UniTok.load(args.path) as ut:
        ut.summarize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', '-a', type=str, default='summarize', choices=['summarize', 'integrate'])

    args, _ = parser.parse_known_args()
    action = args.action

    if action == 'integrate':
        integrate()
    else:
        summarize()
