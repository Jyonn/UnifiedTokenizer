import argparse
import random

from prettytable import PrettyTable

from UniTok import UniDep


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

    indices = []
    sample_size = depot.sample_size
    if args.head:
        indices += list(range(args.head))
    if args.rand:
        for _ in range(args.rand):
            indices.append(random.randint(0, sample_size - 1))
    if args.tail:
        indices += list(range(sample_size - args.tail, sample_size))

    cols = args.cols if args.cols else list(depot.cols.keys())

    table = PrettyTable(cols)
    for i in indices:
        row = [depot[i][col] for col in cols]
        table.add_row(row)
    print(table)
