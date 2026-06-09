from argparse import ArgumentParser

import pandas as pd


def get_args():
    parser = ArgumentParser()

    parser.add_argument("--input_dir", type=str, help="input csv")
    parser.add_argument("--output_name", type=str, help="output csv")
    parser.add_argument("--label_column", type=str)
    parser.add_argument("--per_class", type=int, help="num samples per class")
    parser.add_argument("--rs", type=int, help="random state")
    return parser.parse_args()


def main(args):
    df = pd.read_csv(args.input_dir, index_col=False)

    benign = df[df[args.label_column] == 0].sample(
        n=args.per_class, random_state=args.rs
    )
    malicious = df[df[args.label_column] == 1].sample(
        n=args.per_class, random_state=args.rs
    )

    subset = pd.concat([benign, malicious])

    subset.to_csv(args.output_name, index=False)


if __name__ == "__main__":
    args = get_args()
    main(args)
