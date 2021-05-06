import argparse
import json

from ffm import Dataset, train


def ffm_train() -> None:
    parser = argparse.ArgumentParser(description="LibFFM CLI")
    parser.add_argument("tr_path", help="File path to training set", type=str)
    parser.add_argument(
        "-p", help="Set path to the validation set", type=str, default=""
    )
    parser.add_argument(
        "-W", help="Set path of importance weights file for training set", default=""
    )
    parser.add_argument(
        "-WV", help="Set path of importance weights file for validation set", default=""
    )
    parser.add_argument("-f", help="Set path for production model file", default="")
    parser.add_argument("-m", help="Set key prefix for production model", default="")
    parser.add_argument(
        "--json-meta", help="Generate a meta file if sets json file path", default=""
    )

    parser.add_argument(
        "-l", help="Set regularization parameter (lambda)", type=float, default=0.00002
    )
    parser.add_argument("-k", help="Set number of latent factors", type=int, default=4)
    parser.add_argument("-t", help="Set number of iterations", type=int, default=15)
    parser.add_argument("-r", help="Set learning rate (eta)", type=float, default=0.2)
    parser.add_argument("-s", help="Set number of threads", type=int, default=1)
    parser.add_argument(
        "--no-norm", help="Disable instance-wise normalization", action="store_true"
    )
    parser.add_argument(
        "--no-rand",
        help="Disable random update <training_set>.bin will be generated",
        action="store_true",
    )
    parser.add_argument(
        "--auto-stop",
        help="Stop at the iteration that achieves the best validation loss (must be used with -p)",
        action="store_true",
    )
    parser.add_argument(
        "--auto-stop-threshold",
        help="Set the threshold count for stop at the iteration that achieves the best validation loss (must be used with --auto-stop)",
        type=int,
        default=-1,
    )
    parser.add_argument("--quiet", "-q", help="quiet", action="store_true")
    args = parser.parse_args()

    train_data = Dataset.read_ffm_data(args.tr_path, weights_path=args.W)
    valid_data = Dataset.read_ffm_data(args.p, weights_path=args.WV)
    model = train(
        train_data=train_data,
        valid_data=valid_data,
        eta=args.r,
        lam=args.l,
        nr_iters=args.t,
        k=args.k,
        nr_threads=args.s,
        auto_stop_threshold=args.auto_stop_threshold,
        quiet=args.quiet,
        normalization=not args.no_norm,
        random=not args.no_rand,
        auto_stop=args.auto_stop,
    )

    if args.f:
        with open(args.f, "w") as f:
            model.dump_libffm_weights(f, key_prefix=args.m)

    if args.json_meta:
        with open(args.json_meta, "w") as f:
            json.dump({"best_iteration": model.best_iteration}, f)


if __name__ == "__main__":
    ffm_train()
