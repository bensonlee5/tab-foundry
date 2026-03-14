import sys

from tab_foundry.bench.prior_train import main


if __name__ == "__main__":
    raise SystemExit(main(["experiment=cls_benchmark_staged_prior", *sys.argv[1:]]))
