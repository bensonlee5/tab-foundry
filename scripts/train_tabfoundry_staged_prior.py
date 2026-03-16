import sys

from tab_foundry.bench.prior_train import main


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if not any(str(arg).startswith("experiment=") for arg in argv):
        argv.append("experiment=cls_benchmark_staged_prior")
    raise SystemExit(main(argv))
