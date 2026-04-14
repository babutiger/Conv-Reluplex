#!/usr/bin/env python3
"""Check whether the local environment matches the repository expectations."""

from pathlib import Path
import os
import sys


def repo_root():
    configured = os.environ.get("CONV_RELUPLEX_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


ROOT = repo_root()
MNIST_TRAIN = Path(
    os.environ.get("CONV_RELUPLEX_MNIST_TRAIN_DIR", ROOT / "data_set" / "mnist_data" / "train")
).expanduser()
MNIST_TEST = Path(
    os.environ.get("CONV_RELUPLEX_MNIST_TEST_DIR", ROOT / "data_set" / "mnist_data" / "test")
).expanduser()
ADV_TRAIN = Path(
    os.environ.get("CONV_RELUPLEX_ADV_TRAIN_DIR", ROOT / "adversarial_train_data_train")
).expanduser()
ADV_TEST = Path(
    os.environ.get("CONV_RELUPLEX_ADV_TEST_DIR", ROOT / "adversarial_train_data_test")
).expanduser()


def check_dir(path):
    return path.exists() and path.is_dir()


def check_file(path):
    return path.exists() and path.is_file()


def print_status(level, label, path):
    print("[{}] {}: {}".format(level, label, path))


def main():
    failed = False

    print_status("OK" if check_dir(ROOT) else "FAIL", "Repository root", ROOT)

    legacy_pkg = ROOT / "mycode" / "mnist_all_minish_one_map_9_9" / "__init__.py"
    legacy_ok = check_file(legacy_pkg)
    print_status("OK" if legacy_ok else "FAIL", "Legacy import compatibility package", legacy_pkg)
    failed = failed or (not legacy_ok)

    model_files = [
        ROOT / "model_9_9" / "model.ckpt.meta",
        ROOT / "model_9_9" / "model.ckpt.index",
        ROOT / "model_9_9" / "model.ckpt.data-00000-of-00001",
    ]
    missing_model_files = [path for path in model_files if not check_file(path)]
    if missing_model_files:
        for path in missing_model_files:
            print_status("FAIL", "Missing baseline model file", path)
        failed = True
    else:
        print_status("OK", "Baseline checkpoint", ROOT / "model_9_9")

    if check_dir(MNIST_TRAIN):
        print_status("OK", "MNIST train directory", MNIST_TRAIN)
    else:
        print_status("WARN", "MNIST train directory missing", MNIST_TRAIN)

    if check_dir(MNIST_TEST):
        print_status("OK", "MNIST test directory", MNIST_TEST)
    else:
        print_status("WARN", "MNIST test directory missing", MNIST_TEST)

    if check_dir(ADV_TRAIN):
        print_status("OK", "Adversarial train data directory", ADV_TRAIN)
    else:
        print_status("WARN", "Adversarial train data directory missing", ADV_TRAIN)

    if check_dir(ADV_TEST):
        print_status("OK", "Adversarial test data directory", ADV_TEST)
    else:
        print_status("WARN", "Adversarial test data directory missing", ADV_TEST)

    print("")
    print("Environment variables you can use to override paths:")
    print("  CONV_RELUPLEX_ROOT")
    print("  CONV_RELUPLEX_MNIST_TRAIN_DIR")
    print("  CONV_RELUPLEX_MNIST_TEST_DIR")
    print("  CONV_RELUPLEX_ADV_TRAIN_DIR")
    print("  CONV_RELUPLEX_ADV_TEST_DIR")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
