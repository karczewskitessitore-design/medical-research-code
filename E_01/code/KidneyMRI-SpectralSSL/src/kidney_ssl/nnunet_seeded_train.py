from __future__ import annotations

import argparse

from nnunetv2.run.run_training import run_training

from kidney_ssl.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seeded wrapper around the official nnU-Net v2 training API"
    )
    parser.add_argument("dataset")
    parser.add_argument("configuration")
    parser.add_argument("fold")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed_everything(args.seed)
    run_training(
        args.dataset,
        args.configuration,
        args.fold,
        trainer_class_name=args.trainer,
    )


if __name__ == "__main__":
    main()
