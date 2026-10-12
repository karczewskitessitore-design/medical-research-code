from __future__ import annotations

import argparse
import json
from pathlib import Path

from kidney_ssl.data.audit import audit_dataset
from kidney_ssl.data.download import download_dataset
from kidney_ssl.data.folds import make_folds
from kidney_ssl.benchmark import run_benchmark
from kidney_ssl.benchmark_v2 import run_benchmark_v2
from kidney_ssl.bilateral_audit import audit_bilateral_manifest
from kidney_ssl.comparison import compare_benchmark
from kidney_ssl.config import load_config
from kidney_ssl.evaluation import evaluate
from kidney_ssl.evaluation_v2 import evaluate_v2
from kidney_ssl.external_data import prepare_external
from kidney_ssl.external_evaluation import evaluate_external
from kidney_ssl.pilot import run_pilot
from kidney_ssl.protocol_v2 import lock_protocol
from kidney_ssl.q2_comparison import compare_q2_gate
from kidney_ssl.training import train
from kidney_ssl.training_v2 import train_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kidney-seg")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download, verify, and safely extract Zenodo data")
    download.add_argument("--data-root", required=True)

    audit = commands.add_parser("audit", help="Create manifest, anomaly log, and SHA-256 inventory")
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--allow-anomalies", action="store_true")

    folds = commands.add_parser("make-folds", help="Create five subject-level folds and nested label subsets")
    folds.add_argument("--manifest", required=True)
    folds.add_argument("--seed", type=int, default=20260802)

    bilateral = commands.add_parser("audit-bilateral", help="Audit RAS left/right kidney separation")
    bilateral.add_argument("--manifest", required=True)
    bilateral.add_argument("--output-dir", required=True)

    training = commands.add_parser("train", help="Train one configuration")
    training.add_argument("--config", required=True)
    training.add_argument("--fold", type=int, default=0)
    training.add_argument("--seed", type=int, default=20260802)

    evaluation = commands.add_parser("evaluate", help="Evaluate the EMA Teacher in original image space")
    evaluation.add_argument("--run-dir", required=True)

    pilot = commands.add_parser("pilot", help="Run B0-B3 sequentially with monitoring")
    pilot.add_argument("--project-root", default=None)
    pilot.add_argument("--fold", type=int, default=0)
    pilot.add_argument("--seed", type=int, default=20260802)

    benchmark = commands.add_parser("benchmark", help="Run the preregistered SCI comparison suite")
    benchmark.add_argument("--suite", required=True)
    benchmark.add_argument(
        "--phase", choices=("dev", "confirmatory", "screen", "internal", "external"), required=True
    )
    benchmark.add_argument("--resume", action="store_true")
    benchmark.add_argument("--project-root", default=None)
    benchmark.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Run only the listed method IDs from one locked v2 suite",
    )

    comparison = commands.add_parser("compare", help="Aggregate a completed confirmatory benchmark")
    comparison.add_argument("--benchmark-dir", required=True)
    comparison.add_argument("--suite", default=None)
    comparison.add_argument("--draws", type=int, default=10000)
    comparison.add_argument("--seed", type=int, default=20260803)
    comparison.add_argument("--q2-gate", action="store_true")

    external = commands.add_parser(
        "prepare-external", help="Download, verify, extract, and map CHAOS or AMOS"
    )
    external.add_argument("--dataset", choices=("chaos", "amos"), required=True)
    external.add_argument("--data-root", required=True)
    external.add_argument("--no-download", action="store_true")

    external_evaluation = commands.add_parser(
        "evaluate-external", help="Run a source-only schema-v2 ensemble on an external dataset"
    )
    external_evaluation.add_argument("--run-dir", required=True)
    external_evaluation.add_argument(
        "--dataset", choices=("chaos", "amos", "chaos_t2_spir", "amos22_mri"), required=True
    )

    locking = commands.add_parser("lock-protocol", help="Freeze the v2 suite/source/data hashes")
    locking.add_argument("--suite", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        print(json.dumps(download_dataset(args.data_root), ensure_ascii=False, indent=2))
    elif args.command == "audit":
        print(
            json.dumps(
                audit_dataset(args.data_root, strict_expected_counts=not args.allow_anomalies),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "make-folds":
        print(make_folds(args.manifest, args.seed))
    elif args.command == "audit-bilateral":
        print(audit_bilateral_manifest(args.manifest, args.output_dir))
    elif args.command == "train":
        config = load_config(args.config)
        if int(config.get("schema_version", 1)) == 2 and str(
            config["model"].get("trainer")
        ) != "nnunetv2":
            print(train_v2(args.config, args.fold, args.seed))
        else:
            print(train(args.config, args.fold, args.seed))
    elif args.command == "evaluate":
        config = load_config(Path(args.run_dir) / "resolved_config.yaml")
        if int(config.get("schema_version", 1)) == 2:
            if str(config["model"].get("trainer")) == "nnunetv2":
                print(Path(args.run_dir) / "evaluation_summary.json")
            else:
                print(evaluate_v2(args.run_dir))
        else:
            print(evaluate(args.run_dir))
    elif args.command == "pilot":
        print(run_pilot(args.project_root, args.fold, args.seed))
    elif args.command == "benchmark":
        if args.phase in {"screen", "internal", "external"}:
            print(
                run_benchmark_v2(
                    args.suite,
                    args.phase,
                    args.resume,
                    args.project_root,
                    args.methods,
                )
            )
        else:
            print(run_benchmark(args.suite, args.phase, args.resume, args.project_root))
    elif args.command == "compare":
        if args.q2_gate:
            print(compare_q2_gate(args.benchmark_dir, args.suite, args.seed))
        else:
            print(compare_benchmark(args.benchmark_dir, args.suite, args.draws, args.seed))
    elif args.command == "prepare-external":
        print(prepare_external(args.dataset, args.data_root, not args.no_download))
    elif args.command == "evaluate-external":
        print(evaluate_external(args.run_dir, args.dataset))
    elif args.command == "lock-protocol":
        print(lock_protocol(args.suite))


if __name__ == "__main__":
    main()
