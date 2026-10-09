from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import NoReturn

from app.evaluation.loader import DEFAULT_DATASET, PROJECT_ROOT, load_dataset
from app.evaluation.models import EvaluationError, RunConfig
from app.evaluation.pricing import load_prices
from app.evaluation.reporting import (
    compare_reports,
    read_report,
    write_comparison,
    write_report,
)
from app.evaluation.runner import run_evaluation
from app.evaluation.targets import (
    EXISTING_PROVIDERS,
    FixtureTarget,
    create_live_target,
    require_live,
)


class SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.exit(2, "Invalid evaluation CLI arguments; use --help.\n")


def build_parser() -> argparse.ArgumentParser:
    parser = SafeParser(
        description="AI Core isolated evaluation tools (fixture mode by default)."
    )
    commands = parser.add_subparsers(
        dest="command", required=True, parser_class=SafeParser
    )
    validate = commands.add_parser(
        "validate", help="Validate versioned dataset without network access."
    )
    validate.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run = commands.add_parser(
        "run", help="Evaluate a fixture or explicitly enabled existing provider."
    )
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--fixture", choices=("target-a", "target-b"), default="target-a")
    run.add_argument("--provider", choices=EXISTING_PROVIDERS)
    run.add_argument("--model")
    run.add_argument("--live", action="store_true")
    run.add_argument("--streaming", action="store_true")
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument("--timeout-seconds", type=float, default=60.0)
    run.add_argument("--max-output-chars", type=int, default=100000)
    run.add_argument("--prices", type=Path)
    run.add_argument("--image-mapping", type=Path)
    run.add_argument("--output-dir", type=Path)
    compare = commands.add_parser(
        "compare", help="Compare compatible evaluation reports without network access."
    )
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "eval-results" / "comparison"
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    if args.provider or args.live:
        require_live(args.live)
        if not args.provider:
            raise EvaluationError(
                "Live evaluation requires an existing provider target"
            )
    elif args.model or args.image_mapping:
        raise EvaluationError(
            "Model and image mappings apply only to live provider targets"
        )
    dataset = load_dataset(args.dataset)
    prices = load_prices(args.prices) if args.prices else None
    config = RunConfig(
        streaming=args.streaming,
        repetitions=args.repetitions,
        timeout_seconds=args.timeout_seconds,
        max_output_chars=args.max_output_chars,
    )
    if args.provider:
        mapping = (
            json.loads(args.image_mapping.read_text(encoding="utf-8"))
            if args.image_mapping
            else None
        )
        target = await create_live_target(
            dataset, args.provider, args.model, live=args.live, image_mapping=mapping
        )
        report = await run_evaluation(dataset, target, config, prices=prices)
    else:
        fixture = FixtureTarget.load(dataset, args.fixture)
        report = await run_evaluation(
            dataset, fixture, config, prices=prices, clock=fixture.clock
        )
    directory = (
        args.output_dir or PROJECT_ROOT / "eval-results" / report.metadata.provider
    )
    write_report(report, directory)
    print("Evaluation report saved (" + report.metadata.mode + ").")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            dataset = load_dataset(args.dataset)
            print(
                f"Dataset valid: {dataset.manifest.dataset_version}; "
                f"cases={len(dataset.cases)}; sha256={dataset.hash}"
            )
        elif args.command == "run":
            asyncio.run(_run(args))
        else:
            comparison = compare_reports(
                read_report(args.left), read_report(args.right)
            )
            write_comparison(comparison, args.output_dir)
            print("Compatible evaluation comparison saved.")
        return 0
    except EvaluationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Evaluation cancelled.", file=sys.stderr)
        return 130
    except Exception:
        print(
            "Evaluation operation failed; check input configuration.", file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
