"""Command line entry point.

python -m flood run --region lagos                 run every stage that is out of date
python -m flood run --region lagos --stage hazard  run one stage (repeat for more)
python -m flood run --region lagos --force         rerun even if up to date
python -m flood status --region lagos              show which stages are up to date
python -m flood stages                             list the stages in run order
python -m flood fetch-boundaries --region lagos    download app admin unit outlines
"""

from __future__ import annotations

import argparse
import logging
import sys

from flood.config import ConfigError, MissingDataError, load_region


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flood",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def region_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--region", required=True, help="region name or path to a region yaml")
        p.add_argument("--data-dir", help="raw data folder (default: $FLOOD_DATA_DIR)")
        p.add_argument(
            "--output-dir", help="output folder (default: $FLOOD_OUTPUT_DIR or outputs/)"
        )

    run = sub.add_parser("run", help="run the pipeline")
    region_args(run)
    run.add_argument(
        "--stage",
        action="append",
        dest="stages",
        metavar="NAME",
        help="run only this stage (can be repeated)",
    )
    run.add_argument("--force", action="store_true", help="rerun stages even if up to date")

    status = sub.add_parser("status", help="show which stages are up to date")
    region_args(status)

    fetch = sub.add_parser(
        "fetch-boundaries", help="download redistributable admin unit outlines for the app"
    )
    region_args(fetch)

    sub.add_parser("stages", help="list the stages in run order")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from flood.stages import all_stages, run_pipeline

    if args.command == "stages":
        for stage in all_stages():
            print(f"{stage.name:<12} {stage.description}")
        return 0

    try:
        region = load_region(args.region, args.data_dir, args.output_dir)
        if args.command == "status":
            for stage in all_stages():
                fresh, reason = stage.is_up_to_date(region)
                print(f"{stage.name:<12} {'up to date' if fresh else reason}")
            return 0
        if args.command == "fetch-boundaries":
            from flood.boundaries import fetch

            fetch(region)
            return 0
        run_pipeline(region, args.stages, force=args.force)
    except (ConfigError, MissingDataError, KeyError) as exc:
        logging.getLogger("flood").error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
