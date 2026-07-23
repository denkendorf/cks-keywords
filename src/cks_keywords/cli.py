from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .profiles import FrozenCKS, available_profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cks-keywords", description="Composite Keyword Scoring")
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify-profile", help="Verify a packaged frozen profile")
    verify.add_argument("--profile", default="paper1-w050-s40", choices=available_profiles())

    extract = sub.add_parser("extract", help="Extract keywords with a frozen profile")
    extract.add_argument("input_csv")
    extract.add_argument("output_csv")
    extract.add_argument("--profile", default="paper1-w050-s40", choices=available_profiles())
    extract.add_argument("--id-column", default="record_id")
    extract.add_argument("--text-column", default="abstract")
    extract.add_argument("--title-column", default="title")
    extract.add_argument("--top-n", type=int, default=10)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-profile":
        profile = FrozenCKS.from_profile(args.profile)
        print(json.dumps(profile.manifest, indent=2, ensure_ascii=False))
        return 0 if profile.verify_manifest() else 1
    if args.command == "extract":
        frame = pd.read_csv(args.input_csv)
        profile = FrozenCKS.from_profile(args.profile)
        result = profile.extract_keywords_batch(
            frame,
            id_column=args.id_column,
            text_column=args.text_column,
            title_column=args.title_column if args.title_column in frame.columns else None,
            top_n=args.top_n,
        )
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"Saved {len(result):,} keyword rows to {args.output_csv}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
