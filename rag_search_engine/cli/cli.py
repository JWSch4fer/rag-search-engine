#!/usr/bin/env python3
import argparse


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    return parser


def main() -> None:
    parser = make_parser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ________________________________________________________________________________
    # ____________________global search utilities_____________________________________
    # ________________________________________________________________________________
    build_p = subparsers.add_parser("build", help="Build the inverted index")
    build_p.add_argument(
        "file_path",
        type=str,
        help="Path to source data (default: %(default)s)",
    )
    build_p.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Rebuild cache even if a cached index is present",
    )
    # attach handler
    build_p.set_defaults(func=handle_build)

    # ________________________________________________________________________________
    # ____________________key word search_____________________________________________
    # ________________________________________________________________________________

    args.func(args)


if __name__ == "__main__":
    main()
